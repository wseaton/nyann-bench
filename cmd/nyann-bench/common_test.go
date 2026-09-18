package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neuralmagic/nyann-bench/pkg/config"
	"github.com/neuralmagic/nyann-bench/pkg/dataset"
	"github.com/neuralmagic/nyann-bench/pkg/recorder"
)

func startTestServer(t *testing.T) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"data":[{"id":"test-model"}]}`)
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"count":10}`)
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"text\":\"The answer is 42. #### 42\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":5,\"total_tokens\":5}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return strings.TrimPrefix(server.URL, "http://")
}

// TestFiniteDatasetRunsFullDuration verifies that a finite eval dataset
// with MaxRequests=0 runs for the full stage duration instead of stopping
// after exhausting the dataset. This is the regression from PR #47 where
// auto-set MaxRequests caused multi-stage benchmarks to stop early.
func TestFiniteDatasetRunsFullDuration(t *testing.T) {
	addr := startTestServer(t)

	dir := t.TempDir()
	testPath := filepath.Join(dir, "gsm8k_test.jsonl")
	items := `{"question":"What is 1+1?","answer":"1+1=2\n#### 2"}
{"question":"What is 2+2?","answer":"2+2=4\n#### 4"}
{"question":"What is 3+3?","answer":"3+3=6\n#### 6"}
`
	if err := os.WriteFile(testPath, []byte(items), 0644); err != nil {
		t.Fatal(err)
	}

	ds, err := dataset.NewGSM8K(testPath, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	sc := &config.ScenarioConfig{
		Target: "http://" + addr + "/v1",
		Model:  "test-model",
		Workload: config.Workload{
			Type:      "gsm8k",
			GSM8KPath: testPath,
		},
		Stages: []config.ScenarioStage{{
			Name:        "bench-stage",
			Duration:    2 * time.Second,
			Mode:        "concurrent",
			Concurrency: 4,
			MaxRequests: 0, // unlimited — should run for full duration
		}},
	}

	var summaryRequests int
	// Race-enabled package tests can briefly starve this process while other
	// packages compile and run. Keep the normal attempt fast, then retry once
	// with a generous bounded window without weakening the wraparound check.
	for attempt, duration := range []time.Duration{2 * time.Second, 10 * time.Second} {
		sc.Stages[0].Duration = duration
		ctx, cancel := context.WithCancel(context.Background())
		summary, err := runScenario(ctx, cancel, scenarioOpts{
			Target:   "http://" + addr + "/v1",
			Model:    "test-model",
			Scenario: sc,
			Dataset:  ds,
		})
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		summaryRequests = summary.TotalRequests
		if summaryRequests > 3 {
			break
		}
		if attempt == 0 {
			t.Logf("only %d requests completed in %s under contention; retrying with %s", summaryRequests, duration, 10*time.Second)
		}
	}

	// With 3 items and 4 concurrent streams in a timed unlimited stage, the
	// dataset must wrap and produce more requests than its finite length.
	if summaryRequests <= 3 {
		t.Fatalf("expected more than 3 requests (dataset should wrap around), got %d", summaryRequests)
	}
}

// TestExplicitMaxRequestsStopsEarly verifies that setting MaxRequests
// explicitly on a stage stops it after the configured number of requests.
func TestExplicitMaxRequestsStopsEarly(t *testing.T) {
	addr := startTestServer(t)

	dir := t.TempDir()
	testPath := filepath.Join(dir, "gsm8k_test.jsonl")
	items := `{"question":"What is 1+1?","answer":"1+1=2\n#### 2"}
{"question":"What is 2+2?","answer":"2+2=4\n#### 4"}
{"question":"What is 3+3?","answer":"3+3=6\n#### 6"}
{"question":"What is 4+4?","answer":"4+4=8\n#### 8"}
{"question":"What is 5+5?","answer":"5+5=10\n#### 10"}
`
	if err := os.WriteFile(testPath, []byte(items), 0644); err != nil {
		t.Fatal(err)
	}

	ds, err := dataset.NewGSM8K(testPath, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	sc := &config.ScenarioConfig{
		Target: "http://" + addr + "/v1",
		Model:  "test-model",
		Workload: config.Workload{
			Type:      "gsm8k",
			GSM8KPath: testPath,
		},
		Stages: []config.ScenarioStage{{
			Name:        "gsm8k-eval",
			Duration:    30 * time.Second,
			Mode:        "concurrent",
			Concurrency: 16,
			MaxRequests: 5,
		}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	summary, err := runScenario(ctx, cancel, scenarioOpts{
		Target:   "http://" + addr + "/v1",
		Model:    "test-model",
		Scenario: sc,
		Dataset:  ds,
	})
	if err != nil {
		t.Fatal(err)
	}

	if summary.TotalRequests != 5 {
		t.Fatalf("expected exactly 5 requests, got %d", summary.TotalRequests)
	}
}

// sessionTestServer serves /v1/models and streams one-token chat replies
// carrying cached-token usage and an X-Upstream-Host header. It returns the
// session header value of every chat request, in arrival order.
func sessionTestServer(t *testing.T) (string, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var sessions []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			fmt.Fprint(w, `{"data":[{"id":"test-model"}]}`)
			return
		}
		mu.Lock()
		sessions = append(sessions, r.Header.Get("X-Session-Id"))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Upstream-Host", "10.0.0.7:8000")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"total_tokens\":21,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), sessions...)
	}
}

func readRequestRecords(t *testing.T, path string) []recorder.Record {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var records []recorder.Record
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var r recorder.Record
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatal(err)
		}
		records = append(records, r)
	}
	return records
}

func TestScenarioSessionWorkloadEndToEnd(t *testing.T) {
	url, _ := sessionTestServer(t)
	outDir := t.TempDir()
	sc := &config.ScenarioConfig{
		Workload: config.Workload{
			Type:          "synthetic",
			ISL:           16,
			OSL:           4,
			Turns:         3,
			CharsPerToken: 4,
			SessionHeader: "X-Session-Id",
			ThinkTime:     &config.ThinkTime{Median: config.Duration(100 * time.Millisecond)},
			RecordHeaders: []string{"X-Upstream-Host"},
		},
		Stages: []config.ScenarioStage{{
			Duration: time.Second,
			Mode:     "poisson",
			Rate:     10,
		}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := runScenario(ctx, cancel, scenarioOpts{
		Target:    url + "/v1",
		Model:     "test-model",
		Scenario:  sc,
		OutputDir: outDir,
	}); err != nil {
		t.Fatal(err)
	}

	records := readRequestRecords(t, filepath.Join(outDir, "requests_0.jsonl"))
	end := map[string]float64{}
	for _, r := range records {
		end[r.RequestID] = r.EndTime
	}
	sessions := map[string]string{}
	laterTurns := 0
	for _, r := range records {
		if r.SessionID == "" {
			t.Fatalf("record %s has no session_id", r.RequestID)
		}
		if prev, ok := sessions[r.ConversationID]; ok && prev != r.SessionID {
			t.Fatalf("conversation %s changed session id", r.ConversationID)
		}
		sessions[r.ConversationID] = r.SessionID
		if r.CachedTokens == nil || *r.CachedTokens != 7 {
			t.Fatalf("record %s: cached_tokens %v, want 7", r.RequestID, r.CachedTokens)
		}
		if r.Headers["X-Upstream-Host"] != "10.0.0.7:8000" {
			t.Fatalf("record %s: headers %v", r.RequestID, r.Headers)
		}
		if r.Turn > 0 {
			prevEnd, ok := end[fmt.Sprintf("%s-t%d", r.ConversationID, r.Turn-1)]
			if !ok {
				t.Fatalf("record %s has no previous turn", r.RequestID)
			}
			if gap := r.StartTime - prevEnd; gap < 0.1 {
				t.Fatalf("record %s started %.3fs after the previous turn, want >= 0.1s", r.RequestID, gap)
			}
			laterTurns++
		}
	}
	if len(sessions) < 3 || laterTurns == 0 {
		t.Fatalf("expected several multi-turn sessions, got %d sessions and %d later turns", len(sessions), laterTurns)
	}
}

func TestStagesWithDifferentSessionWorkloadsRunSeparately(t *testing.T) {
	url, sessions := sessionTestServer(t)
	plain := &config.Workload{Type: "synthetic", ISL: 16, OSL: 4, Turns: 1, CharsPerToken: 4}
	withSession := *plain
	withSession.SessionHeader = "X-Session-Id"
	sc := &config.ScenarioConfig{
		Workload: *plain,
		Stages: []config.ScenarioStage{
			{Duration: 300 * time.Millisecond, Mode: "concurrent", Concurrency: 1, Workload: plain},
			{Duration: 300 * time.Millisecond, Mode: "concurrent", Concurrency: 1, Workload: &withSession},
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := runScenario(ctx, cancel, scenarioOpts{
		Target:   url + "/v1",
		Model:    "test-model",
		Scenario: sc,
	}); err != nil {
		t.Fatal(err)
	}

	var with, without int
	for _, s := range sessions() {
		if s == "" {
			without++
		} else {
			with++
		}
	}
	if with == 0 || without == 0 {
		t.Fatalf("expected requests from both stages, got %d with a session id and %d without", with, without)
	}
}

// tokenizeHandler answers /tokenize with a whitespace word count and records
// the model named in each call.
func tokenizeHandler(t *testing.T) (http.HandlerFunc, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var models []string
	h := func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model  string `json:"model"`
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		models = append(models, body.Model)
		mu.Unlock()
		fmt.Fprintf(w, `{"count":%d}`, max(1, len(strings.Fields(body.Prompt))))
	}
	return h, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), models...)
	}
}

func tokenizeServer(t *testing.T) (string, func() []string) {
	t.Helper()
	h, models := tokenizeHandler(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			http.NotFound(w, r)
			return
		}
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv.URL, models
}

func TestTokenizerTargetKeepsTokenizationOffTheInferenceTarget(t *testing.T) {
	var targetTokenizeCalls atomic.Int64
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"adapter-3"}]}`)
		case "/tokenize":
			targetTokenizeCalls.Add(1)
			http.Error(w, "tokenize must not reach the inference target", http.StatusTeapot)
		default:
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	t.Cleanup(target.Close)
	tokURL, tokModels := tokenizeServer(t)

	sc := &config.ScenarioConfig{
		Workload: config.Workload{Type: "synthetic", ISL: 64, OSL: 4, Turns: 2},
		Stages:   []config.ScenarioStage{{Duration: 300 * time.Millisecond, Mode: "concurrent", Concurrency: 1}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	summary, err := runScenario(ctx, cancel, scenarioOpts{
		Target:          target.URL + "/v1",
		Model:           "adapter-3",
		Scenario:        sc,
		TokenizerTarget: tokURL + "/v1",
		TokenizerModel:  "base-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	if summary.TotalRequests == 0 {
		t.Fatal("no requests ran")
	}
	if n := targetTokenizeCalls.Load(); n != 0 {
		t.Fatalf("inference target received %d /tokenize calls", n)
	}
	models := tokModels()
	if len(models) == 0 {
		t.Fatal("tokenizer target received no /tokenize calls")
	}
	for _, m := range models {
		if m != "base-model" {
			t.Fatalf("tokenizer call used model %q, want base-model", m)
		}
	}
}

func TestTokenizationDefaultsToTargetAndModel(t *testing.T) {
	tokenize, tokModels := tokenizeHandler(t)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"adapter-3"}]}`)
		case "/tokenize":
			tokenize(w, r)
		default:
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	t.Cleanup(target.Close)

	sc := &config.ScenarioConfig{
		Workload: config.Workload{Type: "synthetic", ISL: 64, OSL: 4, Turns: 1},
		Stages:   []config.ScenarioStage{{Duration: 200 * time.Millisecond, Mode: "concurrent", Concurrency: 1}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := runScenario(ctx, cancel, scenarioOpts{
		Target:   target.URL + "/v1",
		Model:    "adapter-3",
		Scenario: sc,
	}); err != nil {
		t.Fatal(err)
	}
	models := tokModels()
	if len(models) == 0 {
		t.Fatal("no /tokenize calls reached the target")
	}
	for _, m := range models {
		if m != "adapter-3" {
			t.Fatalf("tokenizer call used model %q, want the request model adapter-3", m)
		}
	}
}

func thinkGapsForSeed(t *testing.T, seed int64) map[string]float64 {
	t.Helper()
	url, _ := sessionTestServer(t)
	outDir := t.TempDir()
	sc := &config.ScenarioConfig{
		Workload: config.Workload{
			Type: "synthetic", ISL: 16, OSL: 4, Turns: 3, CharsPerToken: 4,
			ThinkTime: &config.ThinkTime{Median: config.Duration(80 * time.Millisecond), Sigma: 0.8, Max: config.Duration(400 * time.Millisecond)},
		},
		Stages: []config.ScenarioStage{{Duration: time.Second, Mode: "poisson", Rate: 10}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := runScenario(ctx, cancel, scenarioOpts{
		Target: url + "/v1", Model: "test-model", Scenario: sc, OutputDir: outDir, Seed: seed,
	}); err != nil {
		t.Fatal(err)
	}
	records := readRequestRecords(t, filepath.Join(outDir, "requests_0.jsonl"))
	end := map[string]float64{}
	for _, r := range records {
		end[r.RequestID] = r.EndTime
	}
	gaps := map[string]float64{}
	for _, r := range records {
		if prev, ok := end[fmt.Sprintf("%s-t%d", r.ConversationID, r.Turn-1)]; ok && r.Turn > 0 {
			gaps[r.RequestID] = r.StartTime - prev
		}
	}
	return gaps
}

func TestScenarioSeedReplaysThinkTimes(t *testing.T) {
	const tolerance = 0.03
	a, b := thinkGapsForSeed(t, 5), thinkGapsForSeed(t, 5)
	compared, differs := 0, false
	for id, ga := range a {
		gb, ok := b[id]
		if !ok {
			continue
		}
		compared++
		if d := ga - gb; d > tolerance || d < -tolerance {
			t.Fatalf("%s paused %.3fs and %.3fs with the same seed", id, ga, gb)
		}
	}
	if compared < 5 {
		t.Fatalf("only %d think gaps in common", compared)
	}
	for id, ga := range thinkGapsForSeed(t, 6) {
		if gb, ok := a[id]; ok && (ga-gb > tolerance || gb-ga > tolerance) {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("a different seed replayed the same think gaps")
	}
}
