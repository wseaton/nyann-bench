package loadgen_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/neuralmagic/nyann-bench/pkg/config"
	"github.com/neuralmagic/nyann-bench/pkg/dataset"
	"github.com/neuralmagic/nyann-bench/pkg/loadgen"
	"github.com/neuralmagic/nyann-bench/pkg/recorder"
)

const sessionHeader = "X-Session-Id"

type seenRequest struct {
	requestID string
	sessionID string
	hasHeader bool
}

// sessionServer records the request id and session header of every chat
// request and streams a one-token reply whose usage reports 7 cached prompt
// tokens, with an X-Upstream-Host and an X-Other response header.
func sessionServer(t *testing.T) (string, func() []seenRequest) {
	t.Helper()
	var mu sync.Mutex
	var seen []seenRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hasHeader := r.Header[http.CanonicalHeaderKey(sessionHeader)]
		mu.Lock()
		seen = append(seen, seenRequest{
			requestID: r.Header.Get("X-Request-Id"),
			sessionID: r.Header.Get(sessionHeader),
			hasHeader: hasHeader,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Upstream-Host", "10.0.0.7:8000")
		w.Header().Set("X-Other", "unrecorded")
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`)
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21,"prompt_tokens_details":{"cached_tokens":7}}}`)
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() []seenRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]seenRequest(nil), seen...)
	}
}

// convOf extracts the conversation id from an X-Request-Id of the form
// "<model>|<conv>-t<turn>".
func convOf(t *testing.T, requestID string) string {
	t.Helper()
	_, rest, ok := strings.Cut(requestID, "|")
	if !ok {
		t.Fatalf("malformed X-Request-Id %q", requestID)
	}
	i := strings.LastIndex(rest, "-t")
	if i < 0 {
		t.Fatalf("malformed X-Request-Id %q", requestID)
	}
	return rest[:i]
}

func runGenerator(t *testing.T, gen *loadgen.Generator) []recorder.Record {
	t.Helper()
	outDir := t.TempDir()
	rec, err := recorder.New(outDir, 0)
	if err != nil {
		t.Fatal(err)
	}
	gen.Recorder = rec
	if _, err := gen.Run(context.Background()); err != nil {
		t.Fatalf("generator run failed: %v", err)
	}
	rec.Close()
	return readRecords(t, filepath.Join(outDir, "requests_0.jsonl"))
}

func TestSessionHeaderStablePerConversation(t *testing.T) {
	for _, tc := range []struct {
		name string
		gen  func() *loadgen.Generator
	}{
		{"concurrent", func() *loadgen.Generator {
			return &loadgen.Generator{Mode: loadgen.ModeConcurrent, Concurrency: 3}
		}},
		{"poisson", func() *loadgen.Generator {
			return &loadgen.Generator{Mode: loadgen.ModePoisson, Rate: 40}
		}},
		{"conversation_pool", func() *loadgen.Generator {
			return &loadgen.Generator{Mode: loadgen.ModeConversationPool, Concurrency: 2, ConversationPoolSize: 4}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, requests := sessionServer(t)
			gen := tc.gen()
			gen.Target = url + "/v1"
			gen.Model = "test-model"
			gen.Duration = 400 * time.Millisecond
			gen.Dataset = dataset.NewSynthetic(16, 4, 3, 4.0)
			gen.SessionHeader = sessionHeader
			records := runGenerator(t, gen)

			sessionByConv := map[string]string{}
			convBySession := map[string]string{}
			turnsByConv := map[string]int{}
			for _, r := range requests() {
				if r.sessionID == "" {
					t.Fatalf("request %s carried no session id", r.requestID)
				}
				conv := convOf(t, r.requestID)
				turnsByConv[conv]++
				if prev, ok := sessionByConv[conv]; ok && prev != r.sessionID {
					t.Fatalf("conversation %s changed session id from %s to %s", conv, prev, r.sessionID)
				}
				sessionByConv[conv] = r.sessionID
				if prev, ok := convBySession[r.sessionID]; ok && prev != conv {
					t.Fatalf("session id %s shared by conversations %s and %s", r.sessionID, prev, conv)
				}
				convBySession[r.sessionID] = conv
			}
			multiTurn := 0
			for _, n := range turnsByConv {
				if n > 1 {
					multiTurn++
				}
			}
			if len(sessionByConv) < 2 || multiTurn == 0 {
				t.Fatalf("expected several conversations with more than one turn, got %v", turnsByConv)
			}

			if len(records) == 0 {
				t.Fatal("no records written")
			}
			for _, rec := range records {
				if want := sessionByConv[rec.ConversationID]; rec.SessionID != want {
					t.Fatalf("record %s: session_id %q, want %q", rec.RequestID, rec.SessionID, want)
				}
			}
		})
	}
}

func TestNoSessionHeaderByDefault(t *testing.T) {
	url, requests := sessionServer(t)
	records := runGenerator(t, &loadgen.Generator{
		Target:      url + "/v1",
		Model:       "test-model",
		Concurrency: 1,
		Duration:    200 * time.Millisecond,
		Dataset:     dataset.NewSynthetic(16, 4, 2, 4.0),
	})
	for _, r := range requests() {
		if r.hasHeader {
			t.Fatalf("request %s carried a session header with none configured", r.requestID)
		}
	}
	for _, rec := range records {
		if rec.SessionID != "" {
			t.Fatalf("record %s: session_id %q, want empty", rec.RequestID, rec.SessionID)
		}
	}
}

func TestThinkTimeSeparatesTurnsOnly(t *testing.T) {
	url, _ := sessionServer(t)
	const pause = 150 * time.Millisecond
	records := runGenerator(t, &loadgen.Generator{
		Target:      url + "/v1",
		Model:       "test-model",
		Concurrency: 1,
		Duration:    1500 * time.Millisecond,
		Dataset:     dataset.NewSynthetic(16, 4, 3, 4.0),
		ThinkTime:   &config.ThinkTime{Median: config.Duration(pause)},
	})

	byID := map[string]recorder.Record{}
	for _, rec := range records {
		byID[rec.RequestID] = rec
	}
	var turnGaps, convGaps int
	for _, rec := range records {
		if rec.Turn > 0 {
			prev, ok := byID[fmt.Sprintf("%s-t%d", rec.ConversationID, rec.Turn-1)]
			if !ok {
				t.Fatalf("record %s has no previous turn", rec.RequestID)
			}
			gap := time.Duration((rec.StartTime - prev.EndTime) * float64(time.Second))
			if gap < pause {
				t.Fatalf("record %s started %s after the previous turn, want >= %s", rec.RequestID, gap, pause)
			}
			turnGaps++
			continue
		}
		// The stream's previous conversation ended on its last turn; a new
		// conversation starts without a pause.
		var idx int
		if _, err := fmt.Sscanf(rec.ConversationID, "w0-c%d", &idx); err != nil {
			t.Fatalf("unexpected conversation id %q: %v", rec.ConversationID, err)
		}
		if idx == 0 {
			continue
		}
		last, ok := byID[fmt.Sprintf("w0-c%d-t2", idx-1)]
		if !ok {
			t.Fatalf("conversation w0-c%d has no last turn", idx-1)
		}
		gap := time.Duration((rec.StartTime - last.EndTime) * float64(time.Second))
		if gap >= pause {
			t.Fatalf("conversation %s started %s after the previous one, want no think time", rec.ConversationID, gap)
		}
		convGaps++
	}
	if turnGaps < 2 || convGaps < 1 {
		t.Fatalf("expected gaps between turns and between conversations, got %d and %d", turnGaps, convGaps)
	}
}

func TestThinkTimeEndsWithRun(t *testing.T) {
	url, _ := sessionServer(t)
	start := time.Now()
	records := runGenerator(t, &loadgen.Generator{
		Target:      url + "/v1",
		Model:       "test-model",
		Concurrency: 2,
		Duration:    300 * time.Millisecond,
		Dataset:     dataset.NewSynthetic(16, 4, 2, 4.0),
		ThinkTime:   &config.ThinkTime{Median: config.Duration(10 * time.Second)},
	})
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("run took %s; think time did not end with the run", elapsed)
	}
	if len(records) == 0 {
		t.Fatal("no records written")
	}
	for _, rec := range records {
		if rec.Turn != 0 {
			t.Fatalf("record %s: a turn after a 10s think time ran inside a 300ms run", rec.RequestID)
		}
	}
}

func TestRecordsCachedTokensAndNamedHeaders(t *testing.T) {
	url, _ := sessionServer(t)
	records := runGenerator(t, &loadgen.Generator{
		Target:        url + "/v1",
		Model:         "test-model",
		Concurrency:   1,
		Duration:      200 * time.Millisecond,
		Dataset:       dataset.NewSynthetic(16, 4, 1, 4.0),
		RecordHeaders: []string{"x-upstream-host", "x-missing"},
	})
	if len(records) == 0 {
		t.Fatal("no records written")
	}
	for _, rec := range records {
		if rec.CachedTokens == nil || *rec.CachedTokens != 7 {
			t.Fatalf("record %s: cached_tokens %v, want 7", rec.RequestID, rec.CachedTokens)
		}
		if rec.PromptTokens != 20 {
			t.Fatalf("record %s: prompt_tokens %d, want 20", rec.RequestID, rec.PromptTokens)
		}
		want := map[string]string{"x-upstream-host": "10.0.0.7:8000"}
		if len(rec.Headers) != len(want) || rec.Headers["x-upstream-host"] != want["x-upstream-host"] {
			t.Fatalf("record %s: headers %v, want %v", rec.RequestID, rec.Headers, want)
		}
	}
}

func TestRecordOmitsCachedTokensWhenUnreported(t *testing.T) {
	url, _ := captureServer(t)
	outDir := t.TempDir()
	rec, err := recorder.New(outDir, 0)
	if err != nil {
		t.Fatal(err)
	}
	gen := &loadgen.Generator{
		Target:      url + "/v1",
		Model:       "test-model",
		Concurrency: 1,
		Duration:    200 * time.Millisecond,
		Dataset:     dataset.NewSynthetic(16, 4, 1, 4.0),
		Recorder:    rec,
	}
	if _, err := gen.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	rec.Close()

	lines := readRecords(t, filepath.Join(outDir, "requests_0.jsonl"))
	if len(lines) == 0 {
		t.Fatal("no records written")
	}
	for _, r := range lines {
		if r.CachedTokens != nil {
			t.Fatalf("record %s: cached_tokens %d, want absent", r.RequestID, *r.CachedTokens)
		}
		raw, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(raw), "cached_tokens") {
			t.Fatalf("record %s serialized cached_tokens: %s", r.RequestID, raw)
		}
	}
}

func TestRateBasedStagesRecordEveryRequest(t *testing.T) {
	for _, mode := range []loadgen.Mode{loadgen.ModePoisson, loadgen.ModeConstant} {
		t.Run(string(mode), func(t *testing.T) {
			for attempt := 0; attempt < 5; attempt++ {
				url, requests := sessionServer(t)
				outDir := t.TempDir()
				rec, err := recorder.New(outDir, 0)
				if err != nil {
					t.Fatal(err)
				}
				gen := &loadgen.Generator{
					Target:   url + "/v1",
					Model:    "test-model",
					Mode:     mode,
					Rate:     200,
					Dataset:  dataset.NewSynthetic(16, 4, 2, 4.0),
					Recorder: rec,
				}
				gen.RunStages(context.Background(), []loadgen.Stage{{Duration: 200 * time.Millisecond}}, nil, nil)
				rec.Close()

				sent := len(requests())
				written := len(readRecords(t, filepath.Join(outDir, "requests_0.jsonl")))
				if sent == 0 || written != sent {
					t.Fatalf("attempt %d: server saw %d requests, %d records written", attempt, sent, written)
				}
			}
		})
	}
}

// seededWorkload runs one seeded Poisson workload and returns the offsets of
// session starts from the first start and the think gap before each later turn.
func seededWorkload(t *testing.T, seed int64) ([]float64, map[string]float64) {
	t.Helper()
	url, _ := sessionServer(t)
	records := runGenerator(t, &loadgen.Generator{
		Target:    url + "/v1",
		Model:     "test-model",
		Mode:      loadgen.ModePoisson,
		Rate:      15,
		Duration:  time.Second,
		Dataset:   dataset.NewSynthetic(16, 4, 3, 4.0),
		ThinkTime: &config.ThinkTime{Median: config.Duration(80 * time.Millisecond), Sigma: 0.8, Max: config.Duration(400 * time.Millisecond)},
		Seed:      seed,
	})
	byID := map[string]recorder.Record{}
	var starts []float64
	for _, r := range records {
		byID[r.RequestID] = r
		if r.Turn == 0 {
			starts = append(starts, r.StartTime)
		}
	}
	sort.Float64s(starts)
	base := starts[0]
	for i := range starts {
		starts[i] -= base
	}
	gaps := map[string]float64{}
	for _, r := range records {
		if prev, ok := byID[fmt.Sprintf("%s-t%d", r.ConversationID, r.Turn-1)]; ok && r.Turn > 0 {
			gaps[r.RequestID] = r.StartTime - prev.EndTime
		}
	}
	return starts, gaps
}

func TestSeedReplaysArrivalsAndThinkTimes(t *testing.T) {
	const tolerance = 0.03 // seconds of scheduling jitter
	startsA, gapsA := seededWorkload(t, 7)
	startsB, gapsB := seededWorkload(t, 7)
	if len(startsA) < 5 || len(gapsA) < 5 {
		t.Fatalf("too little traffic to compare: %d sessions, %d gaps", len(startsA), len(gapsA))
	}
	n := min(len(startsA), len(startsB))
	if abs := len(startsA) - len(startsB); abs > 1 || abs < -1 {
		t.Fatalf("same seed dispatched %d and %d sessions", len(startsA), len(startsB))
	}
	for i := 0; i < n; i++ {
		if d := startsA[i] - startsB[i]; d > tolerance || d < -tolerance {
			t.Fatalf("session %d started at +%.3fs and +%.3fs with the same seed", i, startsA[i], startsB[i])
		}
	}
	compared := 0
	for id, a := range gapsA {
		b, ok := gapsB[id]
		if !ok {
			continue
		}
		compared++
		if d := a - b; d > tolerance || d < -tolerance {
			t.Fatalf("%s paused %.3fs and %.3fs with the same seed", id, a, b)
		}
	}
	if compared < 5 {
		t.Fatalf("only %d think gaps in common", compared)
	}

	_, gapsC := seededWorkload(t, 8)
	differs := false
	for id, a := range gapsA {
		if c, ok := gapsC[id]; ok && (a-c > tolerance || c-a > tolerance) {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("a different seed replayed the same think gaps")
	}
}

func TestRateBasedStagesRunAtTheirOwnRates(t *testing.T) {
	url, requests := sessionServer(t)
	gen := &loadgen.Generator{
		Target:   url + "/v1",
		Model:    "test-model",
		Mode:     loadgen.ModeConstant,
		Rate:     10,
		Dataset:  dataset.NewSynthetic(16, 4, 1, 4.0),
		Recorder: recorder.NewMemory(),
	}
	var firstStage int
	gen.RunStages(context.Background(), []loadgen.Stage{
		{Duration: time.Second, Rate: 10},
		{Duration: time.Second, Rate: 60},
	}, func(i, _ int) {
		if i == 1 {
			firstStage = len(requests())
		}
	}, nil)

	seen := requests()
	secondStage := len(seen) - firstStage
	if firstStage < 8 || firstStage > 11 {
		t.Errorf("first stage at 10/s sent %d requests in 1s", firstStage)
	}
	if secondStage < 50 || secondStage > 61 {
		t.Errorf("second stage at 60/s sent %d requests in 1s", secondStage)
	}
	ids := map[string]bool{}
	for _, r := range seen {
		if ids[r.requestID] {
			t.Fatalf("request id %q repeats across stages", r.requestID)
		}
		ids[r.requestID] = true
	}
}
