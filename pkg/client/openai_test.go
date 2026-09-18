package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientDoesNotFollowRedirects(t *testing.T) {
	reached := false
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"should-not-be-read"}]}`))
	}))
	defer destination.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()

	if _, err := New(redirect.URL).DetectModel(context.Background()); err == nil {
		t.Fatal("redirect response unexpectedly produced a model")
	}
	if reached {
		t.Fatal("client followed a redirect outside the selected target")
	}
}

func TestChatStreamSeparatesReasoningFromGeneratedText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think \"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"more \"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	c := New(server.URL + "/v1")
	result := c.ChatStream(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "question"}},
	})

	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.Content != "answer" {
		t.Fatalf("Content = %q, want %q", result.Content, "answer")
	}
	if result.Reasoning != "think more " {
		t.Fatalf("Reasoning = %q, want %q", result.Reasoning, "think more ")
	}
	if result.GeneratedText != "think more answer" {
		t.Fatalf("GeneratedText = %q, want %q", result.GeneratedText, "think more answer")
	}
	if len(result.TokenTimes) != 3 {
		t.Fatalf("TokenTimes has %d entries, want 3", len(result.TokenTimes))
	}
}

// usageServer asserts the caller asked for a usage chunk, then streams one
// token followed by a usage chunk carrying cached prompt tokens.
func usageServer(t *testing.T, usage string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			StreamOptions *struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if body.StreamOptions == nil || !body.StreamOptions.IncludeUsage {
			http.Error(w, "stream_options.include_usage not set", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Upstream-Host", "10.0.0.7:8000")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"text\":\"hi\",\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":%s}\n\n", usage)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server
}

func TestChatStreamRequestsUsageAndParsesCachedTokens(t *testing.T) {
	server := usageServer(t, `{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101,"prompt_tokens_details":{"cached_tokens":96}}`)

	result := New(server.URL+"/v1").ChatStream(context.Background(), &Request{
		Model:         "test-model",
		Messages:      []Message{{Role: "user", Content: "question"}},
		StreamOptions: map[string]any{"include_usage": true},
	})
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.Usage == nil || result.Usage.PromptTokens != 100 {
		t.Fatalf("Usage = %+v, want prompt_tokens 100", result.Usage)
	}
	if result.Usage.PromptTokensDetails == nil || result.Usage.PromptTokensDetails.CachedTokens != 96 {
		t.Fatalf("PromptTokensDetails = %+v, want cached_tokens 96", result.Usage.PromptTokensDetails)
	}
	if got := result.Header.Get("X-Upstream-Host"); got != "10.0.0.7:8000" {
		t.Fatalf("X-Upstream-Host = %q, want %q", got, "10.0.0.7:8000")
	}
}

func TestChatStreamWithoutPromptTokensDetails(t *testing.T) {
	server := usageServer(t, `{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101}`)

	result := New(server.URL+"/v1").ChatStream(context.Background(), &Request{
		Model:         "test-model",
		Messages:      []Message{{Role: "user", Content: "question"}},
		StreamOptions: map[string]any{"include_usage": true},
	})
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.Usage == nil || result.Usage.PromptTokensDetails != nil {
		t.Fatalf("Usage = %+v, want usage without prompt_tokens_details", result.Usage)
	}
}

func TestCompletionStreamRequestsUsage(t *testing.T) {
	server := usageServer(t, `{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}`)

	result := New(server.URL+"/v1").CompletionStream(context.Background(), &CompletionRequest{
		Model:         "test-model",
		Prompt:        "question",
		StreamOptions: map[string]any{"include_usage": true},
	})
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.Usage == nil || result.Usage.PromptTokens != 10 {
		t.Fatalf("Usage = %+v, want prompt_tokens 10", result.Usage)
	}
}

func TestChatStreamKeepsHeadersOnErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Upstream-Host", "10.0.0.9:8000")
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	result := New(server.URL+"/v1").ChatStream(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "question"}},
	})
	if result.Err == nil {
		t.Fatal("expected an error for a 503 response")
	}
	if got := result.Header.Get("X-Upstream-Host"); got != "10.0.0.9:8000" {
		t.Fatalf("X-Upstream-Host = %q, want %q", got, "10.0.0.9:8000")
	}
}
