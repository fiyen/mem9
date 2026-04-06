package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	t.Run("empty api key returns nil", func(t *testing.T) {
		if got := New(Config{}); got != nil {
			t.Fatalf("expected nil client, got %#v", got)
		}
	})

	t.Run("defaults and trims base url", func(t *testing.T) {
		client := New(Config{APIKey: "key", BaseURL: "https://example.com/v1////"})
		if client == nil {
			t.Fatal("expected client, got nil")
		}
		if client.baseURL != "https://example.com/v1" {
			t.Fatalf("baseURL = %q, want %q", client.baseURL, "https://example.com/v1")
		}
		if client.model != "gpt-4o-mini" {
			t.Fatalf("model = %q, want %q", client.model, "gpt-4o-mini")
		}
	})

	t.Run("defaults applied when fields empty", func(t *testing.T) {
		client := New(Config{APIKey: "key"})
		if client == nil {
			t.Fatal("expected client, got nil")
		}
		if client.baseURL != "https://api.openai.com/v1" {
			t.Fatalf("baseURL = %q, want %q", client.baseURL, "https://api.openai.com/v1")
		}
		if client.model != "gpt-4o-mini" {
			t.Fatalf("model = %q, want %q", client.model, "gpt-4o-mini")
		}
	})
}

func TestStripMarkdownFences(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "raw json", in: `{"a":1}`, want: `{"a":1}`},
		{name: "json fence", in: "```json\n{\"a\":1}\n```", want: `{"a":1}`},
		{name: "plain fence", in: "```\n{\"a\":1}\n```", want: `{"a":1}`},
		{name: "nested content", in: "```json\n{\"a\":\"```not fence```\"}\n```", want: "{\"a\":\"```not fence```\"}"},
		{name: "whitespace", in: " \n```json\n  {\"a\":1}\n``` \n", want: `{"a":1}`},
		{name: "empty", in: "", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := StripMarkdownFences(tt.in); got != tt.want {
				t.Fatalf("StripMarkdownFences(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestParseJSON(t *testing.T) {
	type payload struct {
		Name string `json:"name"`
	}

	tests := []struct {
		name    string
		input   string
		want    payload
		wantErr bool
	}{
		{name: "raw json", input: `{"name":"alpha"}`, want: payload{Name: "alpha"}},
		{name: "fenced json", input: "```json\n{\"name\":\"beta\"}\n```", want: payload{Name: "beta"}},
		{name: "invalid json", input: "{", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseJSON[payload](tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("ParseJSON result = %#v, want %#v", got, tt.want)
			}
		})
	}

	t.Run("different type", func(t *testing.T) {
		got, err := ParseJSON[map[string]int]("```\n{\"a\":2}\n```")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got["a"] != 2 {
			t.Fatalf("ParseJSON map value = %d, want %d", got["a"], 2)
		}
	})
}

func TestComplete(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				t.Fatalf("method = %s, want POST", r.Method)
			}
			if r.URL.Path != "/chat/completions" {
				t.Fatalf("path = %s, want /chat/completions", r.URL.Path)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer key" {
				t.Fatalf("Authorization header = %q, want %q", got, "Bearer key")
			}
			if got := r.Header.Get("Content-Type"); got != "application/json" {
				t.Fatalf("Content-Type header = %q, want %q", got, "application/json")
			}

			var req chatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.Model != "test-model" {
				t.Fatalf("model = %q, want %q", req.Model, "test-model")
			}
			if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[1].Role != "user" {
				t.Fatalf("unexpected messages: %#v", req.Messages)
			}
			if req.Temperature != 0.1 {
				t.Fatalf("temperature = %v, want %v", req.Temperature, 0.1)
			}
			if req.MaxTokens != defaultMaxOutputTokens {
				t.Fatalf("max_tokens = %d, want %d", req.MaxTokens, defaultMaxOutputTokens)
			}
			if req.MaxCompletionTokens != defaultMaxOutputTokens {
				t.Fatalf("max_completion_tokens = %d, want %d", req.MaxCompletionTokens, defaultMaxOutputTokens)
			}
			if req.EnableThinking != nil {
				t.Fatalf("enable_thinking = %v, want nil", *req.EnableThinking)
			}
			if req.ReasoningEffort != "" {
				t.Fatalf("reasoning_effort = %q, want empty", req.ReasoningEffort)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "test-model"})
		if client == nil {
			t.Fatal("expected client, got nil")
		}

		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "hello" {
			t.Fatalf("content = %q, want %q", got, "hello")
		}
	})

	t.Run("api error response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"error":{"message":"bad request"}}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "test-model"})
		_, err := client.Complete(context.Background(), "sys", "user")
		if err == nil || !strings.Contains(err.Error(), "llm error: bad request") {
			t.Fatalf("expected llm error, got %v", err)
		}
	})

	t.Run("empty choices", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[]}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "test-model"})
		_, err := client.Complete(context.Background(), "sys", "user")
		if err == nil || !strings.Contains(err.Error(), "llm returned no choices") {
			t.Fatalf("expected empty choices error, got %v", err)
		}
	})

	t.Run("http error", func(t *testing.T) {
		client := New(Config{APIKey: "key", BaseURL: "http://example.com", Model: "test-model"})
		if client == nil {
			t.Fatal("expected client, got nil")
		}
		client.http = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("boom")
		})}

		_, err := client.Complete(context.Background(), "sys", "user")
		if err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("expected request error, got %v", err)
		}
	})

	t.Run("qwen model disables thinking with enable_thinking", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req chatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.EnableThinking == nil || *req.EnableThinking {
				t.Fatalf("enable_thinking = %v, want %v", req.EnableThinking, false)
			}
			if req.MaxTokens != defaultMaxOutputTokens {
				t.Fatalf("max_tokens = %d, want %d", req.MaxTokens, defaultMaxOutputTokens)
			}
			if req.MaxCompletionTokens != defaultMaxOutputTokens {
				t.Fatalf("max_completion_tokens = %d, want %d", req.MaxCompletionTokens, defaultMaxOutputTokens)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "qwen-plus"})
		if client == nil {
			t.Fatal("expected client, got nil")
		}

		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "hello" {
			t.Fatalf("content = %q, want %q", got, "hello")
		}
	})

	t.Run("chat completions retries without max_completion_tokens when provider rejects it", func(t *testing.T) {
		callCount := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			callCount++

			var req chatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}

			if callCount == 1 {
				if req.MaxCompletionTokens != defaultMaxOutputTokens {
					t.Fatalf("max_completion_tokens = %d, want %d", req.MaxCompletionTokens, defaultMaxOutputTokens)
				}
				http.Error(w, `{"error":{"message":"unsupported field max_completion_tokens"}}`, http.StatusBadRequest)
				return
			}

			if req.MaxCompletionTokens != 0 {
				t.Fatalf("max_completion_tokens = %d, want 0 on retry", req.MaxCompletionTokens)
			}
			if req.MaxTokens != defaultMaxOutputTokens {
				t.Fatalf("max_tokens = %d, want %d", req.MaxTokens, defaultMaxOutputTokens)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "test-model"})
		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "hello" {
			t.Fatalf("content = %q, want %q", got, "hello")
		}
		if callCount != 2 {
			t.Fatalf("callCount = %d, want 2", callCount)
		}
	})

	t.Run("gpt-5 uses responses api with hard cap and thinking disabled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/responses" {
				t.Fatalf("path = %s, want /responses", r.URL.Path)
			}

			var req responsesRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.MaxOutputTokens != defaultMaxOutputTokens {
				t.Fatalf("max_output_tokens = %d, want %d", req.MaxOutputTokens, defaultMaxOutputTokens)
			}
			if req.Reasoning == nil || req.Reasoning.Effort != "none" {
				t.Fatalf("reasoning = %#v, want none", req.Reasoning)
			}
			if len(req.Input) != 2 || req.Input[0].Role != "system" || req.Input[1].Role != "user" {
				t.Fatalf("unexpected input messages: %#v", req.Input)
			}
			if got := req.Input[0].Content[0].Type; got != "input_text" {
				t.Fatalf("input content type = %q, want input_text", got)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":[{"type":"message","content":[{"type":"output_text","text":"hello from responses"}]}]}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "gpt-5"})
		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "hello from responses" {
			t.Fatalf("content = %q, want %q", got, "hello from responses")
		}
	})

	t.Run("responses api falls back to chat completions when unavailable", func(t *testing.T) {
		var paths []string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			paths = append(paths, r.URL.Path)
			switch r.URL.Path {
			case "/responses":
				http.Error(w, `{"error":{"message":"not found"}}`, http.StatusNotFound)
			case "/chat/completions":
				var req chatRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatalf("decode chat request: %v", err)
				}
				if req.MaxTokens != defaultMaxOutputTokens {
					t.Fatalf("max_tokens = %d, want %d", req.MaxTokens, defaultMaxOutputTokens)
				}
				if req.MaxCompletionTokens != defaultMaxOutputTokens {
					t.Fatalf("max_completion_tokens = %d, want %d", req.MaxCompletionTokens, defaultMaxOutputTokens)
				}
				if req.ReasoningEffort != "none" {
					t.Fatalf("reasoning_effort = %q, want none", req.ReasoningEffort)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"chat fallback"}}]}`))
			default:
				t.Fatalf("unexpected path %s", r.URL.Path)
			}
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "gpt-5"})
		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "chat fallback" {
			t.Fatalf("content = %q, want %q", got, "chat fallback")
		}
		if len(paths) != 2 || paths[0] != "/responses" || paths[1] != "/chat/completions" {
			t.Fatalf("paths = %v, want [/responses /chat/completions]", paths)
		}
	})

	t.Run("gpt-5.1 uses none reasoning effort", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req responsesRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.Reasoning == nil || req.Reasoning.Effort != "none" {
				t.Fatalf("reasoning = %#v, want none", req.Reasoning)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output_text":"ok"}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "gpt-5.1-mini"})
		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "ok" {
			t.Fatalf("content = %q, want %q", got, "ok")
		}
	})

	t.Run("codex uses none reasoning effort", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req responsesRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if req.Reasoning == nil || req.Reasoning.Effort != "none" {
				t.Fatalf("reasoning = %#v, want none", req.Reasoning)
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output_text":"ok"}`))
		}))
		defer server.Close()

		client := New(Config{APIKey: "key", BaseURL: server.URL, Model: "gpt-5-codex"})
		got, err := client.Complete(context.Background(), "sys", "user")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "ok" {
			t.Fatalf("content = %q, want %q", got, "ok")
		}
	})
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
