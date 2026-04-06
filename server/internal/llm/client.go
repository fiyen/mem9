package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/qiffang/mnemos/server/internal/metrics"
)

type Client struct {
	apiKey      string
	baseURL     string
	model       string
	temperature float64
	debugLLM    bool
	http        *http.Client
}

type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Temperature float64
	DebugLLM    bool
}

func New(cfg Config) *Client {
	if cfg.APIKey == "" {
		return nil
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0.1
	}
	return &Client{
		apiKey:      cfg.APIKey,
		baseURL:     strings.TrimRight(cfg.BaseURL, "/"),
		model:       cfg.Model,
		temperature: cfg.Temperature,
		debugLLM:    cfg.DebugLLM,
		http: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type responseFormat struct {
	Type string `json:"type"`
}

const defaultMaxOutputTokens = 2048

type chatRequest struct {
	Model               string          `json:"model"`
	Messages            []Message       `json:"messages"`
	Temperature         float64         `json:"temperature"`
	ResponseFormat      *responseFormat `json:"response_format,omitempty"`
	EnableThinking      *bool           `json:"enable_thinking,omitempty"`
	ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitempty"`
	MaxTokens           int             `json:"max_tokens,omitempty"`
	PromptCacheKey      string          `json:"prompt_cache_key,omitempty"`
}

type responsesRequest struct {
	Model           string                  `json:"model"`
	Input           []responsesInputMessage `json:"input"`
	Text            *responsesTextConfig    `json:"text,omitempty"`
	Reasoning       *responsesReasoning     `json:"reasoning,omitempty"`
	MaxOutputTokens int                     `json:"max_output_tokens,omitempty"`
	PromptCacheKey  string                  `json:"prompt_cache_key,omitempty"`
}

type responsesInputMessage struct {
	Role    string                  `json:"role"`
	Content []responsesInputContent `json:"content"`
}

type responsesInputContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responsesTextConfig struct {
	Format *responseFormat `json:"format,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
		// Anthropic-style cache fields (used by some OpenAI-compatible proxies).
		CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	} `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type responsesResponse struct {
	OutputText string `json:"output_text,omitempty"`
	Output     []struct {
		Type    string `json:"type"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output,omitempty"`
	Usage *struct {
		InputTokens        int `json:"input_tokens"`
		OutputTokens       int `json:"output_tokens"`
		TotalTokens        int `json:"total_tokens"`
		InputTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details,omitempty"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	} `json:"usage,omitempty"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type modelOptions struct {
	UseResponsesAPI bool
	EnableThinking  *bool
	ReasoningEffort string
}

// HTTPStatusError is returned when the LLM API responds with an HTTP error status code.
// This enables callers (e.g., CompleteJSON) to detect specific HTTP codes.
type HTTPStatusError struct {
	Code int
	Body string
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("llm http %d: %s", e.Code, e.Body)
}

// Complete sends a chat completion request to the LLM.
func (c *Client) Complete(ctx context.Context, system, user string) (string, error) {
	return c.complete(ctx, system, user, nil)
}

// CompleteJSON sends a chat completion request with response_format: json_object.
// This instructs the model to return valid JSON, improving reliability.
// If the provider returns HTTP 400 (e.g., Ollama, some vLLM builds that don't support
// response_format), it automatically retries without the parameter.
func (c *Client) CompleteJSON(ctx context.Context, system, user string) (string, error) {
	result, err := c.complete(ctx, system, user, &responseFormat{Type: "json_object"})
	if err != nil {
		var httpErr *HTTPStatusError
		if errors.As(err, &httpErr) && httpErr.Code == http.StatusBadRequest {
			slog.Warn("LLM rejected response_format:json_object (HTTP 400), retrying without it")
			return c.complete(ctx, system, user, nil)
		}
	}
	return result, err
}

func (c *Client) complete(ctx context.Context, system, user string, respFmt *responseFormat) (string, error) {
	messages := []Message{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}
	promptCacheKey := buildPromptCacheKey(system, respFmt)

	opts := modelOptionsForModel(c.model)
	if opts.UseResponsesAPI {
		result, err := c.doResponsesRequest(ctx, messages, respFmt, promptCacheKey, opts)
		if err == nil {
			return result, nil
		}

		var httpErr *HTTPStatusError
		if errors.As(err, &httpErr) && (httpErr.Code == http.StatusBadRequest || httpErr.Code == http.StatusNotFound || httpErr.Code == http.StatusMethodNotAllowed) {
			slog.Warn("LLM responses API unavailable, retrying with chat completions", "model", c.model, "status", httpErr.Code)
		} else {
			return "", err
		}
	}

	return c.doChatRequest(ctx, messages, respFmt, promptCacheKey, opts)
}

func (c *Client) doChatRequest(ctx context.Context, messages []Message, respFmt *responseFormat, promptCacheKey string, opts modelOptions) (string, error) {
	cr := chatRequest{
		Model:               c.model,
		Messages:            messages,
		Temperature:         c.temperature,
		ResponseFormat:      respFmt,
		EnableThinking:      opts.EnableThinking,
		ReasoningEffort:     opts.ReasoningEffort,
		MaxCompletionTokens: defaultMaxOutputTokens,
		MaxTokens:           defaultMaxOutputTokens,
		PromptCacheKey:      promptCacheKey,
	}

	result, err := c.doChatCompletionRequest(ctx, cr)
	if err != nil {
		var httpErr *HTTPStatusError
		if errors.As(err, &httpErr) && httpErr.Code == http.StatusBadRequest && cr.MaxCompletionTokens != 0 {
			slog.Warn("LLM rejected max_completion_tokens (HTTP 400), retrying with max_tokens only", "model", c.model)
			cr.MaxCompletionTokens = 0
			result, err = c.doChatCompletionRequest(ctx, cr)
			if err == nil {
				return result, nil
			}
		}

		// If 400 and reasoning/thinking parameters were sent, retry without them (provider may not support them).
		if errors.As(err, &httpErr) && httpErr.Code == http.StatusBadRequest && (opts.EnableThinking != nil || opts.ReasoningEffort != "") {
			slog.Warn("LLM rejected thinking parameters (HTTP 400), retrying without them", "model", c.model)
			cr.EnableThinking = nil
			cr.ReasoningEffort = ""
			return c.doChatCompletionRequest(ctx, cr)
		}
	}
	return result, err
}

// doChatCompletionRequest sends a single chat completion request and handles metrics/response parsing.
func (c *Client) doChatCompletionRequest(ctx context.Context, cr chatRequest) (string, error) {
	start := time.Now()

	body, err := json.Marshal(cr)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(time.Since(start).Seconds())
		return "", fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	duration := time.Since(start).Seconds()

	// Surface HTTP errors as typed errors so callers can detect specific status codes.
	if resp.StatusCode >= 400 {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", &HTTPStatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("decode response: %w", err)
	}

	if chatResp.Error != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("llm error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("llm returned no choices")
	}

	content := chatResp.Choices[0].Message.Content
	if c.debugLLM {
		slog.Debug("llm raw response", "model", c.model, "len", len(content), "raw", content)
	}

	metrics.LLMRequestDuration.WithLabelValues(c.model, "success").Observe(duration)
	if chatResp.Usage != nil {
		u := chatResp.Usage
		metrics.LLMTokensTotal.WithLabelValues(c.model, "input").Add(float64(u.PromptTokens))
		metrics.LLMTokensTotal.WithLabelValues(c.model, "output").Add(float64(u.CompletionTokens))
		metrics.LLMTokensTotal.WithLabelValues(c.model, "total").Add(float64(u.TotalTokens))

		// Cache tokens: try OpenAI-style (prompt_tokens_details.cached_tokens), then Anthropic-style.
		cacheRead := u.CacheReadInputTokens
		if cacheRead == 0 && u.PromptTokensDetails != nil {
			cacheRead = u.PromptTokensDetails.CachedTokens
		}
		if cacheRead > 0 {
			metrics.LLMTokensTotal.WithLabelValues(c.model, "cache_read").Add(float64(cacheRead))
		}
		if u.CacheCreationInputTokens > 0 {
			metrics.LLMTokensTotal.WithLabelValues(c.model, "cache_creation").Add(float64(u.CacheCreationInputTokens))
		}
	}
	return content, nil
}

func (c *Client) doResponsesRequest(ctx context.Context, messages []Message, respFmt *responseFormat, promptCacheKey string, opts modelOptions) (string, error) {
	start := time.Now()

	reqBody := responsesRequest{
		Model:           c.model,
		Input:           make([]responsesInputMessage, 0, len(messages)),
		MaxOutputTokens: defaultMaxOutputTokens,
		PromptCacheKey:  promptCacheKey,
	}
	if respFmt != nil {
		reqBody.Text = &responsesTextConfig{Format: respFmt}
	}
	if opts.ReasoningEffort != "" {
		reqBody.Reasoning = &responsesReasoning{Effort: opts.ReasoningEffort}
	}
	for _, msg := range messages {
		reqBody.Input = append(reqBody.Input, responsesInputMessage{
			Role: msg.Role,
			Content: []responsesInputContent{
				{Type: "input_text", Text: msg.Content},
			},
		})
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(time.Since(start).Seconds())
		return "", fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	duration := time.Since(start).Seconds()
	if resp.StatusCode >= 400 {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", &HTTPStatusError{Code: resp.StatusCode, Body: string(respBody)}
	}

	var parsed responsesResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("decode response: %w", err)
	}
	if parsed.Error != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("llm error: %s", parsed.Error.Message)
	}

	content := strings.TrimSpace(parsed.OutputText)
	if content == "" {
		var sb strings.Builder
		for _, item := range parsed.Output {
			for _, part := range item.Content {
				if part.Type != "output_text" || part.Text == "" {
					continue
				}
				if sb.Len() > 0 {
					sb.WriteByte('\n')
				}
				sb.WriteString(part.Text)
			}
		}
		content = strings.TrimSpace(sb.String())
	}
	if content == "" {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(duration)
		return "", fmt.Errorf("llm returned no output text")
	}

	if c.debugLLM {
		slog.Debug("llm raw response", "model", c.model, "len", len(content), "raw", content)
	}

	metrics.LLMRequestDuration.WithLabelValues(c.model, "success").Observe(duration)
	if parsed.Usage != nil {
		u := parsed.Usage
		metrics.LLMTokensTotal.WithLabelValues(c.model, "input").Add(float64(u.InputTokens))
		metrics.LLMTokensTotal.WithLabelValues(c.model, "output").Add(float64(u.OutputTokens))
		metrics.LLMTokensTotal.WithLabelValues(c.model, "total").Add(float64(u.TotalTokens))

		cacheRead := u.CacheReadInputTokens
		if cacheRead == 0 && u.InputTokensDetails != nil {
			cacheRead = u.InputTokensDetails.CachedTokens
		}
		if cacheRead > 0 {
			metrics.LLMTokensTotal.WithLabelValues(c.model, "cache_read").Add(float64(cacheRead))
		}
		if u.CacheCreationInputTokens > 0 {
			metrics.LLMTokensTotal.WithLabelValues(c.model, "cache_creation").Add(float64(u.CacheCreationInputTokens))
		}
	}

	return content, nil
}

func (c *Client) DebugLLM() bool {
	return c.debugLLM
}

func modelOptionsForModel(model string) modelOptions {
	normalized := strings.ToLower(strings.TrimSpace(model))

	switch {
	case strings.Contains(normalized, "qwen"):
		enableThinking := false
		return modelOptions{EnableThinking: &enableThinking}
	case strings.Contains(normalized, "codex"):
		return modelOptions{UseResponsesAPI: true, ReasoningEffort: "none"}
	case strings.HasPrefix(normalized, "gpt-5"):
		return modelOptions{UseResponsesAPI: true, ReasoningEffort: "none"}
	}
	return modelOptions{}
}

func buildPromptCacheKey(system string, respFmt *responseFormat) string {
	task := inferPromptTask(system)
	formatType := "text"
	if respFmt != nil && respFmt.Type != "" {
		formatType = respFmt.Type
	}
	hash := sha256.Sum256([]byte(normalizePromptCacheSource(system)))
	return fmt.Sprintf("mnemos:v1:%s:%s:%s", task, formatType, hex.EncodeToString(hash[:6]))
}

func inferPromptTask(system string) string {
	normalized := strings.ToLower(system)
	switch {
	case strings.Contains(normalized, "memory management engine"):
		return "reconcile"
	case strings.Contains(normalized, "message tags"):
		return "extract_tags"
	case strings.Contains(normalized, "information extraction engine"):
		return "extract"
	default:
		return "generic"
	}
}

func normalizePromptCacheSource(s string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
}

func StripMarkdownFences(s string) string {
	re := regexp.MustCompile("(?s)^\\s*```(?:json)?\\s*\n?(.*?)\\s*```\\s*$")
	if match := re.FindStringSubmatch(s); len(match) > 1 {
		return strings.TrimSpace(match[1])
	}
	return strings.TrimSpace(s)
}

func ParseJSON[T any](raw string) (T, error) {
	var result T
	cleaned := StripMarkdownFences(raw)
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		return result, fmt.Errorf("invalid JSON: %w", err)
	}
	return result, nil
}
