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

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	EnableThinking *bool           `json:"enable_thinking,omitempty"`
	PromptCacheKey string          `json:"prompt_cache_key,omitempty"`
}

type responsesRequest struct {
	Model          string          `json:"model"`
	Input          []Message       `json:"input"`
	Temperature    float64         `json:"temperature,omitempty"`
	Format         *responseFormat `json:"format,omitempty"`
	PromptCacheKey string          `json:"prompt_cache_key,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
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
	Usage llmUsage `json:"usage,omitempty"`
}

type responsesResponse struct {
	Output []struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type    string `json:"type"`
			Text    string `json:"text,omitempty"`
			Refusal string `json:"refusal,omitempty"`
		} `json:"content"`
	} `json:"output"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
	Usage      llmUsage `json:"usage,omitempty"`
	OutputText string   `json:"output_text,omitempty"`
}

type llmUsage struct {
	InputTokens        int `json:"input_tokens,omitempty"`
	OutputTokens       int `json:"output_tokens,omitempty"`
	PromptTokens       int `json:"prompt_tokens,omitempty"`
	CompletionTokens   int `json:"completion_tokens,omitempty"`
	TotalTokens        int `json:"total_tokens,omitempty"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"completion_tokens_details,omitempty"`
}

type llmUsageSnapshot struct {
	InputTokens     int
	OutputTokens    int
	TotalTokens     int
	CachedTokens    int
	ReasoningTokens int
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

	enableThinking := disableThinkingOptions(c.model)

	result, err := c.doRequest(ctx, chatRequest{
		Model:          c.model,
		Messages:       messages,
		Temperature:    c.temperature,
		ResponseFormat: respFmt,
		EnableThinking: enableThinking,
		PromptCacheKey: promptCacheKey,
	})
	if err != nil {
		// If 400 and thinking parameters were sent, retry without them (provider may not support them).
		var httpErr *HTTPStatusError
		if errors.As(err, &httpErr) && httpErr.Code == http.StatusBadRequest && enableThinking != nil {
			slog.Warn("LLM rejected thinking parameters (HTTP 400), retrying without them", "model", c.model)
			return c.doRequest(ctx, chatRequest{
				Model:          c.model,
				Messages:       messages,
				Temperature:    c.temperature,
				ResponseFormat: respFmt,
				PromptCacheKey: promptCacheKey,
			})
		}
	}
	return result, err
}

// doRequest sends a single chat completion request and handles metrics/response parsing.
func (c *Client) doRequest(ctx context.Context, cr chatRequest) (string, error) {
	start := time.Now()
	if useResponsesAPI(c.model) {
		content, usage, endpoint, err := c.doResponsesRequest(ctx, cr)
		if err == nil {
			c.observeUsage(endpoint, cr.PromptCacheKey, usage, start)
			return content, nil
		}
		if shouldFallbackToChat(err) {
			slog.Warn("responses api request failed, falling back to chat completions", "model", c.model, "err", err)
		} else {
			metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(time.Since(start).Seconds())
			return "", err
		}
	}

	content, usage, endpoint, err := c.doChatRequest(ctx, cr)
	if err != nil {
		metrics.LLMRequestDuration.WithLabelValues(c.model, "error").Observe(time.Since(start).Seconds())
		return "", err
	}
	c.observeUsage(endpoint, cr.PromptCacheKey, usage, start)
	return content, nil
}

func (c *Client) DebugLLM() bool {
	return c.debugLLM
}

func disableThinkingOptions(model string) *bool {
	if strings.Contains(strings.ToLower(model), "qwen") {
		enableThinking := false
		return &enableThinking
	}
	return nil
}

func useResponsesAPI(model string) bool {
	return !strings.Contains(strings.ToLower(model), "qwen")
}

func shouldFallbackToChat(err error) bool {
	var httpErr *HTTPStatusError
	if !errors.As(err, &httpErr) {
		return false
	}
	switch httpErr.Code {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusUnsupportedMediaType, http.StatusUnprocessableEntity, http.StatusNotImplemented:
		return true
	default:
		return false
	}
}

func (c *Client) doResponsesRequest(ctx context.Context, cr chatRequest) (string, llmUsageSnapshot, string, error) {
	reqBody := responsesRequest{
		Model:          cr.Model,
		Input:          cr.Messages,
		Temperature:    cr.Temperature,
		Format:         cr.ResponseFormat,
		PromptCacheKey: cr.PromptCacheKey,
	}
	respBody, err := c.sendRequest(ctx, "/responses", reqBody)
	if err != nil {
		return "", llmUsageSnapshot{}, "responses", err
	}

	var response responsesResponse
	if err := json.Unmarshal(respBody, &response); err != nil {
		return "", llmUsageSnapshot{}, "responses", fmt.Errorf("decode responses response: %w", err)
	}
	if response.Error != nil {
		return "", llmUsageSnapshot{}, "responses", fmt.Errorf("llm error: %s", response.Error.Message)
	}

	content := extractResponsesText(response)
	if content == "" {
		var legacy chatResponse
		if err := json.Unmarshal(respBody, &legacy); err == nil && len(legacy.Choices) > 0 {
			content = legacy.Choices[0].Message.Content
			if c.debugLLM {
				slog.Debug("llm raw response", "model", c.model, "endpoint", "responses_legacy_shape", "len", len(content), "raw", content)
			}
			return content, snapshotUsage(legacy.Usage), "responses", nil
		}
		return "", llmUsageSnapshot{}, "responses", fmt.Errorf("responses api returned no text output")
	}
	if c.debugLLM {
		slog.Debug("llm raw response", "model", c.model, "endpoint", "responses", "len", len(content), "raw", content)
	}
	return content, snapshotUsage(response.Usage), "responses", nil
}

func (c *Client) doChatRequest(ctx context.Context, cr chatRequest) (string, llmUsageSnapshot, string, error) {
	respBody, err := c.sendRequest(ctx, "/chat/completions", cr)
	if err != nil {
		return "", llmUsageSnapshot{}, "chat_completions", err
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", llmUsageSnapshot{}, "chat_completions", fmt.Errorf("decode response: %w", err)
	}

	if chatResp.Error != nil {
		return "", llmUsageSnapshot{}, "chat_completions", fmt.Errorf("llm error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", llmUsageSnapshot{}, "chat_completions", fmt.Errorf("llm returned no choices")
	}

	content := chatResp.Choices[0].Message.Content
	if c.debugLLM {
		slog.Debug("llm raw response", "model", c.model, "endpoint", "chat_completions", "len", len(content), "raw", content)
	}
	return content, snapshotUsage(chatResp.Usage), "chat_completions", nil
}

func (c *Client) sendRequest(ctx context.Context, path string, payload interface{}) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, &HTTPStatusError{Code: resp.StatusCode, Body: string(respBody)}
	}
	return respBody, nil
}

func (c *Client) observeUsage(endpoint, promptCacheKey string, usage llmUsageSnapshot, start time.Time) {
	duration := time.Since(start).Seconds()
	metrics.LLMRequestDuration.WithLabelValues(c.model, "success").Observe(duration)
	metrics.LLMInputTokens.WithLabelValues(c.model, endpoint).Add(float64(usage.InputTokens))
	metrics.LLMCachedInputTokens.WithLabelValues(c.model, endpoint).Add(float64(usage.CachedTokens))
	metrics.LLMOutputTokens.WithLabelValues(c.model, endpoint).Add(float64(usage.OutputTokens))
	metrics.LLMReasoningTokens.WithLabelValues(c.model, endpoint).Add(float64(usage.ReasoningTokens))

	if usage.CachedTokens > 0 || c.debugLLM {
		slog.Info("llm request completed",
			"model", c.model,
			"endpoint", endpoint,
			"prompt_cache_key", promptCacheKey,
			"input_tokens", usage.InputTokens,
			"cached_tokens", usage.CachedTokens,
			"output_tokens", usage.OutputTokens,
			"reasoning_tokens", usage.ReasoningTokens,
			"total_tokens", usage.TotalTokens,
			"cache_hit", usage.CachedTokens > 0,
			"duration_seconds", duration,
		)
	}
}

func snapshotUsage(usage llmUsage) llmUsageSnapshot {
	inputTokens := usage.InputTokens
	if inputTokens == 0 {
		inputTokens = usage.PromptTokens
	}
	outputTokens := usage.OutputTokens
	if outputTokens == 0 {
		outputTokens = usage.CompletionTokens
	}
	cachedTokens := usage.InputTokensDetails.CachedTokens
	if cachedTokens == 0 {
		cachedTokens = usage.PromptTokensDetails.CachedTokens
	}
	reasoningTokens := usage.OutputTokensDetails.ReasoningTokens
	if reasoningTokens == 0 {
		reasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
	}
	return llmUsageSnapshot{
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		TotalTokens:     usage.TotalTokens,
		CachedTokens:    cachedTokens,
		ReasoningTokens: reasoningTokens,
	}
}

func extractResponsesText(resp responsesResponse) string {
	if strings.TrimSpace(resp.OutputText) != "" {
		return resp.OutputText
	}

	var parts []string
	for _, output := range resp.Output {
		if output.Type != "message" {
			continue
		}
		for _, content := range output.Content {
			switch content.Type {
			case "output_text", "text":
				if text := strings.TrimSpace(content.Text); text != "" {
					parts = append(parts, text)
				}
			case "refusal":
				if text := strings.TrimSpace(content.Refusal); text != "" {
					parts = append(parts, text)
				}
			}
		}
	}
	return strings.Join(parts, "\n")
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
