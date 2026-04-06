package service

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/qiffang/mnemos/server/internal/domain"
)

const (
	DefaultTraceLimit           = 5
	maxTraceLimit               = 20
	traceSemanticMinScore       = 0.45
	traceMaxCandidateNodes      = 64
	traceSkipContentLength      = 12000
	traceSkipStructuredLength   = 1200
	traceSkipMultilineThreshold = 120
)

type TraceService struct {
	memories *MemoryService
	sessions *SessionService
}

type MemoryTrace struct {
	Memory    *domain.Memory  `json:"memory"`
	SessionID string          `json:"session_id"`
	Query     string          `json:"query"`
	Evidence  []TraceEvidence `json:"evidence"`
}

type TraceEvidence struct {
	ID      string   `json:"id"`
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Score   *float64 `json:"score,omitempty"`
	Seq     int      `json:"seq"`
}

type traceMetadata struct {
	Role        string `json:"role"`
	Seq         int    `json:"seq"`
	ContentType string `json:"content_type"`
}

func NewTraceService(memories *MemoryService, sessions *SessionService) *TraceService {
	return &TraceService{
		memories: memories,
		sessions: sessions,
	}
}

func (s *TraceService) Trace(ctx context.Context, id, q string, limit int) (*MemoryTrace, error) {
	if s.memories == nil || s.sessions == nil {
		return nil, fmt.Errorf("trace service unavailable: %w", domain.ErrNotSupported)
	}

	mem, err := s.memories.Get(ctx, id)
	if err != nil {
		return nil, err
	}

	sessionID := strings.TrimSpace(mem.SessionID)
	if sessionID == "" {
		return nil, fmt.Errorf("memory %s has no session provenance: %w", id, domain.ErrNotSupported)
	}

	requestedQuery := strings.TrimSpace(q)
	memoryQuery := strings.TrimSpace(mem.Content)
	query := requestedQuery
	if query == "" {
		query = memoryQuery
	}
	if query == "" {
		return nil, &domain.ValidationError{Field: "q", Message: "query is empty and memory has no content"}
	}

	traceLimit := normalizeTraceLimit(limit)

	results, err := s.semanticEvidence(ctx, sessionID, query, traceLimit)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 && requestedQuery != "" && memoryQuery != "" && memoryQuery != requestedQuery {
		results, err = s.semanticEvidence(ctx, sessionID, memoryQuery, traceLimit)
		if err != nil {
			return nil, err
		}
	}
	if len(results) == 0 {
		results, err = s.recentEvidence(ctx, sessionID, traceLimit)
		if err != nil {
			return nil, err
		}
	}

	evidence := make([]TraceEvidence, 0, len(results))
	for _, result := range results {
		evidence = append(evidence, newTraceEvidence(result))
	}

	return &MemoryTrace{
		Memory:    mem,
		SessionID: sessionID,
		Query:     query,
		Evidence:  evidence,
	}, nil
}

func (s *TraceService) semanticEvidence(ctx context.Context, sessionID, query string, limit int) ([]domain.Memory, error) {
	if s.sessions.embedder == nil {
		return nil, nil
	}

	candidates, err := s.traceCandidates(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	model := traceEmbeddingModel(s.sessions)
	if err := s.ensureTraceEmbeddings(ctx, model, candidates); err != nil {
		return nil, err
	}

	queryVec, err := s.sessions.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("trace embed query: %w", err)
	}

	results, err := s.sessions.TraceVectorSearch(ctx, sessionID, model, queryVec, limit)
	if err != nil {
		return nil, fmt.Errorf("trace vector search: %w", err)
	}
	results = applyMinScore(results, traceSemanticMinScore)
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func (s *TraceService) traceCandidates(ctx context.Context, sessionID string) ([]*domain.Session, error) {
	rows, err := s.sessions.ListBySessionID(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("trace list session nodes: %w", err)
	}
	return selectTraceCandidates(rows), nil
}

func (s *TraceService) ensureTraceEmbeddings(ctx context.Context, model string, candidates []*domain.Session) error {
	existing, err := s.sessions.ListTraceEmbeddingsBySessionID(ctx, candidates[0].SessionID)
	if err != nil {
		return fmt.Errorf("trace list cached embeddings: %w", err)
	}

	existingByNodeID := make(map[string]*domain.SessionTraceEmbedding, len(existing))
	for _, entry := range existing {
		existingByNodeID[entry.NodeID] = entry
	}

	pending := make([]*domain.SessionTraceEmbedding, 0)
	for _, candidate := range candidates {
		entry := existingByNodeID[candidate.ID]
		if entry != nil &&
			entry.ContentHash == candidate.ContentHash &&
			entry.Role == candidate.Role &&
			entry.EmbeddingModel == model {
			continue
		}

		embedding, err := s.sessions.embedder.Embed(ctx, candidate.Content)
		if err != nil {
			return fmt.Errorf("trace embed node %s: %w", candidate.ID, err)
		}
		pending = append(pending, &domain.SessionTraceEmbedding{
			NodeID:         candidate.ID,
			SessionID:      candidate.SessionID,
			ContentHash:    candidate.ContentHash,
			Role:           candidate.Role,
			EmbeddingModel: model,
			Embedding:      embedding,
		})
	}

	if len(pending) == 0 {
		return nil
	}
	if err := s.sessions.UpsertTraceEmbeddings(ctx, pending); err != nil {
		return fmt.Errorf("trace cache upsert: %w", err)
	}
	return nil
}

func (s *TraceService) recentEvidence(ctx context.Context, sessionID string, limit int) ([]domain.Memory, error) {
	recentLimit := limit
	if recentLimit > DefaultTraceLimit {
		recentLimit = DefaultTraceLimit
	}

	sessions, err := s.sessions.ListRecentBySessionID(ctx, sessionID, recentLimit)
	if err != nil {
		return nil, fmt.Errorf("trace recent session messages: %w", err)
	}

	results := make([]domain.Memory, 0, len(sessions))
	for i := len(sessions) - 1; i >= 0; i-- {
		sess := sessions[i]
		results = append(results, domain.Memory{
			ID:         sess.ID,
			Content:    sess.Content,
			MemoryType: domain.TypeSession,
			Source:     sess.Source,
			Tags:       sess.Tags,
			AgentID:    sess.AgentID,
			SessionID:  sess.SessionID,
			State:      sess.State,
			CreatedAt:  sess.CreatedAt,
			UpdatedAt:  sess.UpdatedAt,
			Metadata:   mustTraceMetadata(sess.Role, sess.Seq, sess.ContentType),
		})
	}
	return results, nil
}

func normalizeTraceLimit(limit int) int {
	if limit <= 0 {
		return DefaultTraceLimit
	}
	if limit > maxTraceLimit {
		return maxTraceLimit
	}
	return limit
}

func traceEmbeddingModel(sessions *SessionService) string {
	model := strings.TrimSpace(sessions.embedder.Model())
	if model == "" {
		return "unknown"
	}
	return model
}

func selectTraceCandidates(rows []*domain.Session) []*domain.Session {
	preferred := make([]*domain.Session, 0)
	fallback := make([]*domain.Session, 0)
	for _, row := range rows {
		switch {
		case isPreferredTraceCandidate(row):
			preferred = append(preferred, row)
		case isFallbackTraceCandidate(row):
			fallback = append(fallback, row)
		}
	}
	if len(preferred) > 0 {
		return limitTraceCandidates(preferred)
	}
	return limitTraceCandidates(fallback)
}

func limitTraceCandidates(rows []*domain.Session) []*domain.Session {
	if len(rows) <= traceMaxCandidateNodes {
		return rows
	}
	return rows[:traceMaxCandidateNodes]
}

func isPreferredTraceCandidate(row *domain.Session) bool {
	role := normalizeTraceRole(row.Role)
	if role != "user" && role != "assistant" {
		return false
	}
	return !isTraceNoiseContent(row)
}

func isFallbackTraceCandidate(row *domain.Session) bool {
	if isTraceExcludedRole(row.Role) {
		return false
	}
	if isTraceNoiseContent(row) {
		return false
	}
	return containsSemanticText(row.Content)
}

func isTraceExcludedRole(role string) bool {
	switch normalizeTraceRole(role) {
	case "", "system", "tool", "toolresult", "tool_result", "function", "developer":
		return true
	default:
		return false
	}
}

func normalizeTraceRole(role string) string {
	return strings.ToLower(strings.TrimSpace(role))
}

func isTraceNoiseContent(row *domain.Session) bool {
	content := strings.TrimSpace(row.Content)
	if content == "" {
		return true
	}
	if len(content) > traceSkipContentLength {
		return true
	}
	if strings.Count(content, "\n") > traceSkipMultilineThreshold {
		return true
	}
	if row.ContentType == "json" && len(content) > traceSkipStructuredLength {
		return true
	}
	if len(content) > traceSkipStructuredLength && (looksStructured(content) || strings.Contains(content, "```")) {
		return true
	}
	return !containsSemanticText(content)
}

func looksStructured(content string) bool {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return false
	}
	if len(trimmed) > 1 && (trimmed[0] == '{' || trimmed[0] == '[') && json.Valid([]byte(trimmed)) {
		return true
	}
	if strings.Count(trimmed, "{") >= 8 || strings.Count(trimmed, "[") >= 8 {
		return true
	}
	if strings.Count(trimmed, "\t") >= 8 {
		return true
	}
	if strings.Count(trimmed, "http://")+strings.Count(trimmed, "https://") >= 8 {
		return true
	}
	return false
}

func containsSemanticText(content string) bool {
	letters := 0
	for _, r := range content {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			letters++
			if letters >= 12 {
				return true
			}
		}
	}
	return false
}

func newTraceEvidence(mem domain.Memory) TraceEvidence {
	var meta traceMetadata
	if len(mem.Metadata) > 0 {
		_ = json.Unmarshal(mem.Metadata, &meta)
	}

	return TraceEvidence{
		ID:      mem.ID,
		Role:    meta.Role,
		Content: mem.Content,
		Score:   mem.Score,
		Seq:     meta.Seq,
	}
}

func mustTraceMetadata(role string, seq int, contentType string) json.RawMessage {
	meta, _ := json.Marshal(traceMetadata{
		Role:        role,
		Seq:         seq,
		ContentType: contentType,
	})
	return meta
}
