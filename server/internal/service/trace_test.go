package service

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/qiffang/mnemos/server/internal/domain"
	"github.com/qiffang/mnemos/server/internal/embed"
)

type traceMemoryRepo struct {
	getByIDResult *domain.Memory
	getByIDErr    error
}

func (m *traceMemoryRepo) Create(context.Context, *domain.Memory) error { return nil }
func (m *traceMemoryRepo) GetByID(context.Context, string) (*domain.Memory, error) {
	return m.getByIDResult, m.getByIDErr
}
func (m *traceMemoryRepo) UpdateOptimistic(context.Context, *domain.Memory, int) error { return nil }
func (m *traceMemoryRepo) SoftDelete(context.Context, string, string) error            { return nil }
func (m *traceMemoryRepo) ArchiveMemory(context.Context, string, string) error         { return nil }
func (m *traceMemoryRepo) ArchiveAndCreate(context.Context, string, string, *domain.Memory) error {
	return nil
}
func (m *traceMemoryRepo) SetState(context.Context, string, domain.MemoryState) error { return nil }
func (m *traceMemoryRepo) List(context.Context, domain.MemoryFilter) ([]domain.Memory, int, error) {
	return nil, 0, nil
}
func (m *traceMemoryRepo) Count(context.Context) (int, error)                 { return 0, nil }
func (m *traceMemoryRepo) BulkCreate(context.Context, []*domain.Memory) error { return nil }
func (m *traceMemoryRepo) VectorSearch(context.Context, []float32, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *traceMemoryRepo) AutoVectorSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *traceMemoryRepo) KeywordSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *traceMemoryRepo) FTSSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *traceMemoryRepo) FTSAvailable() bool { return false }
func (m *traceMemoryRepo) ListBootstrap(context.Context, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *traceMemoryRepo) NearDupSearch(context.Context, string) (string, float64, error) {
	return "", 0, nil
}
func (m *traceMemoryRepo) CountStats(context.Context) (int64, int64, error) { return 0, 0, nil }

func TestTraceService_UsesSemanticSessionCacheAndPopulatesLazily(t *testing.T) {
	embedSrv := newTraceEmbeddingServer(map[string][]float32{
		"the user had three questions":                        {1, 0},
		"the user asked about pricing, retention, conversion": {1, 0},
		"we discussed pricing, retention, and conversion":     {0.8, 0.2},
	})
	defer embedSrv.Close()

	memRepo := &traceMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:        "mem-1",
			Content:   "the user had three questions",
			SessionID: "sess-1",
		},
	}
	sessionRepo := &stubSessionRepo{
		sessionRows: []*domain.Session{
			{
				ID:          "raw-1",
				SessionID:   "sess-1",
				Role:        "user",
				Content:     "the user asked about pricing, retention, conversion",
				ContentType: "text",
				ContentHash: "hash-1",
				Seq:         12,
				State:       domain.StateActive,
			},
			{
				ID:          "raw-2",
				SessionID:   "sess-1",
				Role:        "assistant",
				Content:     "we discussed pricing, retention, and conversion",
				ContentType: "text",
				ContentHash: "hash-2",
				Seq:         13,
				State:       domain.StateActive,
			},
			{
				ID:          "raw-3",
				SessionID:   "sess-1",
				Role:        "toolResult",
				Content:     longTraceNoise("api log line"),
				ContentType: "text",
				ContentHash: "hash-3",
				Seq:         14,
				State:       domain.StateActive,
			},
		},
	}

	svc := NewTraceService(
		&MemoryService{memories: memRepo},
		NewSessionService(sessionRepo, embedSrv.Embedder(), ""),
	)

	trace, err := svc.Trace(context.Background(), "mem-1", "", 0)
	if err != nil {
		t.Fatalf("Trace() error: %v", err)
	}

	if trace.Query != "the user had three questions" {
		t.Fatalf("trace.Query = %q", trace.Query)
	}
	if len(trace.Evidence) != 2 {
		t.Fatalf("len(trace.Evidence) = %d", len(trace.Evidence))
	}
	if trace.Evidence[0].ID != "raw-1" || trace.Evidence[0].Role != "user" || trace.Evidence[0].Seq != 12 {
		t.Fatalf("unexpected first evidence: %#v", trace.Evidence[0])
	}
	if sessionRepo.traceUpsertCalls != 1 {
		t.Fatalf("traceUpsertCalls = %d", sessionRepo.traceUpsertCalls)
	}
	if len(sessionRepo.traceUpserted) != 2 {
		t.Fatalf("len(traceUpserted) = %d", len(sessionRepo.traceUpserted))
	}
	if embedSrv.CallCount("the user asked about pricing, retention, conversion") != 1 {
		t.Fatalf("expected one node embedding call for raw-1, got %d", embedSrv.CallCount("the user asked about pricing, retention, conversion"))
	}
	if embedSrv.CallCount("we discussed pricing, retention, and conversion") != 1 {
		t.Fatalf("expected one node embedding call for raw-2, got %d", embedSrv.CallCount("we discussed pricing, retention, and conversion"))
	}
	if embedSrv.CallCount(longTraceNoise("api log line")) != 0 {
		t.Fatalf("expected noisy tool result to be skipped, got %d calls", embedSrv.CallCount(longTraceNoise("api log line")))
	}

	_, err = svc.Trace(context.Background(), "mem-1", "", 0)
	if err != nil {
		t.Fatalf("Trace() second call error: %v", err)
	}
	if sessionRepo.traceUpsertCalls != 1 {
		t.Fatalf("expected cache reuse without new upsert, got %d calls", sessionRepo.traceUpsertCalls)
	}
	if embedSrv.CallCount("the user asked about pricing, retention, conversion") != 1 {
		t.Fatalf("expected cached node embedding reuse, got %d calls", embedSrv.CallCount("the user asked about pricing, retention, conversion"))
	}
}

func TestTraceService_FallsBackFromExplicitQueryToMemoryContent(t *testing.T) {
	embedSrv := newTraceEmbeddingServer(map[string][]float32{
		"what were the three questions":               {0, 1},
		"the third question was the conversion path":  {1, 0},
		"pricing, retention, and the conversion path": {1, 0},
	})
	defer embedSrv.Close()

	memRepo := &traceMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:        "mem-1",
			Content:   "the third question was the conversion path",
			SessionID: "sess-1",
		},
	}
	sessionRepo := &stubSessionRepo{
		sessionRows: []*domain.Session{{
			ID:          "raw-2",
			SessionID:   "sess-1",
			Role:        "user",
			Content:     "pricing, retention, and the conversion path",
			ContentType: "text",
			ContentHash: "hash-2",
			Seq:         4,
			State:       domain.StateActive,
		}},
	}

	svc := NewTraceService(
		&MemoryService{memories: memRepo},
		NewSessionService(sessionRepo, embedSrv.Embedder(), ""),
	)

	trace, err := svc.Trace(context.Background(), "mem-1", "what were the three questions", 3)
	if err != nil {
		t.Fatalf("Trace() error: %v", err)
	}

	if trace.Query != "what were the three questions" {
		t.Fatalf("trace.Query = %q", trace.Query)
	}
	if len(trace.Evidence) != 1 {
		t.Fatalf("len(trace.Evidence) = %d", len(trace.Evidence))
	}
	if trace.Evidence[0].ID != "raw-2" || trace.Evidence[0].Seq != 4 || trace.Evidence[0].Role != "user" {
		t.Fatalf("unexpected evidence: %#v", trace.Evidence[0])
	}
	if embedSrv.CallCount("what were the three questions") != 1 {
		t.Fatalf("expected explicit query embedding once, got %d", embedSrv.CallCount("what were the three questions"))
	}
	if embedSrv.CallCount("the third question was the conversion path") != 1 {
		t.Fatalf("expected memory-content fallback embedding once, got %d", embedSrv.CallCount("the third question was the conversion path"))
	}
}

func TestTraceService_PrefersUserAssistantTextOverToolResultNoise(t *testing.T) {
	embedSrv := newTraceEmbeddingServer(map[string][]float32{
		"where did the user mention the launch date": {1, 0},
		"the user said the launch date is april 30":  {1, 0},
	})
	defer embedSrv.Close()

	memRepo := &traceMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:        "mem-1",
			Content:   "where did the user mention the launch date",
			SessionID: "sess-1",
		},
	}
	sessionRepo := &stubSessionRepo{
		sessionRows: []*domain.Session{
			{
				ID:          "raw-user",
				SessionID:   "sess-1",
				Role:        "user",
				Content:     "the user said the launch date is april 30",
				ContentType: "text",
				ContentHash: "hash-user",
				Seq:         2,
				State:       domain.StateActive,
			},
			{
				ID:          "raw-tool",
				SessionID:   "sess-1",
				Role:        "toolResult",
				Content:     longTraceNoise("launch-date-tool-output"),
				ContentType: "text",
				ContentHash: "hash-tool",
				Seq:         3,
				State:       domain.StateActive,
			},
		},
	}

	svc := NewTraceService(
		&MemoryService{memories: memRepo},
		NewSessionService(sessionRepo, embedSrv.Embedder(), ""),
	)

	trace, err := svc.Trace(context.Background(), "mem-1", "", 2)
	if err != nil {
		t.Fatalf("Trace() error: %v", err)
	}

	if len(trace.Evidence) != 1 {
		t.Fatalf("len(trace.Evidence) = %d", len(trace.Evidence))
	}
	if trace.Evidence[0].ID != "raw-user" {
		t.Fatalf("expected user evidence, got %#v", trace.Evidence[0])
	}
	if embedSrv.CallCount(longTraceNoise("launch-date-tool-output")) != 0 {
		t.Fatalf("expected tool result noise to be skipped, got %d calls", embedSrv.CallCount(longTraceNoise("launch-date-tool-output")))
	}
}

func TestTraceService_FallsBackToRecentSessionRows(t *testing.T) {
	now := time.Date(2026, 4, 6, 12, 0, 0, 0, time.UTC)
	memRepo := &traceMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:        "mem-1",
			Content:   "the third question was the conversion path",
			SessionID: "sess-1",
		},
	}
	sessionRepo := &stubSessionRepo{
		sessionRows: []*domain.Session{{
			ID:          "raw-noise",
			SessionID:   "sess-1",
			Role:        "toolResult",
			Content:     longTraceNoise("noisy payload"),
			ContentType: "text",
			ContentHash: "hash-noise",
			Seq:         7,
			State:       domain.StateActive,
		}},
		recentSessions: []*domain.Session{
			{
				ID:          "raw-9",
				SessionID:   "sess-1",
				Role:        "assistant",
				Content:     "this is the latest answer",
				ContentType: "text",
				Seq:         9,
				State:       domain.StateActive,
				CreatedAt:   now.Add(2 * time.Minute),
				UpdatedAt:   now.Add(2 * time.Minute),
			},
			{
				ID:          "raw-8",
				SessionID:   "sess-1",
				Role:        "user",
				Content:     "this is the second-latest question",
				ContentType: "text",
				Seq:         8,
				State:       domain.StateActive,
				CreatedAt:   now.Add(time.Minute),
				UpdatedAt:   now.Add(time.Minute),
			},
		},
	}

	svc := NewTraceService(
		&MemoryService{memories: memRepo},
		NewSessionService(sessionRepo, nil, ""),
	)

	trace, err := svc.Trace(context.Background(), "mem-1", "what were the three questions", 7)
	if err != nil {
		t.Fatalf("Trace() error: %v", err)
	}

	if sessionRepo.recentSessionID != "sess-1" {
		t.Fatalf("recentSessionID = %q", sessionRepo.recentSessionID)
	}
	if sessionRepo.recentLimit != DefaultTraceLimit {
		t.Fatalf("recentLimit = %d", sessionRepo.recentLimit)
	}
	if len(trace.Evidence) != 2 {
		t.Fatalf("len(trace.Evidence) = %d", len(trace.Evidence))
	}
	if trace.Evidence[0].ID != "raw-8" || trace.Evidence[0].Seq != 8 || trace.Evidence[0].Role != "user" {
		t.Fatalf("unexpected first evidence: %#v", trace.Evidence[0])
	}
	if trace.Evidence[1].ID != "raw-9" || trace.Evidence[1].Seq != 9 || trace.Evidence[1].Role != "assistant" {
		t.Fatalf("unexpected second evidence: %#v", trace.Evidence[1])
	}
}

func TestTraceService_ReturnsNotSupportedWhenMemoryHasNoSessionID(t *testing.T) {
	memRepo := &traceMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:      "mem-1",
			Content: "abstract memory",
		},
	}

	svc := NewTraceService(
		&MemoryService{memories: memRepo},
		NewSessionService(&stubSessionRepo{}, nil, ""),
	)

	_, err := svc.Trace(context.Background(), "mem-1", "", 0)
	if !errors.Is(err, domain.ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported, got %v", err)
	}
}

type traceEmbeddingServer struct {
	server  *httptest.Server
	mu      sync.Mutex
	vectors map[string][]float32
	calls   map[string]int
}

func newTraceEmbeddingServer(vectors map[string][]float32) *traceEmbeddingServer {
	s := &traceEmbeddingServer{
		vectors: vectors,
		calls:   make(map[string]int),
	}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		s.mu.Lock()
		s.calls[req.Input]++
		vector := append([]float32(nil), s.vectors[req.Input]...)
		s.mu.Unlock()

		if len(vector) == 0 {
			vector = []float32{0, 0}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{
				"embedding": vector,
			}},
		})
	}))
	return s
}

func (s *traceEmbeddingServer) Close() {
	s.server.Close()
}

func (s *traceEmbeddingServer) Embedder() *embed.Embedder {
	return embed.New(embed.Config{
		APIKey:  "test-key",
		BaseURL: s.server.URL,
		Model:   "trace-test-model",
		Dims:    2,
	})
}

func (s *traceEmbeddingServer) CallCount(text string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[text]
}

func longTraceNoise(prefix string) string {
	return prefix + "\n" + repeatLine(`{"level":"debug","payload":"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}`, 160)
}

func repeatLine(line string, count int) string {
	out := ""
	for i := 0; i < count; i++ {
		out += line + "\n"
	}
	return out
}
