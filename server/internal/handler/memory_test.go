package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/qiffang/mnemos/server/internal/domain"
	"github.com/qiffang/mnemos/server/internal/llm"
	"github.com/qiffang/mnemos/server/internal/middleware"
	"github.com/qiffang/mnemos/server/internal/service"
)

// testMemoryRepo is a minimal MemoryRepo mock for handler tests.
type testMemoryRepo struct {
	createCalls   []*domain.Memory
	getByIDResult *domain.Memory
	getByIDErr    error
}

func (m *testMemoryRepo) Create(_ context.Context, mem *domain.Memory) error {
	m.createCalls = append(m.createCalls, mem)
	return nil
}

func (m *testMemoryRepo) GetByID(_ context.Context, id string) (*domain.Memory, error) {
	if m.getByIDErr != nil || m.getByIDResult != nil {
		return m.getByIDResult, m.getByIDErr
	}
	for _, mem := range m.createCalls {
		if mem.ID == id {
			cp := *mem
			return &cp, nil
		}
	}
	return nil, domain.ErrNotFound
}
func (m *testMemoryRepo) UpdateOptimistic(context.Context, *domain.Memory, int) error { return nil }
func (m *testMemoryRepo) SoftDelete(context.Context, string, string) error            { return nil }
func (m *testMemoryRepo) ArchiveMemory(context.Context, string, string) error         { return nil }
func (m *testMemoryRepo) ArchiveAndCreate(_ context.Context, _, _ string, mem *domain.Memory) error {
	m.createCalls = append(m.createCalls, mem)
	return nil
}
func (m *testMemoryRepo) SetState(context.Context, string, domain.MemoryState) error { return nil }
func (m *testMemoryRepo) List(context.Context, domain.MemoryFilter) ([]domain.Memory, int, error) {
	return nil, 0, nil
}
func (m *testMemoryRepo) Count(context.Context) (int, error)                 { return 0, nil }
func (m *testMemoryRepo) BulkCreate(context.Context, []*domain.Memory) error { return nil }
func (m *testMemoryRepo) VectorSearch(context.Context, []float32, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}

func (m *testMemoryRepo) AutoVectorSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}

func (m *testMemoryRepo) KeywordSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}

func (m *testMemoryRepo) FTSSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (m *testMemoryRepo) FTSAvailable() bool { return false }
func (m *testMemoryRepo) ListBootstrap(context.Context, int) ([]domain.Memory, error) {
	return nil, nil
}

func (m *testMemoryRepo) NearDupSearch(context.Context, string) (string, float64, error) {
	return "", 0, nil
}

func (m *testMemoryRepo) CountStats(context.Context) (int64, int64, error) { return 0, 0, nil }

// testSessionRepo is a minimal SessionRepo mock for handler tests.
type testSessionRepo struct {
	bulkCreateCalled bool
	patchTagsCalled  bool
	sessions         []*domain.Session // captured from BulkCreate
	getByIDResult    *domain.Session
	getByIDErr       error
	sessionRows      []*domain.Session
	recentSessions   []*domain.Session
	keywordResults   []domain.Memory
	keywordErr       error
	ftsAvail         bool
	lastFilter       domain.MemoryFilter
	lastQuery        string
}

func (s *testSessionRepo) BulkCreate(_ context.Context, sessions []*domain.Session) error {
	s.bulkCreateCalled = true
	s.sessions = sessions
	return nil
}
func (s *testSessionRepo) GetByID(context.Context, string) (*domain.Session, error) {
	return s.getByIDResult, s.getByIDErr
}
func (s *testSessionRepo) ListBySessionID(context.Context, string) ([]*domain.Session, error) {
	return s.sessionRows, nil
}
func (s *testSessionRepo) PatchTags(context.Context, string, string, []string) error {
	s.patchTagsCalled = true
	return nil
}

func (s *testSessionRepo) AutoVectorSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}

func (s *testSessionRepo) VectorSearch(context.Context, []float32, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}

func (s *testSessionRepo) FTSSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, nil
}
func (s *testSessionRepo) KeywordSearch(_ context.Context, q string, f domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	s.lastQuery = q
	s.lastFilter = f
	return s.keywordResults, s.keywordErr
}
func (s *testSessionRepo) FTSAvailable() bool { return s.ftsAvail }
func (s *testSessionRepo) ListTraceEmbeddingsBySessionID(context.Context, string) ([]*domain.SessionTraceEmbedding, error) {
	return nil, nil
}
func (s *testSessionRepo) UpsertTraceEmbeddings(context.Context, []*domain.SessionTraceEmbedding) error {
	return nil
}
func (s *testSessionRepo) TraceVectorSearch(context.Context, string, string, []float32, int) ([]domain.Memory, error) {
	return nil, nil
}
func (s *testSessionRepo) ListBySessionIDs(context.Context, []string, int) ([]*domain.Session, error) {
	return nil, nil
}
func (s *testSessionRepo) ListRecentBySessionID(context.Context, string, int) ([]*domain.Session, error) {
	return s.recentSessions, nil
}

// newTestServer creates a Server with pre-populated svcCache for testing.
func newTestServer(memRepo *testMemoryRepo, sessRepo *testSessionRepo) *Server {
	srv := NewServer(nil, nil, "", nil, nil, "", false, service.ModeSmart, "", slog.Default())
	svc := resolvedSvc{
		memory:  service.NewMemoryService(memRepo, nil, nil, "", service.ModeSmart),
		ingest:  service.NewIngestService(memRepo, nil, nil, "", service.ModeSmart),
		session: service.NewSessionService(sessRepo, nil, ""),
	}
	svc.trace = service.NewTraceService(svc.memory, svc.session)
	// Pre-populate svcCache so resolveServices returns our test services.
	// Key format matches resolveServices: fmt.Sprintf("db-%p", auth.TenantDB)
	// When TenantDB is nil, %p formats as "0x0".
	srv.svcCache.Store(tenantSvcKey("db-0x0"), svc)
	return srv
}

func newTestRouter(srv *Server) http.Handler {
	addAuth := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			auth := &domain.AuthInfo{AgentName: "test-agent"}
			ctx := middleware.WithAuthContext(r.Context(), auth)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
	return srv.Router(addAuth, func(next http.Handler) http.Handler { return next }, addAuth)
}

// makeRequest creates an HTTP request with auth context injected.
func makeRequest(t *testing.T, method, path string, body any) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	// Inject auth context using middleware's context key.
	auth := &domain.AuthInfo{AgentName: "test-agent"}
	ctx := middleware.WithAuthContext(req.Context(), auth)
	return req.WithContext(ctx)
}

func TestCreateMemory_SyncContent_Returns200(t *testing.T) {
	srv := newTestServer(&testMemoryRepo{}, &testSessionRepo{})

	body := map[string]any{
		"content": "test memory content",
		"sync":    true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestCreateMemory_AsyncContent_Returns202(t *testing.T) {
	srv := newTestServer(&testMemoryRepo{}, &testSessionRepo{})

	body := map[string]any{
		"content": "test memory content",
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "accepted" {
		t.Errorf("expected status=accepted, got %q", resp["status"])
	}
}

func TestCreateMemory_SyncMessages_Returns200(t *testing.T) {
	memRepo := &testMemoryRepo{}
	srv := newTestServer(memRepo, &testSessionRepo{})

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi there"},
		},
		"session_id": "test-session",
		"sync":       true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if len(memRepo.createCalls) != 1 {
		t.Fatalf("expected 1 raw memory created, got %d", len(memRepo.createCalls))
	}
}

func TestCreateMemory_SyncMessages_ModeRaw_BypassesLLM(t *testing.T) {
	// Mock LLM that always returns 500. Raw mode should bypass it entirely.
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer llmServer.Close()

	llmClient := llm.New(llm.Config{
		APIKey:  "test-key",
		BaseURL: llmServer.URL,
		Model:   "test-model",
	})

	memRepo := &testMemoryRepo{}
	sessRepo := &testSessionRepo{}
	srv := NewServer(nil, nil, "", nil, llmClient, "", false, service.ModeSmart, "", slog.Default())
	svc := resolvedSvc{
		memory:  service.NewMemoryService(memRepo, llmClient, nil, "", service.ModeSmart),
		ingest:  service.NewIngestService(memRepo, llmClient, nil, "", service.ModeSmart),
		session: service.NewSessionService(sessRepo, nil, ""),
	}
	svc.trace = service.NewTraceService(svc.memory, svc.session)
	srv.svcCache.Store(tenantSvcKey("db-0x0"), svc)

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi there"},
		},
		"session_id": "test-session",
		"mode":       "raw",
		"sync":       true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 for raw mode, got %d: %s", rr.Code, rr.Body.String())
	}
	if len(memRepo.createCalls) != 1 {
		t.Fatalf("expected 1 raw memory created, got %d", len(memRepo.createCalls))
	}
	if sessRepo.patchTagsCalled {
		t.Fatalf("did not expect session tag patching in raw mode")
	}
}

func TestCreateMemory_AsyncMessages_Returns202(t *testing.T) {
	srv := newTestServer(&testMemoryRepo{}, &testSessionRepo{})

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello"},
		},
		"session_id": "test-session",
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "accepted" {
		t.Errorf("expected status=accepted, got %q", resp["status"])
	}
}

func TestGetSessionMessage_RouterPaths_ReturnsMessage(t *testing.T) {
	now := time.Date(2026, 4, 6, 12, 0, 0, 0, time.UTC)
	sessRepo := &testSessionRepo{
		getByIDResult: &domain.Session{
			ID:          "node-123",
			SessionID:   "session-1",
			AgentID:     "agent-1",
			Source:      "openclaw",
			Seq:         7,
			Role:        "assistant",
			Content:     "raw text",
			ContentType: "text",
			Tags:        []string{"tag-a"},
			State:       domain.StateActive,
			CreatedAt:   now,
			UpdatedAt:   now,
		},
	}
	router := newTestRouter(newTestServer(&testMemoryRepo{}, sessRepo))

	tests := []string{
		"/v1alpha2/mem9s/session-messages/node-123",
		"/v1alpha1/mem9s/test-tenant/session-messages/node-123",
	}

	for _, path := range tests {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()

		router.ServeHTTP(rr, req)

		if rr.Code != http.StatusOK {
			t.Fatalf("%s: expected 200, got %d: %s", path, rr.Code, rr.Body.String())
		}

		var got sessionMessageResponse
		if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
			t.Fatalf("%s: decode response: %v", path, err)
		}
		if got.ID != "node-123" || got.Content != "raw text" || got.Seq != 7 {
			t.Fatalf("%s: unexpected body: %#v", path, got)
		}
	}
}

func TestGetSessionMessage_NotFound_Returns404(t *testing.T) {
	sessRepo := &testSessionRepo{getByIDErr: domain.ErrNotFound}
	router := newTestRouter(newTestServer(&testMemoryRepo{}, sessRepo))

	req := httptest.NewRequest(http.MethodGet, "/v1alpha2/mem9s/session-messages/missing-node", nil)
	rr := httptest.NewRecorder()

	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", rr.Code, rr.Body.String())
	}

	body, err := io.ReadAll(rr.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if !strings.Contains(string(body), "not found") {
		t.Fatalf("expected not found body, got %s", string(body))
	}
}

func TestGetMemoryTrace_RouterPaths_ReturnsTrace(t *testing.T) {
	memRepo := &testMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:         "mem-123",
			Content:    "abstract memory",
			MemoryType: domain.TypeInsight,
			SessionID:  "session-1",
			State:      domain.StateActive,
		},
	}
	sessRepo := &testSessionRepo{
		recentSessions: []*domain.Session{{
			ID:          "raw-node-1",
			SessionID:   "session-1",
			Role:        "user",
			Content:     "用户原始问题",
			ContentType: "text",
			Seq:         12,
			State:       domain.StateActive,
		}},
	}
	router := newTestRouter(newTestServer(memRepo, sessRepo))

	tests := []string{
		"/v1alpha2/mem9s/memories/mem-123/trace?q=%E4%B8%89%E4%B8%AA%E9%97%AE%E9%A2%98%E6%98%AF%E4%BB%80%E4%B9%88&limit=2",
		"/v1alpha1/mem9s/test-tenant/memories/mem-123/trace?q=%E4%B8%89%E4%B8%AA%E9%97%AE%E9%A2%98%E6%98%AF%E4%BB%80%E4%B9%88&limit=2",
	}

	for _, path := range tests {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()

		router.ServeHTTP(rr, req)

		if rr.Code != http.StatusOK {
			t.Fatalf("%s: expected 200, got %d: %s", path, rr.Code, rr.Body.String())
		}

		var got service.MemoryTrace
		if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
			t.Fatalf("%s: decode response: %v", path, err)
		}
		if got.SessionID != "session-1" || got.Query != "三个问题是什么" {
			t.Fatalf("%s: unexpected trace envelope: %#v", path, got)
		}
		if len(got.Evidence) != 1 {
			t.Fatalf("%s: expected 1 evidence row, got %d", path, len(got.Evidence))
		}
		if got.Evidence[0].ID != "raw-node-1" || got.Evidence[0].Role != "user" || got.Evidence[0].Seq != 12 {
			t.Fatalf("%s: unexpected evidence: %#v", path, got.Evidence[0])
		}
	}
}

func TestGetMemoryTrace_MemoryWithoutSessionID_Returns501(t *testing.T) {
	memRepo := &testMemoryRepo{
		getByIDResult: &domain.Memory{
			ID:      "mem-123",
			Content: "abstract memory",
		},
	}
	router := newTestRouter(newTestServer(memRepo, &testSessionRepo{}))

	req := httptest.NewRequest(http.MethodGet, "/v1alpha2/mem9s/memories/mem-123/trace", nil)
	rr := httptest.NewRecorder()

	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d: %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "session provenance") {
		t.Fatalf("expected provenance error, got %s", rr.Body.String())
	}
}

// failSearchMemoryRepo embeds testMemoryRepo but makes KeywordSearch fail,
// triggering gatherExistingMemories → reconcile → ReconcilePhase2 Status:"failed".
type failSearchMemoryRepo struct {
	testMemoryRepo
}

func (m *failSearchMemoryRepo) KeywordSearch(context.Context, string, domain.MemoryFilter, int) ([]domain.Memory, error) {
	return nil, errors.New("simulated search failure")
}

func TestCreateMemory_SyncMessages_Phase1Error_Returns500(t *testing.T) {
	// Mock LLM that always returns 500.
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer llmServer.Close()

	llmClient := llm.New(llm.Config{
		APIKey:  "test-key",
		BaseURL: llmServer.URL,
		Model:   "test-model",
	})

	srv := NewServer(nil, nil, "", nil, llmClient, "", false, service.ModeSmart, "", slog.Default())
	svc := resolvedSvc{
		memory:  service.NewMemoryService(&testMemoryRepo{}, nil, nil, "", service.ModeSmart),
		ingest:  service.NewIngestService(&testMemoryRepo{}, llmClient, nil, "", service.ModeSmart),
		session: service.NewSessionService(&testSessionRepo{}, nil, ""),
	}
	svc.trace = service.NewTraceService(svc.memory, svc.session)
	srv.svcCache.Store(tenantSvcKey("db-0x0"), svc)

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello"},
		},
		"session_id": "test-session",
		"sync":       true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestCreateMemory_SyncMessages_StripsInjectedContext(t *testing.T) {
	// Mock LLM that captures request bodies to verify no injected context reaches the LLM.
	var llmBodies []string
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		llmBodies = append(llmBodies, string(bodyBytes))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"facts":["hello world"],"message_tags":[["greeting"],["reply"]]}`,
				}},
			},
		})
	}))
	defer llmServer.Close()

	llmClient := llm.New(llm.Config{
		APIKey:  "test-key",
		BaseURL: llmServer.URL,
		Model:   "test-model",
	})

	sessRepo := &testSessionRepo{}
	srv := NewServer(nil, nil, "", nil, llmClient, "", false, service.ModeSmart, "", slog.Default())
	svc := resolvedSvc{
		memory:  service.NewMemoryService(&testMemoryRepo{}, nil, nil, "", service.ModeSmart),
		ingest:  service.NewIngestService(&testMemoryRepo{}, llmClient, nil, "", service.ModeSmart),
		session: service.NewSessionService(sessRepo, nil, ""),
	}
	svc.trace = service.NewTraceService(svc.memory, svc.session)
	srv.svcCache.Store(tenantSvcKey("db-0x0"), svc)

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello <relevant-memories>\ninjected memory content\n</relevant-memories> world"},
			{"role": "assistant", "content": "hi there"},
		},
		"session_id": "test-session",
		"sync":       true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	// Verify sessions stored via BulkCreate have injected context stripped.
	for _, sess := range sessRepo.sessions {
		if strings.Contains(sess.Content, "<relevant-memories>") {
			t.Errorf("session content still contains injected context: %s", sess.Content)
		}
		if strings.Contains(sess.Content, "injected memory content") {
			t.Errorf("session content still contains injected memory: %s", sess.Content)
		}
	}

	// Verify LLM prompts (ExtractPhase1) don't contain injected context.
	if len(llmBodies) == 0 {
		t.Fatal("expected at least one LLM request, got none")
	}
	for i, llmBody := range llmBodies {
		if strings.Contains(llmBody, "<relevant-memories>") {
			t.Errorf("LLM request %d still contains injected context tag", i)
		}
		if strings.Contains(llmBody, "injected memory content") {
			t.Errorf("LLM request %d still contains injected memory content", i)
		}
	}
}

func TestCreateMemory_SyncMessages_ReconcileFailure_Returns500(t *testing.T) {
	// Mock LLM that returns valid facts for ExtractPhase1.
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"facts":["test fact"],"message_tags":[["tag1"],["tag2"]]}`,
				}},
			},
		})
	}))
	defer llmServer.Close()

	llmClient := llm.New(llm.Config{
		APIKey:  "test-key",
		BaseURL: llmServer.URL,
		Model:   "test-model",
	})

	memRepo := &failSearchMemoryRepo{}
	srv := NewServer(nil, nil, "", nil, llmClient, "", false, service.ModeSmart, "", slog.Default())
	svc := resolvedSvc{
		memory:  service.NewMemoryService(&memRepo.testMemoryRepo, nil, nil, "", service.ModeSmart),
		ingest:  service.NewIngestService(memRepo, llmClient, nil, "", service.ModeSmart),
		session: service.NewSessionService(&testSessionRepo{}, nil, ""),
	}
	svc.trace = service.NewTraceService(svc.memory, svc.session)
	srv.svcCache.Store(tenantSvcKey("db-0x0"), svc)

	body := map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi there"},
		},
		"session_id": "test-session",
		"sync":       true,
	}
	req := makeRequest(t, http.MethodPost, "/memories", body)
	rr := httptest.NewRecorder()

	srv.createMemory(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
}
