package service

import (
	"context"
	"errors"
	"math"
	"sort"
	"testing"

	"github.com/qiffang/mnemos/server/internal/domain"
)

type stubSessionRepo struct {
	bulkCreateCalled bool
	bulkCreateErr    error
	createdSessions  []*domain.Session
	getByIDResult    *domain.Session
	getByIDErr       error
	sessionRows      []*domain.Session
	sessionRowsErr   error
	recentSessions   []*domain.Session
	recentErr        error
	recentSessionID  string
	recentLimit      int

	patchTagsCalled bool
	patchTagsErr    error
	patchedSession  string
	patchedHash     string
	patchedTags     []string

	keywordResults       []domain.Memory
	keywordResultsByCall [][]domain.Memory
	keywordErr           error
	keywordQueries       []string
	ftsResults           []domain.Memory
	ftsErr               error
	vecResults           []domain.Memory
	vecErr               error
	autoVecResults       []domain.Memory
	autoVecErr           error
	ftsAvail             bool
	traceEmbeddings      map[string]*domain.SessionTraceEmbedding
	traceSearchErr       error
	traceSearchSessionID string
	traceSearchModel     string
	traceSearchLimit     int
	traceUpsertCalls     int
	traceUpserted        []*domain.SessionTraceEmbedding
}

func (s *stubSessionRepo) BulkCreate(_ context.Context, sessions []*domain.Session) error {
	s.bulkCreateCalled = true
	s.createdSessions = sessions
	return s.bulkCreateErr
}

func (s *stubSessionRepo) GetByID(_ context.Context, _ string) (*domain.Session, error) {
	return s.getByIDResult, s.getByIDErr
}

func (s *stubSessionRepo) ListBySessionID(_ context.Context, _ string) ([]*domain.Session, error) {
	return s.sessionRows, s.sessionRowsErr
}

func (s *stubSessionRepo) PatchTags(_ context.Context, sessionID, contentHash string, tags []string) error {
	s.patchTagsCalled = true
	s.patchedSession = sessionID
	s.patchedHash = contentHash
	s.patchedTags = tags
	return s.patchTagsErr
}

func (s *stubSessionRepo) AutoVectorSearch(_ context.Context, _ string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return s.autoVecResults, s.autoVecErr
}

func (s *stubSessionRepo) VectorSearch(_ context.Context, _ []float32, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return s.vecResults, s.vecErr
}

func (s *stubSessionRepo) FTSSearch(_ context.Context, _ string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return s.ftsResults, s.ftsErr
}

func (s *stubSessionRepo) KeywordSearch(_ context.Context, q string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	s.keywordQueries = append(s.keywordQueries, q)
	if len(s.keywordResultsByCall) >= len(s.keywordQueries) {
		return s.keywordResultsByCall[len(s.keywordQueries)-1], s.keywordErr
	}
	return s.keywordResults, s.keywordErr
}

func (s *stubSessionRepo) FTSAvailable() bool { return s.ftsAvail }

func (s *stubSessionRepo) ListTraceEmbeddingsBySessionID(_ context.Context, _ string) ([]*domain.SessionTraceEmbedding, error) {
	if len(s.traceEmbeddings) == 0 {
		return nil, nil
	}
	result := make([]*domain.SessionTraceEmbedding, 0, len(s.traceEmbeddings))
	for _, entry := range s.traceEmbeddings {
		cp := *entry
		cp.Embedding = nil
		result = append(result, &cp)
	}
	return result, nil
}

func (s *stubSessionRepo) UpsertTraceEmbeddings(_ context.Context, entries []*domain.SessionTraceEmbedding) error {
	s.traceUpsertCalls++
	if s.traceEmbeddings == nil {
		s.traceEmbeddings = make(map[string]*domain.SessionTraceEmbedding)
	}
	for _, entry := range entries {
		cp := *entry
		cp.Embedding = append([]float32(nil), entry.Embedding...)
		s.traceEmbeddings[entry.NodeID] = &cp
		s.traceUpserted = append(s.traceUpserted, &cp)
	}
	return nil
}

func (s *stubSessionRepo) TraceVectorSearch(_ context.Context, sessionID, embeddingModel string, queryVec []float32, limit int) ([]domain.Memory, error) {
	if s.traceSearchErr != nil {
		return nil, s.traceSearchErr
	}
	s.traceSearchSessionID = sessionID
	s.traceSearchModel = embeddingModel
	s.traceSearchLimit = limit

	type scored struct {
		mem   domain.Memory
		score float64
	}
	scoredRows := make([]scored, 0, len(s.sessionRows))
	for _, row := range s.sessionRows {
		entry := s.traceEmbeddings[row.ID]
		if entry == nil || entry.SessionID != sessionID || entry.EmbeddingModel != embeddingModel {
			continue
		}
		score := cosineSimilarity(queryVec, entry.Embedding)
		mem := domain.Memory{
			ID:         row.ID,
			Content:    row.Content,
			MemoryType: domain.TypeSession,
			Source:     row.Source,
			Tags:       row.Tags,
			AgentID:    row.AgentID,
			SessionID:  row.SessionID,
			State:      row.State,
			CreatedAt:  row.CreatedAt,
			UpdatedAt:  row.UpdatedAt,
			Metadata:   mustTraceMetadata(row.Role, row.Seq, row.ContentType),
		}
		mem.Score = &score
		scoredRows = append(scoredRows, scored{mem: mem, score: score})
	}

	sort.SliceStable(scoredRows, func(i, j int) bool {
		if scoredRows[i].score == scoredRows[j].score {
			return scoredRows[i].mem.ID < scoredRows[j].mem.ID
		}
		return scoredRows[i].score > scoredRows[j].score
	})

	if limit > len(scoredRows) {
		limit = len(scoredRows)
	}
	result := make([]domain.Memory, 0, limit)
	for i := 0; i < limit; i++ {
		result = append(result, scoredRows[i].mem)
	}
	return result, nil
}

func (s *stubSessionRepo) ListBySessionIDs(_ context.Context, _ []string, _ int) ([]*domain.Session, error) {
	return nil, nil
}

func (s *stubSessionRepo) ListRecentBySessionID(_ context.Context, sessionID string, limit int) ([]*domain.Session, error) {
	s.recentSessionID = sessionID
	s.recentLimit = limit
	return s.recentSessions, s.recentErr
}

func newTestSessionService(repo *stubSessionRepo) *SessionService {
	return NewSessionService(repo, nil, "")
}

func TestSessionService_BulkCreate_buildsCorrectSessions(t *testing.T) {
	repo := &stubSessionRepo{}
	svc := newTestSessionService(repo)

	req := IngestRequest{
		SessionID: "sess-1",
		AgentID:   "agent-x",
		Messages: []IngestMessage{
			{Role: "user", Content: "Hello world"},
			{Role: "assistant", Content: "Hi there"},
		},
	}

	if err := svc.BulkCreate(context.Background(), "source-agent", req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !repo.bulkCreateCalled {
		t.Fatal("expected BulkCreate to be called")
	}
	if len(repo.createdSessions) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(repo.createdSessions))
	}

	s0 := repo.createdSessions[0]
	if s0.SessionID != "sess-1" {
		t.Errorf("session[0].SessionID = %q, want %q", s0.SessionID, "sess-1")
	}
	if s0.AgentID != "agent-x" {
		t.Errorf("session[0].AgentID = %q, want %q", s0.AgentID, "agent-x")
	}
	if s0.Role != "user" {
		t.Errorf("session[0].Role = %q, want %q", s0.Role, "user")
	}
	if s0.Seq != 0 {
		t.Errorf("session[0].Seq = %d, want 0", s0.Seq)
	}
	if s0.Content != "Hello world" {
		t.Errorf("session[0].Content = %q, want %q", s0.Content, "Hello world")
	}
	if s0.ContentHash == "" {
		t.Error("session[0].ContentHash must not be empty")
	}

	s1 := repo.createdSessions[1]
	if s1.Seq != 1 {
		t.Errorf("session[1].Seq = %d, want 1", s1.Seq)
	}
	if s1.Role != "assistant" {
		t.Errorf("session[1].Role = %q, want %q", s1.Role, "assistant")
	}

	if s0.ContentHash == s1.ContentHash {
		t.Error("different messages must produce different content hashes")
	}
}

func TestSessionService_BulkCreate_emptyMessages(t *testing.T) {
	repo := &stubSessionRepo{}
	svc := newTestSessionService(repo)

	req := IngestRequest{SessionID: "sess-1", Messages: []IngestMessage{}}
	if err := svc.BulkCreate(context.Background(), "src", req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo.bulkCreateCalled && len(repo.createdSessions) != 0 {
		t.Error("expected no sessions created for empty messages")
	}
}

func TestSessionService_BulkCreate_propagatesRepoError(t *testing.T) {
	sentinel := errors.New("db down")
	repo := &stubSessionRepo{bulkCreateErr: sentinel}
	svc := newTestSessionService(repo)

	req := IngestRequest{
		SessionID: "s",
		Messages:  []IngestMessage{{Role: "user", Content: "hi"}},
	}
	err := svc.BulkCreate(context.Background(), "src", req)
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

func TestSessionService_PatchTags_delegates(t *testing.T) {
	repo := &stubSessionRepo{}
	svc := newTestSessionService(repo)

	tags := []string{"tech", "question"}
	if err := svc.PatchTags(context.Background(), "sess-1", "hashval", tags); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !repo.patchTagsCalled {
		t.Fatal("expected PatchTags to be called on repo")
	}
	if repo.patchedSession != "sess-1" {
		t.Errorf("patchedSession = %q, want %q", repo.patchedSession, "sess-1")
	}
	if repo.patchedHash != "hashval" {
		t.Errorf("patchedHash = %q, want %q", repo.patchedHash, "hashval")
	}
	if len(repo.patchedTags) != 2 || repo.patchedTags[0] != "tech" {
		t.Errorf("patchedTags = %v, want [tech question]", repo.patchedTags)
	}
}

func TestSessionService_PatchTags_propagatesError(t *testing.T) {
	sentinel := errors.New("patch fail")
	repo := &stubSessionRepo{patchTagsErr: sentinel}
	svc := newTestSessionService(repo)

	err := svc.PatchTags(context.Background(), "s", "h", []string{"t"})
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

func TestSessionService_GetByID_delegates(t *testing.T) {
	expected := &domain.Session{ID: "msg-1", Content: "hello"}
	repo := &stubSessionRepo{getByIDResult: expected}
	svc := newTestSessionService(repo)

	got, err := svc.GetByID(context.Background(), "msg-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil || got.ID != expected.ID || got.Content != expected.Content {
		t.Fatalf("GetByID = %#v, want %#v", got, expected)
	}
}

func TestSessionService_GetByID_propagatesError(t *testing.T) {
	sentinel := errors.New("lookup fail")
	repo := &stubSessionRepo{getByIDErr: sentinel}
	svc := newTestSessionService(repo)

	_, err := svc.GetByID(context.Background(), "msg-1")
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected sentinel error, got %v", err)
	}
}

func TestSessionService_Search_keywordPath_returnsSessionType(t *testing.T) {
	mem := domain.Memory{
		ID:         "m1",
		Content:    "hello",
		MemoryType: domain.TypeSession,
		State:      domain.StateActive,
	}
	repo := &stubSessionRepo{
		keywordResults: []domain.Memory{mem},
		ftsAvail:       false,
	}
	svc := newTestSessionService(repo)

	f := domain.MemoryFilter{Query: "hello", Limit: 5}
	results, err := svc.Search(context.Background(), f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].MemoryType != domain.TypeSession {
		t.Errorf("memory_type = %q, want %q", results[0].MemoryType, domain.TypeSession)
	}
}

func TestSessionService_Search_offsetZeroedBeforeRepo(t *testing.T) {
	var capturedFilter domain.MemoryFilter
	repo := &stubSessionRepo{
		keywordResults: []domain.Memory{},
		ftsAvail:       false,
	}
	repo.keywordResults = nil

	capturingRepo := &capturingSessionRepo{stub: repo, capturedFilter: &capturedFilter}
	svc := NewSessionService(capturingRepo, nil, "")

	f := domain.MemoryFilter{Query: "x", Limit: 10, Offset: 5}
	if _, err := svc.Search(context.Background(), f); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedFilter.Offset != 0 {
		t.Errorf("filter.Offset passed to repo = %d, want 0 (sessions reset offset)", capturedFilter.Offset)
	}
}

func TestSessionService_Search_defaultLimit(t *testing.T) {
	repo := &stubSessionRepo{ftsAvail: false}
	svc := newTestSessionService(repo)

	_, err := svc.Search(context.Background(), domain.MemoryFilter{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSessionContentHash_differentInputsProduceDifferentHashes(t *testing.T) {
	cases := [][2]string{
		{"sess-a role-user content-x", "sess-a role-user content-y"},
		{"sess-a role-user content-x", "sess-b role-user content-x"},
		{"sess-a role-user content-x", "sess-a role-assistant content-x"},
	}
	for _, c := range cases {
		h1 := SessionContentHash("sess-a", "user", c[0])
		h2 := SessionContentHash("sess-a", "user", c[1])
		if h1 == h2 {
			t.Errorf("expected different hashes for different inputs: %q vs %q", c[0], c[1])
		}
	}
}

func TestSessionContentHash_sameInputProducesSameHash(t *testing.T) {
	h1 := SessionContentHash("sess-1", "user", "hello world")
	h2 := SessionContentHash("sess-1", "user", "hello world")
	if h1 != h2 {
		t.Errorf("expected identical hashes, got %q vs %q", h1, h2)
	}
}

type capturingSessionRepo struct {
	stub           *stubSessionRepo
	capturedFilter *domain.MemoryFilter
}

func (c *capturingSessionRepo) BulkCreate(ctx context.Context, s []*domain.Session) error {
	return c.stub.BulkCreate(ctx, s)
}
func (c *capturingSessionRepo) GetByID(ctx context.Context, id string) (*domain.Session, error) {
	return c.stub.GetByID(ctx, id)
}
func (c *capturingSessionRepo) ListBySessionID(ctx context.Context, sessionID string) ([]*domain.Session, error) {
	return c.stub.ListBySessionID(ctx, sessionID)
}
func (c *capturingSessionRepo) PatchTags(ctx context.Context, sid, hash string, tags []string) error {
	return c.stub.PatchTags(ctx, sid, hash, tags)
}
func (c *capturingSessionRepo) AutoVectorSearch(ctx context.Context, q string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	*c.capturedFilter = f
	return c.stub.AutoVectorSearch(ctx, q, f, limit)
}
func (c *capturingSessionRepo) VectorSearch(ctx context.Context, v []float32, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	*c.capturedFilter = f
	return c.stub.VectorSearch(ctx, v, f, limit)
}
func (c *capturingSessionRepo) FTSSearch(ctx context.Context, q string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	*c.capturedFilter = f
	return c.stub.FTSSearch(ctx, q, f, limit)
}
func (c *capturingSessionRepo) KeywordSearch(ctx context.Context, q string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	*c.capturedFilter = f
	return c.stub.KeywordSearch(ctx, q, f, limit)
}
func (c *capturingSessionRepo) FTSAvailable() bool { return c.stub.FTSAvailable() }
func (c *capturingSessionRepo) ListTraceEmbeddingsBySessionID(ctx context.Context, sessionID string) ([]*domain.SessionTraceEmbedding, error) {
	return c.stub.ListTraceEmbeddingsBySessionID(ctx, sessionID)
}
func (c *capturingSessionRepo) UpsertTraceEmbeddings(ctx context.Context, entries []*domain.SessionTraceEmbedding) error {
	return c.stub.UpsertTraceEmbeddings(ctx, entries)
}
func (c *capturingSessionRepo) TraceVectorSearch(ctx context.Context, sessionID, embeddingModel string, queryVec []float32, limit int) ([]domain.Memory, error) {
	return c.stub.TraceVectorSearch(ctx, sessionID, embeddingModel, queryVec, limit)
}

func (c *capturingSessionRepo) ListBySessionIDs(ctx context.Context, ids []string, limit int) ([]*domain.Session, error) {
	return c.stub.ListBySessionIDs(ctx, ids, limit)
}

func (c *capturingSessionRepo) ListRecentBySessionID(ctx context.Context, sessionID string, limit int) ([]*domain.Session, error) {
	return c.stub.ListRecentBySessionID(ctx, sessionID, limit)
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		af := float64(a[i])
		bf := float64(b[i])
		dot += af * bf
		normA += af * af
		normB += bf * bf
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
