package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pgvector/pgvector-go"

	"github.com/qiffang/mnemos/server/internal/domain"
)

type SessionRepo struct {
	db           *sql.DB
	ftsAvailable atomic.Bool
	clusterID    string
}

func NewSessionRepo(db *sql.DB, ftsEnabled bool, clusterID string) *SessionRepo {
	r := &SessionRepo{db: db, clusterID: clusterID}
	r.ftsAvailable.Store(ftsEnabled)
	return r
}

func (r *SessionRepo) FTSAvailable() bool { return r.ftsAvailable.Load() }

const sessionColumns = `id, session_id, agent_id, source, seq, role, content, content_type, content_hash, tags, state, created_at, updated_at`

func (r *SessionRepo) GetByID(ctx context.Context, id string) (*domain.Session, error) {
	row := r.db.QueryRowContext(ctx,
		`SELECT `+sessionColumns+` FROM sessions WHERE id = $1 AND state = 'active'`,
		id,
	)
	return scanSessionDomainRow(row)
}

func (r *SessionRepo) BulkCreate(ctx context.Context, sessions []*domain.Session) error {
	if len(sessions) == 0 {
		return nil
	}

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sessions bulk create begin tx: %w", err)
	}
	defer tx.Rollback()

	for _, s := range sessions {
		_, execErr := tx.ExecContext(ctx,
			`INSERT INTO sessions
				(id, session_id, agent_id, source, seq, role, content, content_type, content_hash, tags, embedding, state, created_at, updated_at)
			 VALUES
				($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'active', NOW(), NOW())
			 ON CONFLICT (session_id, content_hash) DO NOTHING`,
			s.ID,
			nullString(s.SessionID),
			nullString(s.AgentID),
			nullString(s.Source),
			s.Seq,
			s.Role,
			s.Content,
			s.ContentType,
			s.ContentHash,
			marshalTags(s.Tags),
			vecToParam(s.Embedding),
		)
		if execErr != nil {
			return fmt.Errorf("sessions bulk insert %s: %w", s.ID, execErr)
		}
	}

	return tx.Commit()
}

func (r *SessionRepo) PatchTags(ctx context.Context, sessionID, contentHash string, tags []string) error {
	_, err := r.db.ExecContext(ctx,
		`UPDATE sessions
		 SET tags = $1
		 WHERE session_id = $2
		   AND content_hash = $3
		   AND jsonb_array_length(COALESCE(tags, '[]'::jsonb)) = 0`,
		marshalTags(tags),
		sessionID,
		contentHash,
	)
	if err != nil {
		return fmt.Errorf("patch session tags: %w", err)
	}
	return nil
}

func (r *SessionRepo) AutoVectorSearch(ctx context.Context, query string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	if r.FTSAvailable() {
		return r.FTSSearch(ctx, query, f, limit)
	}
	return r.KeywordSearch(ctx, query, f, limit)
}

func (r *SessionRepo) VectorSearch(ctx context.Context, queryVec []float32, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	if len(queryVec) == 0 {
		return nil, nil
	}

	conds, args := buildSessionFilterConds(f)
	conds = append(conds, "embedding IS NOT NULL")

	vecParamIdx := len(args) + 1
	limitParamIdx := vecParamIdx + 1
	where := strings.Join(conds, " AND ")

	query := fmt.Sprintf(`SELECT id, session_id, agent_id, source, seq, role, content, content_type, tags, state, created_at,
		 embedding <=> $%d AS distance
		 FROM sessions
		 WHERE %s
		 ORDER BY embedding <=> $%d
		 LIMIT $%d`, vecParamIdx, where, vecParamIdx, limitParamIdx)

	fullArgs := make([]any, 0, len(args)+2)
	fullArgs = append(fullArgs, args...)
	fullArgs = append(fullArgs, pgvector.NewVector(queryVec), limit)

	rows, err := r.db.QueryContext(ctx, query, fullArgs...)
	if err != nil {
		return nil, fmt.Errorf("sessions vector search: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionRowsWithDistance(rows)
}

func (r *SessionRepo) FTSSearch(ctx context.Context, query string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	conds, args := buildSessionFilterConds(f)
	where := strings.Join(conds, " AND ")

	queryParamIdx := len(args) + 1
	limitParamIdx := queryParamIdx + 1
	sqlQuery := fmt.Sprintf(`SELECT id, session_id, agent_id, source, seq, role, content, content_type, tags, state, created_at,
		 ts_rank(to_tsvector('english', content), plainto_tsquery('english', $%d)) AS fts_score
		 FROM sessions
		 WHERE %s AND to_tsvector('english', content) @@ plainto_tsquery('english', $%d)
		 ORDER BY fts_score DESC, created_at ASC, seq ASC, id ASC
		 LIMIT $%d`, queryParamIdx, where, queryParamIdx, limitParamIdx)

	fullArgs := make([]any, 0, len(args)+2)
	fullArgs = append(fullArgs, args...)
	fullArgs = append(fullArgs, query, limit)

	rows, err := r.db.QueryContext(ctx, sqlQuery, fullArgs...)
	if err != nil {
		return nil, fmt.Errorf("sessions fts search: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionRowsWithFTSScore(rows)
}

func (r *SessionRepo) KeywordSearch(ctx context.Context, query string, f domain.MemoryFilter, limit int) ([]domain.Memory, error) {
	conds, args := buildSessionFilterConds(f)
	if query != "" {
		nextParam := len(args) + 1
		conds = append(conds, fmt.Sprintf("content ILIKE '%%' || $%d || '%%'", nextParam))
		args = append(args, query)
	}

	where := strings.Join(conds, " AND ")
	limitParamIdx := len(args) + 1
	sqlQuery := fmt.Sprintf(`SELECT id, session_id, agent_id, source, seq, role, content, content_type, tags, state, created_at
		 FROM sessions
		 WHERE %s
		 ORDER BY created_at ASC, seq ASC, id ASC
		 LIMIT $%d`, where, limitParamIdx)
	args = append(args, limit)

	rows, err := r.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("sessions keyword search: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionRows(rows)
}

func (r *SessionRepo) ListBySessionIDs(ctx context.Context, sessionIDs []string, limitPerSession int) ([]*domain.Session, error) {
	if len(sessionIDs) == 0 {
		return nil, nil
	}

	placeholders := make([]string, len(sessionIDs))
	args := make([]any, 0, len(sessionIDs)+1)
	for i, id := range sessionIDs {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args = append(args, id)
	}
	limitParamIdx := len(args) + 1
	args = append(args, limitPerSession)

	sqlQuery := `SELECT id, session_id, agent_id, source, seq, role, content, content_type,
		content_hash, tags, state, created_at, updated_at
		FROM (
			SELECT *,
				ROW_NUMBER() OVER (
					PARTITION BY session_id
					ORDER BY created_at ASC, seq ASC, id ASC
				) AS rn
			FROM sessions
			WHERE session_id IN (` + strings.Join(placeholders, ",") + `) AND state = 'active'
		) t
		WHERE rn <= $` + fmt.Sprintf("%d", limitParamIdx) + `
		ORDER BY session_id ASC, created_at ASC, seq ASC, id ASC`

	rows, err := r.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("sessions list by session ids: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionDomainRows(rows)
}

func (r *SessionRepo) ListRecentBySessionID(ctx context.Context, sessionID string, limit int) ([]*domain.Session, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT `+sessionColumns+`
		FROM sessions
		WHERE session_id = $1 AND state = 'active'
		ORDER BY created_at DESC, seq DESC, id DESC
		LIMIT $2`,
		sessionID,
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("sessions list recent by session id: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionDomainRows(rows)
}

func buildSessionFilterConds(f domain.MemoryFilter) ([]string, []any) {
	conds := []string{}
	args := []any{}
	paramIdx := 1

	if f.State == "all" {
	} else if f.State != "" {
		conds = append(conds, fmt.Sprintf("state = $%d", paramIdx))
		args = append(args, f.State)
		paramIdx++
	} else {
		conds = append(conds, "state = 'active'")
	}

	if f.AgentID != "" {
		conds = append(conds, fmt.Sprintf("agent_id = $%d", paramIdx))
		args = append(args, f.AgentID)
		paramIdx++
	}
	if f.SessionID != "" {
		conds = append(conds, fmt.Sprintf("session_id = $%d", paramIdx))
		args = append(args, f.SessionID)
		paramIdx++
	}
	if f.Source != "" {
		conds = append(conds, fmt.Sprintf("source = $%d", paramIdx))
		args = append(args, f.Source)
		paramIdx++
	}
	for _, tag := range f.Tags {
		tagJSON, err := json.Marshal(tag)
		if err != nil {
			continue
		}
		conds = append(conds, fmt.Sprintf("tags @> $%d::jsonb", paramIdx))
		args = append(args, "["+string(tagJSON)+"]")
		paramIdx++
	}
	if len(conds) == 0 {
		conds = append(conds, "1=1")
	}
	return conds, args
}

func scanSessionRows(rows *sql.Rows) ([]domain.Memory, error) {
	var result []domain.Memory
	for rows.Next() {
		m, err := scanSessionRowNoScore(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, *m)
	}
	return result, rows.Err()
}

func scanSessionRowsWithDistance(rows *sql.Rows) ([]domain.Memory, error) {
	var result []domain.Memory
	for rows.Next() {
		m, err := scanSessionRowWithDistance(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, *m)
	}
	return result, rows.Err()
}

func scanSessionRowsWithFTSScore(rows *sql.Rows) ([]domain.Memory, error) {
	var result []domain.Memory
	for rows.Next() {
		m, err := scanSessionRowWithFTSScore(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, *m)
	}
	return result, rows.Err()
}

func scanSessionRowNoScore(rows *sql.Rows) (*domain.Memory, error) {
	var (
		sessionID, agentID, source, role, contentType sql.NullString
		tagsJSON                                      []byte
		state                                         sql.NullString
		seq                                           int
		createdAt                                     time.Time
		m                                             domain.Memory
	)

	if err := rows.Scan(
		&m.ID, &sessionID, &agentID, &source,
		&seq, &role, &m.Content, &contentType,
		&tagsJSON, &state, &createdAt,
	); err != nil {
		return nil, fmt.Errorf("scan session row: %w", err)
	}

	return fillSessionMemory(&m, sessionID, agentID, source, role, contentType, seq, tagsJSON, state, createdAt), nil
}

func scanSessionRowWithDistance(rows *sql.Rows) (*domain.Memory, error) {
	var (
		sessionID, agentID, source, role, contentType sql.NullString
		tagsJSON                                      []byte
		state                                         sql.NullString
		seq                                           int
		createdAt                                     time.Time
		distance                                      float64
		m                                             domain.Memory
	)

	if err := rows.Scan(
		&m.ID, &sessionID, &agentID, &source,
		&seq, &role, &m.Content, &contentType,
		&tagsJSON, &state, &createdAt,
		&distance,
	); err != nil {
		return nil, fmt.Errorf("scan session row with distance: %w", err)
	}

	m = *fillSessionMemory(&m, sessionID, agentID, source, role, contentType, seq, tagsJSON, state, createdAt)
	score := 1 - distance
	m.Score = &score
	return &m, nil
}

func scanSessionRowWithFTSScore(rows *sql.Rows) (*domain.Memory, error) {
	var (
		sessionID, agentID, source, role, contentType sql.NullString
		tagsJSON                                      []byte
		state                                         sql.NullString
		seq                                           int
		createdAt                                     time.Time
		ftsScore                                      float64
		m                                             domain.Memory
	)

	if err := rows.Scan(
		&m.ID, &sessionID, &agentID, &source,
		&seq, &role, &m.Content, &contentType,
		&tagsJSON, &state, &createdAt,
		&ftsScore,
	); err != nil {
		return nil, fmt.Errorf("scan session row with fts score: %w", err)
	}

	m = *fillSessionMemory(&m, sessionID, agentID, source, role, contentType, seq, tagsJSON, state, createdAt)
	m.Score = &ftsScore
	return &m, nil
}

func scanSessionDomainRows(rows *sql.Rows) ([]*domain.Session, error) {
	var result []*domain.Session
	for rows.Next() {
		s, err := scanSessionDomainRowsItem(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, s)
	}
	return result, rows.Err()
}

func scanSessionDomainRow(row *sql.Row) (*domain.Session, error) {
	var (
		sessionID, agentID, source, role, contentType, contentHash sql.NullString
		tagsJSON                                                   []byte
		state                                                      sql.NullString
		s                                                          domain.Session
	)

	if err := row.Scan(
		&s.ID, &sessionID, &agentID, &source,
		&s.Seq, &role, &s.Content, &contentType,
		&contentHash, &tagsJSON, &state,
		&s.CreatedAt, &s.UpdatedAt,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, domain.ErrNotFound
		}
		return nil, fmt.Errorf("scan session domain row: %w", err)
	}

	return fillSessionDomain(&s, sessionID, agentID, source, role, contentType, contentHash, tagsJSON, state), nil
}

func scanSessionDomainRowsItem(rows *sql.Rows) (*domain.Session, error) {
	var (
		sessionID, agentID, source, role, contentType, contentHash sql.NullString
		tagsJSON                                                   []byte
		state                                                      sql.NullString
		s                                                          domain.Session
	)

	if err := rows.Scan(
		&s.ID, &sessionID, &agentID, &source,
		&s.Seq, &role, &s.Content, &contentType,
		&contentHash, &tagsJSON, &state,
		&s.CreatedAt, &s.UpdatedAt,
	); err != nil {
		return nil, fmt.Errorf("scan session domain row: %w", err)
	}

	return fillSessionDomain(&s, sessionID, agentID, source, role, contentType, contentHash, tagsJSON, state), nil
}

func fillSessionDomain(
	s *domain.Session,
	sessionID, agentID, source, role, contentType, contentHash sql.NullString,
	tagsJSON []byte,
	state sql.NullString,
) *domain.Session {
	s.SessionID = sessionID.String
	s.AgentID = agentID.String
	s.Source = source.String
	s.Role = role.String
	s.ContentType = contentType.String
	s.ContentHash = contentHash.String
	s.Tags = unmarshalTags(tagsJSON)
	s.State = domain.MemoryState(state.String)
	if s.State == "" {
		s.State = domain.StateActive
	}
	return s
}

func fillSessionMemory(
	m *domain.Memory,
	sessionID, agentID, source, role, contentType sql.NullString,
	seq int,
	tagsJSON []byte,
	state sql.NullString,
	createdAt time.Time,
) *domain.Memory {
	m.MemoryType = domain.TypeSession
	m.SessionID = sessionID.String
	m.AgentID = agentID.String
	m.Source = source.String
	m.State = domain.MemoryState(state.String)
	if m.State == "" {
		m.State = domain.StateActive
	}
	m.Tags = unmarshalTags(tagsJSON)
	m.CreatedAt = createdAt
	m.UpdatedAt = createdAt
	metaBytes, _ := json.Marshal(map[string]any{
		"role":         role.String,
		"seq":          seq,
		"content_type": contentType.String,
	})
	m.Metadata = metaBytes
	return m
}
