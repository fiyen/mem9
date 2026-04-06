package tidb

import (
	"context"
	"fmt"

	"github.com/qiffang/mnemos/server/internal/domain"
	internaltenant "github.com/qiffang/mnemos/server/internal/tenant"
)

func (r *SessionRepo) ListBySessionID(ctx context.Context, sessionID string) ([]*domain.Session, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT id, session_id, agent_id, source, seq, role, content, content_type,
		content_hash, tags, state, created_at, updated_at
		FROM sessions
		WHERE session_id = ? AND state = 'active'
		ORDER BY created_at ASC, seq ASC, id ASC`,
		sessionID,
	)
	if err != nil {
		if internaltenant.IsTableNotFoundError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("sessions list by session id: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionDomainRows(rows)
}

func (r *SessionRepo) ListTraceEmbeddingsBySessionID(ctx context.Context, sessionID string) ([]*domain.SessionTraceEmbedding, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT node_id, session_id, content_hash, role, embedding_model, created_at, updated_at
		FROM session_trace_embeddings
		WHERE session_id = ?`,
		sessionID,
	)
	if err != nil {
		if internaltenant.IsTableNotFoundError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("session trace embeddings list: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	var entries []*domain.SessionTraceEmbedding
	for rows.Next() {
		entry := &domain.SessionTraceEmbedding{}
		if err := rows.Scan(
			&entry.NodeID,
			&entry.SessionID,
			&entry.ContentHash,
			&entry.Role,
			&entry.EmbeddingModel,
			&entry.CreatedAt,
			&entry.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan session trace embedding: %w", err)
		}
		entries = append(entries, entry)
	}
	return entries, rows.Err()
}

func (r *SessionRepo) UpsertTraceEmbeddings(ctx context.Context, entries []*domain.SessionTraceEmbedding) error {
	if len(entries) == 0 {
		return nil
	}

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("session trace embeddings begin tx: %w", err)
	}
	defer tx.Rollback()

	for _, entry := range entries {
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO session_trace_embeddings
				(node_id, session_id, content_hash, role, embedding_model, embedding, created_at, updated_at)
			 VALUES
				(?, ?, ?, ?, ?, ?, NOW(), NOW())
			 ON DUPLICATE KEY UPDATE
			     session_id = VALUES(session_id),
			     content_hash = VALUES(content_hash),
			     role = VALUES(role),
			     embedding_model = VALUES(embedding_model),
			     embedding = VALUES(embedding),
			     updated_at = NOW()`,
			entry.NodeID,
			entry.SessionID,
			entry.ContentHash,
			entry.Role,
			entry.EmbeddingModel,
			vecToString(entry.Embedding),
		); err != nil {
			if internaltenant.IsTableNotFoundError(err) {
				return nil
			}
			return fmt.Errorf("upsert session trace embedding %s: %w", entry.NodeID, err)
		}
	}

	return tx.Commit()
}

func (r *SessionRepo) TraceVectorSearch(ctx context.Context, sessionID, embeddingModel string, queryVec []float32, limit int) ([]domain.Memory, error) {
	vecStr := vecToString(queryVec)
	if vecStr == nil {
		return nil, nil
	}

	rows, err := r.db.QueryContext(ctx,
		`SELECT s.id, s.session_id, s.agent_id, s.source, s.seq, s.role, s.content, s.content_type, s.tags, s.state, s.created_at,
			VEC_COSINE_DISTANCE(ste.embedding, ?) AS distance
		FROM session_trace_embeddings ste
		JOIN sessions s ON s.id = ste.node_id
		WHERE ste.session_id = ?
		  AND ste.embedding_model = ?
		  AND ste.embedding IS NOT NULL
		  AND s.state = 'active'
		ORDER BY VEC_COSINE_DISTANCE(ste.embedding, ?), s.created_at ASC, s.seq ASC, s.id ASC
		LIMIT ?`,
		vecStr,
		sessionID,
		embeddingModel,
		vecStr,
		limit,
	)
	if err != nil {
		if internaltenant.IsTableNotFoundError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("session trace vector search: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionRowsWithDistance(rows)
}
