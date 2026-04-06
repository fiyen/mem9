package postgres

import (
	"context"
	"fmt"

	"github.com/qiffang/mnemos/server/internal/domain"
)

func (r *SessionRepo) ListBySessionID(ctx context.Context, sessionID string) ([]*domain.Session, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT `+sessionColumns+`
		FROM sessions
		WHERE session_id = $1 AND state = 'active'
		ORDER BY created_at ASC, seq ASC, id ASC`,
		sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("sessions list by session id: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionDomainRows(rows)
}

func (r *SessionRepo) ListTraceEmbeddingsBySessionID(ctx context.Context, sessionID string) ([]*domain.SessionTraceEmbedding, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT node_id, session_id, content_hash, role, embedding_model, created_at, updated_at
		FROM session_trace_embeddings
		WHERE session_id = $1`,
		sessionID,
	)
	if err != nil {
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
				($1, $2, $3, $4, $5, $6, NOW(), NOW())
			 ON CONFLICT (node_id) DO UPDATE
			 SET session_id = EXCLUDED.session_id,
			     content_hash = EXCLUDED.content_hash,
			     role = EXCLUDED.role,
			     embedding_model = EXCLUDED.embedding_model,
			     embedding = EXCLUDED.embedding,
			     updated_at = NOW()`,
			entry.NodeID,
			entry.SessionID,
			entry.ContentHash,
			entry.Role,
			entry.EmbeddingModel,
			vecToParam(entry.Embedding),
		); err != nil {
			return fmt.Errorf("upsert session trace embedding %s: %w", entry.NodeID, err)
		}
	}

	return tx.Commit()
}

func (r *SessionRepo) TraceVectorSearch(ctx context.Context, sessionID, embeddingModel string, queryVec []float32, limit int) ([]domain.Memory, error) {
	if len(queryVec) == 0 {
		return nil, nil
	}

	rows, err := r.db.QueryContext(ctx,
		`SELECT s.id, s.session_id, s.agent_id, s.source, s.seq, s.role, s.content, s.content_type, s.tags, s.state, s.created_at,
			ste.embedding <=> $1 AS distance
		FROM session_trace_embeddings ste
		JOIN sessions s ON s.id = ste.node_id
		WHERE ste.session_id = $2
		  AND ste.embedding_model = $3
		  AND s.state = 'active'
		ORDER BY ste.embedding <=> $1, s.created_at ASC, s.seq ASC, s.id ASC
		LIMIT $4`,
		vecToParam(queryVec),
		sessionID,
		embeddingModel,
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("session trace vector search: cluster_id=%s: %w", r.clusterID, err)
	}
	defer rows.Close()

	return scanSessionRowsWithDistance(rows)
}
