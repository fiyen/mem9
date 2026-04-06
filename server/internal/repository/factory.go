package repository

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/qiffang/mnemos/server/internal/domain"
	"github.com/qiffang/mnemos/server/internal/repository/db9"
	"github.com/qiffang/mnemos/server/internal/repository/postgres"
	"github.com/qiffang/mnemos/server/internal/repository/tidb"
)

// NewDB creates a database connection pool for the specified backend.
func NewDB(backend, dsn string) (*sql.DB, error) {
	switch backend {
	case "db9":
		return db9.NewDB(dsn)
	case "postgres":
		return postgres.NewDB(dsn)
	case "tidb":
		return tidb.NewDB(dsn)
	default:
		return nil, fmt.Errorf("unsupported DB backend: %s", backend)
	}
}

// NewTenantRepo creates a TenantRepo for the specified backend.
func NewTenantRepo(backend string, db *sql.DB) TenantRepo {
	switch backend {
	case "db9":
		return db9.NewTenantRepo(db)
	case "postgres":
		return postgres.NewTenantRepo(db)
	default:
		return tidb.NewTenantRepo(db)
	}
}

// NewUploadTaskRepo creates an UploadTaskRepo for the specified backend.
func NewUploadTaskRepo(backend string, db *sql.DB) UploadTaskRepo {
	switch backend {
	case "db9":
		return db9.NewUploadTaskRepo(db)
	case "postgres":
		return postgres.NewUploadTaskRepo(db)
	default:
		return tidb.NewUploadTaskRepo(db)
	}
}

// NewMemoryRepo creates a MemoryRepo for the specified backend.
// autoModel is used by tidb and db9 backends for auto-embedding features.
func NewMemoryRepo(backend string, db *sql.DB, autoModel string, ftsEnabled bool, clusterID string) MemoryRepo {
	switch backend {
	case "db9":
		return db9.NewMemoryRepo(db, autoModel, ftsEnabled, clusterID)
	case "postgres":
		return postgres.NewMemoryRepo(db, ftsEnabled, clusterID)
	default:
		return tidb.NewMemoryRepo(db, autoModel, ftsEnabled, clusterID)
	}
}

// NewSessionRepo creates a SessionRepo for the specified backend.
func NewSessionRepo(backend string, db *sql.DB, autoModel string, ftsEnabled bool, clusterID string) SessionRepo {
	switch backend {
	case "postgres":
		return postgres.NewSessionRepo(db, ftsEnabled, clusterID)
	case "tidb", "":
		return tidb.NewSessionRepo(db, autoModel, ftsEnabled, clusterID)
	default:
		return stubSessionRepo{}
	}
}

// stubSessionRepo satisfies SessionRepo for backends without session support.
type stubSessionRepo struct{}

func (stubSessionRepo) BulkCreate(_ context.Context, _ []*domain.Session) error { return nil }
func (stubSessionRepo) GetByID(_ context.Context, _ string) (*domain.Session, error) {
	return nil, fmt.Errorf("session message: %w", domain.ErrNotSupported)
}
func (stubSessionRepo) ListBySessionID(_ context.Context, _ string) ([]*domain.Session, error) {
	return nil, fmt.Errorf("session messages: %w", domain.ErrNotSupported)
}
func (stubSessionRepo) PatchTags(_ context.Context, _, _ string, _ []string) error {
	return nil
}
func (stubSessionRepo) AutoVectorSearch(_ context.Context, _ string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return nil, nil
}
func (stubSessionRepo) VectorSearch(_ context.Context, _ []float32, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return nil, nil
}
func (stubSessionRepo) FTSSearch(_ context.Context, _ string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return nil, nil
}
func (stubSessionRepo) KeywordSearch(_ context.Context, _ string, _ domain.MemoryFilter, _ int) ([]domain.Memory, error) {
	return nil, nil
}
func (stubSessionRepo) FTSAvailable() bool { return false }
func (stubSessionRepo) ListTraceEmbeddingsBySessionID(_ context.Context, _ string) ([]*domain.SessionTraceEmbedding, error) {
	return nil, fmt.Errorf("session trace embeddings: %w", domain.ErrNotSupported)
}
func (stubSessionRepo) UpsertTraceEmbeddings(_ context.Context, _ []*domain.SessionTraceEmbedding) error {
	return fmt.Errorf("session trace embeddings: %w", domain.ErrNotSupported)
}
func (stubSessionRepo) TraceVectorSearch(_ context.Context, _, _ string, _ []float32, _ int) ([]domain.Memory, error) {
	return nil, fmt.Errorf("session trace vector search: %w", domain.ErrNotSupported)
}
func (stubSessionRepo) ListBySessionIDs(_ context.Context, _ []string, _ int) ([]*domain.Session, error) {
	return nil, fmt.Errorf("session messages: %w", domain.ErrNotSupported)
}
