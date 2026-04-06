package repository

import (
	"testing"

	repopg "github.com/qiffang/mnemos/server/internal/repository/postgres"
	repotidb "github.com/qiffang/mnemos/server/internal/repository/tidb"
)

func TestNewSessionRepo_Postgres(t *testing.T) {
	repo := NewSessionRepo("postgres", nil, "", true, "cluster-1")

	if _, ok := repo.(*repopg.SessionRepo); !ok {
		t.Fatalf("expected postgres session repo, got %T", repo)
	}
	if !repo.FTSAvailable() {
		t.Fatal("expected postgres session repo to preserve ftsEnabled")
	}
}

func TestNewSessionRepo_DefaultsToTiDB(t *testing.T) {
	repo := NewSessionRepo("", nil, "auto-model", false, "cluster-1")

	if _, ok := repo.(*repotidb.SessionRepo); !ok {
		t.Fatalf("expected tidb session repo, got %T", repo)
	}
}
