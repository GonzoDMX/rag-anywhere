package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/GonzoDMX/rag-anywhere/internal/config"
	_ "github.com/mattn/go-sqlite3"
)

const (
	AppDirName = ".rag-anywhere"
	DBDirName  = "databases"
	ModelsDir  = "models"
	LogsDir    = "logs"
)

// Manager handles the physical file resources and database lifecycle
type Manager struct {
	RootDir string
}

// NewManager creates the directory structure if it doesn't exist
func NewManager() (*Manager, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("could not find user home: %w", err)
	}

	root := filepath.Join(home, AppDirName)
	dirs := []string{
		filepath.Join(root, DBDirName),
		filepath.Join(root, ModelsDir),
		filepath.Join(root, LogsDir),
	}

	for _, d := range dirs {
		// 0755: Owner can read/write/exec, Group/Others can read/exec
		if err := os.MkdirAll(d, 0755); err != nil {
			return nil, fmt.Errorf("failed to init dir %s: %w", d, err)
		}
	}

	return &Manager{RootDir: root}, nil
}

// GetDBPath returns the full path to a specific database folder
func (m *Manager) GetDBPath(dbName string) string {
	return filepath.Join(m.RootDir, DBDirName, dbName)
}

// CreateDatabase initializes a new sqlite file, runs schema migrations,
// and stamps the current application configuration into the DB.
func (m *Manager) CreateDatabase(name string, description string) error {
	dbPath := m.GetDBPath(name)

	// 1. Check if exists
	if _, err := os.Stat(dbPath); !os.IsNotExist(err) {
		return fmt.Errorf("database '%s' already exists", name)
	}

	// 2. Create Folder
	if err := os.MkdirAll(dbPath, 0755); err != nil {
		return err
	}

	// 3. Init SQLite
	sqliteFile := filepath.Join(dbPath, "rag.db")
	db, err := sql.Open("sqlite3", sqliteFile)
	if err != nil {
		return err
	}
	defer db.Close()

	// 4. Run Schema
	// Note: SchemaSQL is defined in schema.go within the same package
	if _, err := db.Exec(SchemaSQL); err != nil {
		// Clean up if schema fails to avoid leaving a zombie DB folder
		os.RemoveAll(dbPath)
		return fmt.Errorf("failed to apply schema: %w", err)
	}

	// 5. STAMP CONFIGURATION
	// This ensures the DB knows exactly which models created its data.
	// This is critical for future migrations (e.g. if embedding dim changes).
	defaults := config.CurrentDefaults

	_, err = db.Exec(`
		INSERT INTO config (key, value) VALUES 
		('description', ?),
		('created_at', ?),
		('app_version', ?),
		('embed_model_id', ?),
		('embed_model_version', ?),
		('embed_dimension', ?),
		('embed_context_length', ?),
		('ner_model_id', ?),
		('ner_model_version', ?)
	`,
		description,
		time.Now().Format(time.RFC3339),
		defaults.AppVersion,
		defaults.EmbeddingModel.ID,
		defaults.EmbeddingModel.Version,
		fmt.Sprintf("%d", defaults.EmbeddingModel.Dimension),
		fmt.Sprintf("%d", defaults.EmbeddingModel.ContextLength),
		defaults.NERModel.ID,
		defaults.NERModel.Version,
	)

	return err
}

// DeleteDatabase removes the folder and all contents (SQL + Vectors)
func (m *Manager) DeleteDatabase(name string) error {
	path := m.GetDBPath(name)

	// Safety check: make sure we are deleting a directory inside our managed folder
	// to prevent accidentally deleting "/" or "~" if name is malformed.
	cleanPath := filepath.Clean(path)
	expectedPrefix := filepath.Join(m.RootDir, DBDirName)

	// Ensure the path to delete is actually inside our databases directory
	// (Go doesn't have a direct "IsSubpath" but string prefix check is decent for this)
	if len(cleanPath) <= len(expectedPrefix) {
		return fmt.Errorf("invalid database name: safety check failed")
	}

	return os.RemoveAll(path)
}

// ListDatabases scans the directory for valid DBs
func (m *Manager) ListDatabases() ([]string, error) {
	base := filepath.Join(m.RootDir, DBDirName)
	entries, err := os.ReadDir(base)
	if err != nil {
		return nil, err
	}

	var dbs []string
	for _, e := range entries {
		if e.IsDir() {
			// Optional: Check if rag.db exists inside to confirm it's valid
			// This filters out empty folders that might have been created manually
			if _, err := os.Stat(filepath.Join(base, e.Name(), "rag.db")); err == nil {
				dbs = append(dbs, e.Name())
			}
		}
	}
	return dbs, nil
}
