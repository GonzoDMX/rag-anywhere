package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/GonzoDMX/rag-anywhere/internal/config"
)

// GetDBConfig reads the configuration table from a specific database
func (m *Manager) GetDBConfig(dbName string) (config.DBState, error) {
	var state config.DBState

	dbPath := filepath.Join(m.GetDBPath(dbName), "rag.db")
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return state, fmt.Errorf("database file not found")
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return state, err
	}
	defer db.Close()

	// Read all config keys into a map for easy access
	rows, err := db.Query("SELECT key, value FROM config")
	if err != nil {
		return state, err
	}
	defer rows.Close()

	kv := make(map[string]string)
	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err == nil {
			kv[k] = v
		}
	}

	// Map to DBState struct
	state.EmbedID = kv["embed_model_id"]
	state.EmbedVersion = kv["embed_model_version"]
	state.NERID = kv["ner_model_id"]
	state.NERVersion = kv["ner_model_version"]

	if dimStr, ok := kv["embed_dimension"]; ok {
		if dim, err := strconv.Atoi(dimStr); err == nil {
			state.EmbedDim = dim
		}
	}

	return state, nil
}
