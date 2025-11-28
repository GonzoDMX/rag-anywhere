package config

import (
	"fmt"
)

type MigrationStatus string

const (
	StatusCompatible      MigrationStatus = "compatible"
	StatusUpdateAvailable MigrationStatus = "update_available" // Optional (NER)
	StatusIncompatible    MigrationStatus = "incompatible"     // Mandatory (Embedding)
)

// DBState represents the config we read from the SQLite table
type DBState struct {
	EmbedID      string
	EmbedVersion string
	EmbedDim     int
	NERID        string
	NERVersion   string
}

// CheckCompatibility compares the DB's state against the App's defaults
func CheckCompatibility(db DBState) (MigrationStatus, []string) {
	var issues []string
	status := StatusCompatible

	// 1. CRITICAL CHECK: Embedding Model
	// If the model name, version, or dimension differs, the vectors are garbage.
	if db.EmbedID != CurrentDefaults.EmbeddingModel.ID ||
		db.EmbedVersion != CurrentDefaults.EmbeddingModel.Version ||
		db.EmbedDim != CurrentDefaults.EmbeddingModel.Dimension {

		status = StatusIncompatible
		issues = append(issues, fmt.Sprintf(
			"Embedding Model Mismatch: DB has %s (%s, dim:%d), App requires %s (%s, dim:%d)",
			db.EmbedID, db.EmbedVersion, db.EmbedDim,
			CurrentDefaults.EmbeddingModel.ID, CurrentDefaults.EmbeddingModel.Version, CurrentDefaults.EmbeddingModel.Dimension,
		))
	}

	// 2. NON-CRITICAL CHECK: NER Model
	// If GLiNER changes, the existing graph is still valid, just "old".
	// We only upgrade status if it's not already incompatible.
	if db.NERID != CurrentDefaults.NERModel.ID ||
		db.NERVersion != CurrentDefaults.NERModel.Version {

		if status == StatusCompatible {
			status = StatusUpdateAvailable
		}
		issues = append(issues, fmt.Sprintf(
			"NER Model Update: DB uses %s (%s), App uses %s (%s)",
			db.NERID, db.NERVersion,
			CurrentDefaults.NERModel.ID, CurrentDefaults.NERModel.Version,
		))
	}

	return status, issues
}
