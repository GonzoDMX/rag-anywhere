package config

// ModelType distinguishes between critical (Embed) and non-critical (NER) models
type ModelType string

const (
	TypeEmbedding ModelType = "embedding"
	TypeNER       ModelType = "ner"
)

// ModelCard defines the exact specifications of a model
type ModelCard struct {
	ID            string    `json:"id"`      // e.g. "google/embedding-gemma"
	Version       string    `json:"version"` // e.g. "v1.0" or specific commit hash
	Type          ModelType `json:"type"`
	Dimension     int       `json:"dimension"`      // Critical for vector stores (e.g. 768)
	ContextLength int       `json:"context_length"` // Max input tokens for model
}

// ProcessingConfig defines how we ingest text
type ProcessingConfig struct {
	ChunkSize    int `json:"chunk_size"`    // e.g. 512
	ChunkOverlap int `json:"chunk_overlap"` // e.g. 50
}

// SystemConfig represents the "Gold Standard" for this version of the App
type SystemConfig struct {
	AppVersion     string
	EmbeddingModel ModelCard
	NERModel       ModelCard
	Processing     ProcessingConfig
}

// CurrentDefaults defines the configuration for THIS version of the binary.
// When you update the app, you change these values here.
var CurrentDefaults = SystemConfig{
	AppVersion: "0.1.0",

	EmbeddingModel: ModelCard{
		ID:            "google/embedding-gemma",
		Version:       "1.0", // Increment this to force migration
		Type:          TypeEmbedding,
		Dimension:     768,
		ContextLength: 2048,
	},

	NERModel: ModelCard{
		ID:            "urchade/gliner_multi",
		Version:       "2.1",
		Type:          TypeNER,
		Dimension:     0,
		ContextLength: 512,
	},

	Processing: ProcessingConfig{
		ChunkSize:    2048,
		ChunkOverlap: 50,
	},
}
