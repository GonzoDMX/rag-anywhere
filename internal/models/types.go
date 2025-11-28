package models

// --- 1. EMBEDDER WORKER ---
type EmbedRequest struct {
	Text string `json:"text"`
}
type EmbedResponse struct {
	Vector []float32 `json:"vector"`
	Error  string    `json:"error,omitempty"`
}

// --- 2. NER WORKER ---
type NERRequest struct {
	Text     string   `json:"text"`
	Keywords []string `json:"keywords,omitempty"` // Optional keywords for NER
}
type Entity struct {
	Text  string `json:"text"`
	Label string `json:"label"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

type NERResponse struct {
	Entities []Entity `json:"entities"` // Your entity struct
	Error    string   `json:"error,omitempty"`
}

// --- 3. VECTOR DB WORKER (FAISS) ---
// This worker is special. It needs commands to manage the index.
type VectorDBRequest struct {
	Command  string      `json:"command"`             // "load", "add", "search", "save"
	DbPath   string      `json:"db_path,omitempty"`   // For loading/saving specific indexes
	Vectors  [][]float32 `json:"vectors,omitempty"`   // Batch adding
	Ids      []int64     `json:"ids,omitempty"`       // IDs matching vectors
	QueryVec []float32   `json:"query_vec,omitempty"` // For searching
	TopK     int         `json:"top_k,omitempty"`
}

type VectorDBResponse struct {
	Status  string    `json:"status"`            // "ok", "loaded", "saved"
	Results []int64   `json:"results,omitempty"` // IDs found
	Scores  []float32 `json:"scores,omitempty"`  // Similarity scores
	Count   int       `json:"count,omitempty"`   // Current index size
	Error   string    `json:"error,omitempty"`
}
