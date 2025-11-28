package models

// ==========================================
// 1. EMBEDDING WORKER
// ==========================================

type WorkerEmbedRequest struct {
	// We use a slice to allow batching (e.g. processing 10 chunks at once)
	Texts []string `json:"texts"`

	// Optional: For document embedding (Gemma format: "title: {Title} | text: {Text}")
	Title string `json:"title,omitempty"`

	// Optional: For query embedding (Gemma format: "task: {Task} | query: {Text}")
	// Valid values: "retrieval", "question_answering", "fact_checking", "code_retrieval"
	TaskType string `json:"task_type,omitempty"`
}

type WorkerEmbedResponse struct {
	// Returns a list of vectors. Each vector is a list of floats.
	Vectors [][]float32 `json:"vectors"`
	Error   string      `json:"error,omitempty"`
}

// ==========================================
// 2. NER WORKER (GLiNER)
// ==========================================

type WorkerNERRequest struct {
	Text   string   `json:"text"`
	Labels []string `json:"labels,omitempty"` // Optional override
}

type WorkerNEREntity struct {
	Text  string `json:"text"`
	Label string `json:"label"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

type WorkerNERResponse struct {
	Entities []WorkerNEREntity `json:"entities"`
	Error    string            `json:"error,omitempty"`
}

// ==========================================
// 3. VECTOR DB WORKER (FAISS)
// ==========================================

type WorkerVectorCmd struct {
	Command   string      `json:"command"`             // "init", "load", "save", "add", "remove", "search"
	Path      string      `json:"path,omitempty"`      // For load/save
	Dimension int         `json:"dimension,omitempty"` // For init
	Vectors   [][]float32 `json:"vectors,omitempty"`   // For add
	Ids       []int64     `json:"ids,omitempty"`       // For add/remove
	Vector    []float32   `json:"vector,omitempty"`    // For search (singular query)
	TopK      int         `json:"top_k,omitempty"`     // For search
}

type WorkerVectorResponse struct {
	Status  string    `json:"status"`
	Count   int       `json:"count,omitempty"`
	Results []int64   `json:"results,omitempty"` // IDs found
	Scores  []float32 `json:"scores,omitempty"`  // Similarity scores
	Error   string    `json:"error,omitempty"`
}
