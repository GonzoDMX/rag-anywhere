package models

// ==========================================
// 1. EMBEDDING WORKER
// ==========================================

type WorkerEmbedRequest struct {
	// Command: "embed" (default) or "count_tokens"
	Command string `json:"command,omitempty"`

	// Required: We use a slice to allow batching (e.g. processing 10 chunks at once)
	Texts []string `json:"texts"`

	// Required for 'embed': Task Optimized embeddings (Gemma format: "task: {Task} | query: {Text}")
	// Valid values: "retrieval_document",  -- For documents to be stored in vector DB
	// 				 "retrieval_query",		-- For user queries
	// 				 "question_answering",  -- Retrieve anwers for singular questions
	// 				 "fact_verification",   -- Verify facts against the vector DB corpus
	// 				 "classification",	    -- For text classification tasks
	// 				 "clustering",		    -- For clustering tasks
	// 				 "semantic_similarity", -- For sentence similarity tasks
	// 				 "code_retrieval"       -- Improved code retrieval for natural language prompts
	TaskType string `json:"task_type,omitempty"`

	// Optional: For document embedding (Gemma format: "title: {Title} | text: {Text}")
	Title string `json:"title,omitempty"`
}

type WorkerEmbedResponse struct {
	// Used for "embed" command
	Vectors [][]float32 `json:"vectors,omitempty"`

	// Used for "count_tokens" command
	TokenCounts []int `json:"token_counts,omitempty"`

	Error string `json:"error,omitempty"`
}

// ==========================================
// 2. NER WORKER (GLiNER)
// ==========================================

type WorkerNERRequest struct {
	// Batch of sub-chunks (strings) to process in parallel
	Texts []string `json:"texts"`
	// Full list of labels. The Python worker will split these into
	// smaller batches (e.g. groups of 10) to maintain accuracy.
	Labels []string `json:"labels"`
}

type WorkerNEREntity struct {
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"` // Relative to the specific sub-chunk
	End   int     `json:"end"`
	Score float32 `json:"score"`
}

type WorkerNERResponse struct {
	// Results is a list of lists.
	// Results[i] contains the entities found in Texts[i]
	Results [][]WorkerNEREntity `json:"results"`
	Error   string              `json:"error,omitempty"`
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
