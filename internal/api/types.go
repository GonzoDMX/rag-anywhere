package api

// ==========================================
// 1. STANDARD ENVELOPE
// ==========================================

// StandardResponse wraps all API responses to ensure consistency.
// Frontend checks "success" first. If false, display "error".
type StandardResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`    // The actual payload (one of the structs below)
	Message string      `json:"message,omitempty"` // User-friendly message
	Error   string      `json:"error,omitempty"`   // User-friendly error message
	Meta    interface{} `json:"meta,omitempty"`    // Pagination, timing, request_id, etc.
}

// ==========================================
// 2. GENERAL SERVICE
// ==========================================

type StatusResponse struct {
	Status    string `json:"status"`    // "healthy", "degraded", "maintenance"
	Uptime    string `json:"uptime"`    // Human readable duration
	ActiveDB  string `json:"active_db"` // Name of the currently loaded DB
	Version   string `json:"version"`
	Port      string `json:"port"`
	GoVersion string `json:"go_version"`
}

type LogsRequest struct {
	Lines int    `json:"lines"` // Default 50
	Level string `json:"level"` // "INFO", "ERROR", "DEBUG"
}

// ==========================================
// 3. DOCUMENT OPERATIONS
// ==========================================

// NOTE: DocAdd and DocAddBatch requests are handled via multipart/form-data.
// There is no JSON struct for the *Input*, but there is for the *Response*.

// DocUploadResponse is returned immediately after a batch POST.
type DocUploadResponse struct {
	BatchID  string   `json:"batch_id"`
	Status   string   `json:"status"`         // "queued"
	Accepted []string `json:"accepted_files"` // List of filenames accepted for processing
	Rejected []string `json:"rejected_files"` // List of filenames rejected (wrong type, too big)
	Message  string   `json:"message"`
}

// BatchStatusResponse is used for polling /api/v1/docs/batch/{id}
type BatchStatusResponse struct {
	BatchID     string   `json:"batch_id"`
	Status      string   `json:"status"` // "processing", "completed", "failed"
	TotalFiles  int      `json:"total_files"`
	Processed   int      `json:"processed_files"`
	Failed      int      `json:"failed_files"`
	ProgressPct float32  `json:"progress_pct"` // 0.0 to 100.0
	CurrentFile string   `json:"current_file,omitempty"`
	Failures    []string `json:"failure_reasons,omitempty"` // Detailed error per file if any
}

// DocListRequest handles complex filtering for the document table.
type DocListRequest struct {
	Page     int                    `json:"page"`              // Default: 1
	PageSize int                    `json:"page_size"`         // Default: 20
	SortBy   string                 `json:"sort_by"`           // "created_at", "name", "size"
	Order    string                 `json:"order"`             // "asc", "desc"
	Filters  map[string]interface{} `json:"filters,omitempty"` // e.g. {"author": "John"}
}

// DocResponse represents a summary of a document.
type DocResponse struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	Size       int64                  `json:"size_bytes"`
	ChunkCount int                    `json:"chunk_count"`
	CreatedAt  string                 `json:"created_at"` // ISO8601
	Metadata   map[string]interface{} `json:"metadata"`
}

// DocFullResponse includes the actual text content.
type DocFullResponse struct {
	DocResponse
	Content string `json:"content"`
}

type DocListResponse struct {
	Docs       []DocResponse `json:"docs"`
	Total      int           `json:"total"`
	TotalPages int           `json:"total_pages"`
}

// ==========================================
// 4. DATABASE OPERATIONS
// ==========================================

type DBCreateRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type DBUseRequest struct {
	Name string `json:"name"`
}

type DBInfoResponse struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	DocCount    int    `json:"doc_count"`
	ChunkCount  int    `json:"chunk_count"`
	VectorCount int    `json:"vector_count"` // Validated against Faiss
	DiskSize    string `json:"disk_size"`    // Human readable
	CreatedAt   string `json:"created_at"`
	IsActive    bool   `json:"is_active"`
}

type DBListResponse struct {
	Databases []DBInfoResponse `json:"databases"`
}

// ==========================================
// 5. KNOWLEDGE GRAPH
// ==========================================

// KGEntity represents a node in the graph.
type KGEntity struct {
	ID        string `json:"id"`
	Text      string `json:"text"`      // The entity name
	Label     string `json:"label"`     // PERSON, ORG, LOC
	Frequency int    `json:"frequency"` // How many times it appears across docs
}

type KGListRequest struct {
	Page     int      `json:"page"`
	PageSize int      `json:"page_size"`
	MinFreq  int      `json:"min_freq"` // Only show entities appearing > X times
	Types    []string `json:"types"`    // Filter by ["PERSON", "ORG"]
	SortBy   string   `json:"sort_by"`  // "frequency", "text"
}

type KGListResponse struct {
	Entities []KGEntity `json:"entities"`
	Total    int        `json:"total"`
}

type KGQueryRequest struct {
	Query string `json:"query"` // Natural language or structured query for graph
	Depth int    `json:"depth"` // How many hops to traverse
}

type KGInfoResponse struct {
	NodeCount    int            `json:"node_count"`
	EdgeCount    int            `json:"edge_count"`
	TopEntities  []KGEntity     `json:"top_entities"`
	LabelDistrib map[string]int `json:"label_distribution"` // e.g. {"PERSON": 50, "ORG": 20}
}

// ==========================================
// 6. SEARCH OPERATIONS
// ==========================================

// SearchResult is the unified return object for all search types.
type SearchResult struct {
	DocID      string                 `json:"doc_id"`
	ChunkID    int                    `json:"chunk_id"`
	Content    string                 `json:"content"`
	Score      float32                `json:"score"`
	Metadata   map[string]interface{} `json:"metadata"`
	Highlights []string               `json:"highlights,omitempty"` // For FTS/Keyword matches
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Took    string         `json:"took_ms"` // Execution time
	Total   int            `json:"total_hits"`
}

// BaseSearchReq contains fields common to ALL searches.
type BaseSearchReq struct {
	Query    string                 `json:"query"`
	TopK     int                    `json:"top_k"`     // Default: 10
	MinScore float32                `json:"min_score"` // Default: 0.0
	Filters  map[string]interface{} `json:"filters,omitempty"`
}

// SearchSemanticReq - Standard RAG
type SearchSemanticReq struct {
	BaseSearchReq
}

// SearchCodeReq - Gemma Code optimized
type SearchCodeReq struct {
	BaseSearchReq
	Language string `json:"language,omitempty"` // "python", "go", etc.
}

// SearchFactReq - Gemma Fact Check optimized
type SearchFactReq struct {
	BaseSearchReq
	Strictness int `json:"strictness"` // 1-5 scale
}

// SearchKeywordReq - SQLite FTS5 + BM25
type SearchKeywordReq struct {
	BaseSearchReq
	ExactMatch     bool     `json:"exact_match"`
	PrefixMatch    bool     `json:"prefix_match"`
	FuzzyDistance  int      `json:"fuzzy_distance"`   // 0, 1, or 2
	MustContain    []string `json:"must_contain"`     // AND logic
	MustNotContain []string `json:"must_not_contain"` // NOT logic
}

// SearchHybridReq - Weighted Semantic + Keyword
type SearchHybridReq struct {
	BaseSearchReq
	Alpha float32 `json:"alpha"` // 0.0 to 1.0 (Weight of Semantic vs Keyword)
}
