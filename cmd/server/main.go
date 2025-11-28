package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"github.com/GonzoDMX/rag-anywhere/internal/api"
	// "github.com/GonzoDMX/rag-anywhere/internal/ipc" // We will uncomment when we wire up Python
)

func main() {
	// 1. Setup Logger
	logger := log.New(os.Stdout, "[RAG-SERVER] ", log.LstdFlags)

	// 2. Initialize Dependencies (Placeholders for now)
	logger.Println("Initializing SQLite...")
	logger.Println("Initializing Python Worker Pools...")
	// embedPool := ipc.NewWorkerPool(...)

	// 3. Setup Router
	mux := http.NewServeMux()

	// --- General ---
	mux.HandleFunc("GET /health", api.HandleHealth)
	mux.HandleFunc("GET /api/v1/system/status", api.HandleStatus)
	mux.HandleFunc("POST /api/v1/system/restart", api.HandleRestart)
	mux.HandleFunc("POST /api/v1/system/logs", api.HandleLogs)
	mux.HandleFunc("POST /api/v1/system/port", api.HandleChangePort)

	// --- Documents ---
	mux.HandleFunc("POST /api/v1/docs/add", api.HandleDocAdd)
	mux.HandleFunc("POST /api/v1/docs/batch", api.HandleDocAddBatch)            // Start batch
	mux.HandleFunc("GET /api/v1/docs/batch/{id}", api.HandleBatchStatus)        // Poll status
	mux.HandleFunc("GET /api/v1/docs/batch/{id}/stream", api.HandleBatchEvents) // SSE Stream
	mux.HandleFunc("POST /api/v1/docs/list", api.HandleDocList)                 // Search/Filter docs
	mux.HandleFunc("POST /api/v1/docs/query", api.HandleDocQuery)               // Metadata query
	mux.HandleFunc("GET /api/v1/docs/{id}", api.HandleDocGet)                   // Get Full Text
	mux.HandleFunc("DELETE /api/v1/docs/{id}", api.HandleDocRemove)

	// --- Database ---
	mux.HandleFunc("POST /api/v1/db/create", api.HandleDBCreate)
	mux.HandleFunc("POST /api/v1/db/use", api.HandleDBUse)
	mux.HandleFunc("GET /api/v1/db/list", api.HandleDBList)
	mux.HandleFunc("GET /api/v1/db/info", api.HandleDBInfo)
	mux.HandleFunc("DELETE /api/v1/db/{name}", api.HandleDBDelete)

	// --- Knowledge Graph ---
	mux.HandleFunc("GET /api/v1/kg/entities", api.HandleKGListEntities)
	mux.HandleFunc("POST /api/v1/kg/entity", api.HandleKGShowEntity)
	mux.HandleFunc("GET /api/v1/kg/info", api.HandleKGInfo)
	mux.HandleFunc("POST /api/v1/kg/query", api.HandleKGQuery)

	// --- Search ---
	mux.HandleFunc("POST /api/v1/search/semantic", api.HandleSearchSemantic)
	mux.HandleFunc("POST /api/v1/search/code", api.HandleSearchCode)
	mux.HandleFunc("POST /api/v1/search/question", api.HandleSearchQuestion)
	mux.HandleFunc("POST /api/v1/search/fact", api.HandleSearchFact)
	mux.HandleFunc("POST /api/v1/search/keyword", api.HandleSearchKeyword)
	mux.HandleFunc("POST /api/v1/search/hybrid", api.HandleSearchHybrid)
	mux.HandleFunc("POST /api/v1/search/kg", api.HandleSearchKG)

	// 4. Middleware Chain
	// We wrap the entire mux with middleware (CORS, Logging)
	handler := MiddlewareChain(mux, logger)

	// 5. Start Server
	port := ":8080"
	logger.Printf("Server starting on %s", port)

	server := &http.Server{
		Addr:         port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second, // Important for long SSE streams!
		IdleTimeout:  60 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil {
		logger.Fatal(err)
	}
}

// MiddlewareChain wraps the router with Logging and CORS
func MiddlewareChain(next http.Handler, logger *log.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// CORS (Important for Tauri/Frontend)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)

		logger.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
