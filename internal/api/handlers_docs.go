package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/GonzoDMX/rag-anywhere/internal/ingest"
	"github.com/GonzoDMX/rag-anywhere/internal/pipeline"
)

// ==========================================
// DOCUMENT OPERATIONS
// ==========================================

// These handlers manage document uploads, retrievals, deletions, and batch processing.

// HandleDocAdd - POST /api/v1/docs/add
// Synchronous: Uploads -> Extracts Text -> Returns Result (or Error).
func HandleDocAdd(w http.ResponseWriter, r *http.Request) {
	// 1. Parse Multipart (Max 32MB)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		errorResponse(w, http.StatusBadRequest, "File too large or invalid")
		return
	}

	// 2. Get File
	file, header, err := r.FormFile("file")
	if err != nil {
		errorResponse(w, http.StatusBadRequest, "Missing 'file' field")
		return
	}
	defer file.Close()

	// 3. Validation (Mime Type)
	buffer := make([]byte, 512)
	file.Read(buffer)
	file.Seek(0, 0)
	mime := http.DetectContentType(buffer)

	if !ingest.IsSupported(header.Filename, buffer) {
		errorResponse(w, http.StatusUnsupportedMediaType, "Unsupported file type: "+mime)
		return
	}

	// 4. Save to Staging
	path, err := saveFileToStaging(file, header.Filename)
	if err != nil {
		errorResponse(w, http.StatusInternalServerError, "Failed to save file")
		return
	}
	defer os.Remove(path) // Cleanup

	// 5. Get Metadata
	var meta map[string]interface{}
	if metaStr := r.FormValue("metadata"); metaStr != "" {
		json.Unmarshal([]byte(metaStr), &meta)
	}

	// 6. PIPELINE: Extract Text
	text, err := pipeline.ExtractDocument(path)
	if err != nil {
		errorResponse(w, http.StatusInternalServerError, "Extraction failed: "+err.Error())
		return
	}

	// Placeholder Response
	fakeDoc := DocResponse{
		ID:         "doc_123",
		Name:       header.Filename,
		Size:       header.Size,
		ChunkCount: 0,
		CreatedAt:  time.Now().Format(time.RFC3339),
		Metadata:   meta,
	}

	jsonResponse(w, http.StatusCreated, StandardResponse{
		Success: true,
		Data:    fakeDoc,
		Meta:    map[string]interface{}{"extracted_chars": len(text)},
	})
}

// HandleDocAddBatch - POST /api/v1/docs/batch
// Asynchronous: Returns Batch ID immediately. Background worker handles Extraction.
func HandleDocAddBatch(w http.ResponseWriter, r *http.Request) {
	// Max 100MB for batches
	if err := r.ParseMultipartForm(100 << 20); err != nil {
		errorResponse(w, http.StatusBadRequest, "Request too large")
		return
	}

	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		errorResponse(w, http.StatusBadRequest, "No files provided")
		return
	}

	batchID := fmt.Sprintf("batch_%d", time.Now().Unix())
	var validPaths []string
	var rejectedFiles []string

	for _, fileHeader := range files {
		f, err := fileHeader.Open()
		if err != nil {
			rejectedFiles = append(rejectedFiles, fileHeader.Filename)
			continue
		}

		// Mime Check
		buf := make([]byte, 512)
		f.Read(buf)
		f.Seek(0, 0)
		if !ingest.IsSupported(fileHeader.Filename, buf) {
			rejectedFiles = append(rejectedFiles, fileHeader.Filename+" (unsupported)")
			f.Close()
			continue
		}

		// Save
		path, err := saveFileToStaging(f, fileHeader.Filename)
		f.Close()

		if err != nil {
			rejectedFiles = append(rejectedFiles, fileHeader.Filename+" (save error)")
		} else {
			validPaths = append(validPaths, path)
		}
	}

	if len(validPaths) == 0 {
		errorResponse(w, http.StatusBadRequest, "No valid files found in batch")
		return
	}

	// Dispatch Background Worker
	go func(bID string, paths []string) {
		log.Printf("[Batch %s] Processing %d files...", bID, len(paths))
		for _, p := range paths {
			_, err := pipeline.ExtractDocument(p)
			if err != nil {
				log.Printf("[Batch %s] Failed to extract %s: %v", bID, p, err)
			} else {
				// TODO: Send to Embedder
			}
			os.Remove(p)
		}
	}(batchID, validPaths)

	jsonResponse(w, http.StatusAccepted, StandardResponse{
		Success: true,
		Data: DocUploadResponse{
			BatchID:  batchID,
			Status:   "queued",
			Accepted: validPaths,
			Rejected: rejectedFiles,
			Message:  fmt.Sprintf("Queued %d files", len(validPaths)),
		},
	})
}

// HandleDocList - POST /api/v1/docs/list
func HandleDocList(w http.ResponseWriter, r *http.Request) {
	var req DocListRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorResponse(w, http.StatusBadRequest, "Invalid JSON")
		return
	}
	// Placeholder
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Data: DocListResponse{Total: 0}})
}

// HandleDocGet - GET /api/v1/docs/{id}
func HandleDocGet(w http.ResponseWriter, r *http.Request) {
	docID := r.PathValue("id")
	jsonResponse(w, http.StatusOK, StandardResponse{
		Success: true,
		Data:    DocFullResponse{Content: "Content for " + docID},
	})
}

// HandleDocRemove - DELETE /api/v1/docs/{id}
func HandleDocRemove(w http.ResponseWriter, r *http.Request) {
	docID := r.PathValue("id")
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Data: map[string]string{"removed": docID}})
}

// HandleDocQuery - POST /api/v1/docs/query
func HandleDocQuery(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true})
}

// HandleBatchStatus - GET /api/v1/docs/batch/{id}
func HandleBatchStatus(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	jsonResponse(w, http.StatusOK, StandardResponse{
		Success: true,
		Data:    BatchStatusResponse{BatchID: id, Status: "processing", ProgressPct: 50.0},
	})
}

// HandleBatchEvents - GET /api/v1/docs/batch/{id}/stream
func HandleBatchEvents(w http.ResponseWriter, r *http.Request) {
	// SSE implementation will go here
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
}
