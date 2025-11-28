package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// jsonResponse sends a standard JSON response
func jsonResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// errorResponse sends a standard Error response
func errorResponse(w http.ResponseWriter, status int, msg string) {
	jsonResponse(w, status, StandardResponse{
		Success: false,
		Error:   msg,
	})
}

// saveFileToStaging saves a multipart file to disk and returns the absolute path.
// It is used by both DocAdd and DocBatch handlers.
func saveFileToStaging(file io.Reader, filename string) (string, error) {
	// Ensure staging dir exists
	stagingDir := "./staging"
	if _, err := os.Stat(stagingDir); os.IsNotExist(err) {
		// 0755: Owner can read/write/exec, Group/Others can read/exec
		if err := os.Mkdir(stagingDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create staging dir: %w", err)
		}
	}

	// Create unique filename to prevent collisions (timestamp_filename)
	timestamp := time.Now().UnixNano()
	safeName := fmt.Sprintf("%d_%s", timestamp, filepath.Base(filename))
	path := filepath.Join(stagingDir, safeName)

	dst, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer dst.Close()

	if _, err := io.Copy(dst, file); err != nil {
		return "", err
	}
	return path, nil
}
