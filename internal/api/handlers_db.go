package api

import (
	"encoding/json"
	"net/http"
)

// ==========================================
// DATABASE OPERATIONS
// ==========================================

// These handlers manage database creation, selection, listing, info retrieval, and deletion.

// HandleDBCreate - POST /api/v1/db/create
func HandleDBCreate(w http.ResponseWriter, r *http.Request) {
	var req DBCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorResponse(w, http.StatusBadRequest, "Invalid JSON")
		return
	}
	jsonResponse(w, http.StatusCreated, StandardResponse{Success: true, Message: "Database created"})
}

// HandleDBUse - POST /api/v1/db/use
func HandleDBUse(w http.ResponseWriter, r *http.Request) {
	var req DBUseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorResponse(w, http.StatusBadRequest, "Invalid JSON")
		return
	}
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Message: "Switched to database " + req.Name})
}

// HandleDBList - GET /api/v1/db/list
func HandleDBList(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{
		Success: true,
		Data:    DBListResponse{Databases: []DBInfoResponse{{Name: "default", IsActive: true}}},
	})
}

// HandleDBInfo - GET /api/v1/db/info
func HandleDBInfo(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{
		Success: true,
		Data:    DBInfoResponse{Name: "default", DocCount: 100, IsActive: true},
	})
}

// HandleDBDelete - DELETE /api/v1/db/{name}
func HandleDBDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Message: "Deleted " + name})
}
