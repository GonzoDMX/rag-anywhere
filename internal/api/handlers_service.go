package api

import (
	"encoding/json"
	"net/http"
)

// ==========================================
// SERVICE OPERATIONS
// ==========================================

// These handlers provide basic service operations such as health checks,
// status reporting, restarting the service, and log retrieval.

// HandleHealth - GET /api/v1/service/health
func HandleHealth(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, map[string]string{"status": "alive"})
}

func HandleStatus(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{
		Success: true,
		Data: StatusResponse{
			Status:   "healthy",
			Uptime:   "10m",
			ActiveDB: "default",
			Version:  "0.1.0",
		},
	})
}

// HandleStatus - GET /api/v1/service/status
func HandleRestart(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Message: "Restarting service..."})
}

// HandleRestart - POST /api/v1/service/restart
func HandleLogs(w http.ResponseWriter, r *http.Request) {
	var req LogsRequest
	json.NewDecoder(r.Body).Decode(&req)
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Data: "Log data placeholder..."})
}

// HandleLogs - POST /api/v1/service/logs
func HandleChangePort(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, http.StatusOK, StandardResponse{Success: true, Message: "Port change requested"})
}
