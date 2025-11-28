package ipc

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
)

// =================================================================================
// 1. HELPER: Environment Detection
// =================================================================================

// getPythonCommand attempts to find the venv python, falling back to system defaults
func getPythonCommand() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "python3" // Fallback
	}

	var venvPath string
	if runtime.GOOS == "windows" {
		venvPath = filepath.Join(cwd, ".venv", "Scripts", "python.exe")
	} else {
		venvPath = filepath.Join(cwd, ".venv", "bin", "python")
	}

	// If .venv exists, use it
	if _, err := os.Stat(venvPath); err == nil {
		return venvPath
	}

	// Fallback to system path
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

// =================================================================================
// 2. SINGLE WORKER (PythonService)
// =================================================================================

// PythonService manages a single background Python process
type PythonService struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader // Changed from Scanner to Reader for large payloads
	mutex   sync.Mutex    // Ensures we don't send overlapping requests to the same process
	running bool
}

// NewPythonService starts a python worker
func NewPythonService(scriptPath string) (*PythonService, error) {
	pythonCmd := getPythonCommand()

	// "-u" flag forces Python to use unbuffered binary stdout.
	// This prevents the program from hanging waiting for output buffer to fill.
	cmd := exec.Command(pythonCmd, "-u", scriptPath)

	// Set working dir to project root to ensure imports work if needed
	cwd, _ := os.Getwd()
	cmd.Dir = cwd

	// Pipes to talk to the child process
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	// Redirect stderr to parent stderr so we can see Python logs in Go console
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start python script %s: %w", scriptPath, err)
	}

	return &PythonService{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewReader(stdout),
		running: true,
	}, nil
}

// Process sends a request to Python and waits for the response
// The payload can be any struct that serializes to JSON
func (s *PythonService) Process(req interface{}, resp interface{}) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.running {
		return fmt.Errorf("python worker is not running")
	}

	// 1. Encode Request to JSON
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshalling error: %w", err)
	}

	// 2. Write to Python (add newline so Python's sys.stdin.readline() knows to stop reading)
	_, err = s.stdin.Write(append(reqBytes, '\n'))
	if err != nil {
		return fmt.Errorf("failed to write to python: %w", err)
	}

	// 3. Read Response from Python
	// We use ReadBytes('\n') instead of Scanner because vector batches can be huge (>64KB)
	respBytes, err := s.stdout.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("failed to read from python (worker might have crashed): %w", err)
	}

	// 4. Decode Response
	if err := json.Unmarshal(respBytes, resp); err != nil {
		return fmt.Errorf("python returned invalid JSON: %s (err: %v)", string(respBytes), err)
	}

	return nil
}

// Close gracefully shuts down the worker
func (s *PythonService) Close() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.running {
		return
	}

	s.running = false
	// Closing stdin usually signals EOF to Python, causing it to exit the loop
	_ = s.stdin.Close()

	// Force kill if it doesn't exit
	if s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
	}
}

// =================================================================================
// 3. WORKER POOL (Load Balancer)
// =================================================================================

// WorkerPool manages multiple instances of the same Python script
// to allow parallel processing (Horizontal Scaling).
type WorkerPool struct {
	workers []*PythonService
	counter uint64 // Used for Round-Robin load balancing
}

// NewWorkerPool starts 'count' copies of the given python script
func NewWorkerPool(scriptPath string, count int) (*WorkerPool, error) {
	if count < 1 {
		count = 1
	}

	var workers []*PythonService

	for i := 0; i < count; i++ {
		w, err := NewPythonService(scriptPath)
		if err != nil {
			// If one fails, close any that already started to avoid leaks
			for _, started := range workers {
				started.Close()
			}
			return nil, fmt.Errorf("failed to start worker %d: %w", i, err)
		}
		workers = append(workers, w)
	}

	return &WorkerPool{
		workers: workers,
		counter: 0,
	}, nil
}

// Process distributes the task to the next available worker using Round-Robin
func (p *WorkerPool) Process(req interface{}, resp interface{}) error {
	if len(p.workers) == 0 {
		return fmt.Errorf("no workers available")
	}

	// Atomic increment ensures thread-safety when selecting a worker
	current := atomic.AddUint64(&p.counter, 1)
	workerIndex := current % uint64(len(p.workers))

	selectedWorker := p.workers[workerIndex]
	return selectedWorker.Process(req, resp)
}

// Close shuts down all workers in the pool
func (p *WorkerPool) Close() {
	for _, w := range p.workers {
		w.Close()
	}
}
