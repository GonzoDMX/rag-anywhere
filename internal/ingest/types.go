package ingest

import (
	"net/http"
	"path/filepath"
	"strings"
)

// SupportedExtensions defines the allow-list for file extensions.
// We use a map for O(1) lookups.
var SupportedExtensions = map[string]bool{
	// Plain Text
	".txt":      true,
	".md":       true,
	".markdown": true,

	// Rich Text / Office
	".pdf":  true,
	".rtf":  true,
	".doc":  true,
	".docx": true,

	// Future: Code files
	// ".py": true, ".go": true, ".js": true,
}

// IsSupported determines if a file should be processed based on its
// content (Magic Numbers) and its name (Extension).
func IsSupported(filename string, headerBytes []byte) bool {
	// 1. Get the file extension (lowercase)
	ext := strings.ToLower(filepath.Ext(filename))

	// If the extension isn't even on our list, reject immediately.
	// This saves us from trying to parse .exe or .iso files even if they mimic text.
	if !SupportedExtensions[ext] {
		return false
	}

	// 2. Sniff the MIME type from the first 512 bytes
	// Go's http.DetectContentType is reliable for binaries, less so for text.
	mime := http.DetectContentType(headerBytes)

	// 3. Complex Logic for Specific Formats

	// CASE A: PDF (Very Reliable)
	if mime == "application/pdf" {
		return true
	}

	// CASE B: DOCX (Tricky)
	// DOCX files are actually ZIP archives containing XML.
	// Go detects them as "application/zip". We must allow ZIP mime ONLY if ext is .docx
	if mime == "application/zip" && ext == ".docx" {
		return true
	}

	// CASE C: DOC (Old Word)
	// Often detected as application/msword or application/octet-stream (OLE2)
	if mime == "application/msword" || mime == "application/octet-stream" {
		if ext == ".doc" {
			return true
		}
	}

	// CASE D: RTF (Rich Text)
	// Can be "text/rtf", "application/rtf", or just "text/plain" depending on headers
	if strings.Contains(mime, "rtf") || (strings.HasPrefix(mime, "text/plain") && ext == ".rtf") {
		return true
	}

	// CASE E: Plain Text / Markdown
	// Go says "text/plain; charset=utf-8".
	// We trust the extension map we checked in Step 1.
	if strings.HasPrefix(mime, "text/plain") {
		return true
	}

	// If we got here, the file has a valid extension (like .docx)
	// but the content didn't match what we expected (e.g., a text file renamed to .docx).
	return false
}

// GetProcessorType returns a standardized string for which parser to use.
// Useful for the pipeline to know if it should use the Go PDF parser or Python.
func GetProcessorType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".pdf":
		return "pdf"
	case ".docx", ".doc":
		return "word"
	case ".rtf":
		return "rtf"
	case ".md", ".markdown", ".txt":
		return "text"
	default:
		return "unknown"
	}
}
