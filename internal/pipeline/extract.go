package pipeline

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/GonzoDMX/rag-anywhere/internal/ingest"
	"github.com/dslipak/pdf" // Pure Go PDF text extractor
)

// MaxFileSize - 50MB hard limit for text extraction
const MaxFileSize = 50 * 1024 * 1024

// Extractor is the interface that different file parsers must implement
type Extractor interface {
	Extract(path string) (string, error)
}

// ExtractDocument is the main entry point.
// It determines the file type and calls the appropriate extractor.
func ExtractDocument(path string) (string, error) {
	// 1. Size Safety Check
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file not found: %w", err)
	}
	if info.Size() > MaxFileSize {
		return "", fmt.Errorf("file exceeds size limit of 50MB")
	}

	// 2. Identify Type (Reuse your ingest package logic)
	// We re-detect here just to be safe, or pass the type in.
	// For simplicity, we assume the handler passed a valid path with extension.
	fileType := ingest.GetProcessorType(path)

	switch fileType {
	case "text":
		return extractText(path)
	case "pdf":
		return extractPDF(path)
	case "word": // .docx
		return extractDOCX(path)
	case "rtf":
		return "", fmt.Errorf("RTF extraction requires python fallback") // Hard in pure Go
	case "unknown":
		return "", fmt.Errorf("unsupported file type")
	}

	return "", fmt.Errorf("no extractor found for %s", fileType)
}

// ---------------------------------------------------------
// 1. PLAIN TEXT EXTRACTOR (.txt, .md)
// ---------------------------------------------------------
func extractText(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	// Convert to string (Go handles UTF-8 naturally)
	return string(content), nil
}

// ---------------------------------------------------------
// 2. PDF EXTRACTOR
// Uses "github.com/dslipak/pdf"
// ---------------------------------------------------------
func extractPDF(path string) (string, error) {
	r, err := pdf.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF: %w", err)
	}

	var buf bytes.Buffer
	// GetPlainReader returns a reader that outputs the text content
	b, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("failed to read PDF text: %w", err)
	}

	buf.ReadFrom(b)
	return buf.String(), nil
}

// ---------------------------------------------------------
// 3. DOCX EXTRACTOR (Native Go / No Heavy Libs)
// DOCX is just a ZIP file. We unzip -> find word/document.xml -> strip tags.
// ---------------------------------------------------------
func extractDOCX(path string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", fmt.Errorf("failed to open DOCX zip: %w", err)
	}
	defer r.Close()

	// Find the main content XML
	var documentXML *zip.File
	for _, f := range r.File {
		if f.Name == "word/document.xml" {
			documentXML = f
			break
		}
	}

	if documentXML == nil {
		return "", fmt.Errorf("invalid docx: missing word/document.xml")
	}

	rc, err := documentXML.Open()
	if err != nil {
		return "", err
	}
	defer rc.Close()

	// Parse XML and extract text
	// We use a stream decoder to be memory efficient
	decoder := xml.NewDecoder(rc)
	var textBuilder strings.Builder

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}

		switch t := token.(type) {
		case xml.StartElement:
			// <w:p> indicates a paragraph break in Word
			if t.Name.Local == "p" {
				textBuilder.WriteString("\n")
			}
			// <w:tab/> indicates a tab
			if t.Name.Local == "tab" {
				textBuilder.WriteString("\t")
			}
		case xml.CharData:
			// The actual text content
			textBuilder.Write(t)
		}
	}

	return textBuilder.String(), nil
}
