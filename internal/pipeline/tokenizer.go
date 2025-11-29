package pipeline

import (
	"regexp"
)

// WhitespaceTokenSplitter mimics the python re.compile(r'\w+(?:[-_]\w+)*|\S')
// It allows us to split text into tokens exactly like GLiNER does.
var tokenRegex = regexp.MustCompile(`\w+(?:[-_]\w+)*|\S`)

// SubChunk represents a slice of text prepared for GLiNER
type SubChunk struct {
	Text         string
	StartCharIdx int // Where this subchunk starts in the parent chunk
	EndCharIdx   int
}

// CreateSubChunks splits a large string into overlapping chunks based on TOKEN count.
// maxTokens: 512 (GLiNER limit) - safety margin (e.g. use 450)
// overlap: 50 tokens
func CreateSubChunks(text string, maxTokens int, overlap int) []SubChunk {
	if maxTokens <= 0 {
		maxTokens = 400 // Safe default
	}
	if overlap >= maxTokens {
		overlap = maxTokens / 10
	}

	// 1. Find all tokens and their byte positions
	// FindAllStringIndex returns [[start, end], [start, end], ...]
	tokenIndices := tokenRegex.FindAllStringIndex(text, -1)

	if len(tokenIndices) == 0 {
		return []SubChunk{}
	}

	var chunks []SubChunk
	totalTokens := len(tokenIndices)

	// 2. Iterate with window
	step := maxTokens - overlap

	for i := 0; i < totalTokens; i += step {
		end := i + maxTokens
		if end > totalTokens {
			end = totalTokens
		}

		// Get the start byte of the first token in this window
		startByte := tokenIndices[i][0]
		// Get the end byte of the last token in this window
		endByte := tokenIndices[end-1][1]

		subText := text[startByte:endByte]

		chunks = append(chunks, SubChunk{
			Text:         subText,
			StartCharIdx: startByte,
			EndCharIdx:   endByte,
		})

		// Optimization: If we reached the end, stop
		if end == totalTokens {
			break
		}
	}

	return chunks
}
