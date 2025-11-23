"""Sub-chunking logic for GLiNER processing.

GLiNER has a 512 token context limit (~384 words). This module splits
embedding chunks into smaller sub-chunks suitable for GLiNER processing.
"""

from typing import List
from .models import SubChunk


class GLiNERSubChunker:
    """Splits text into sub-chunks for GLiNER processing."""

    def __init__(self, word_size: int = 320, overlap: int = 10):
        """
        Initialize sub-chunker.

        Args:
            word_size: Target size in words per sub-chunk (default 320 ~= 420 tokens)
            overlap: Number of words to overlap between sub-chunks (default 10)
        """
        self.word_size = word_size
        self.overlap = overlap

    def split(self, text: str, chunk_id: str = None) -> List[SubChunk]:
        """
        Split text into sub-chunks on word boundaries with overlap.

        Args:
            text: Text to split
            chunk_id: Parent chunk ID for tracking

        Returns:
            List of SubChunk objects
        """
        if not text or not text.strip():
            return []

        # Split on whitespace to get words
        words = text.split()

        if len(words) <= self.word_size:
            # Text is small enough, return as single sub-chunk
            return [
                SubChunk(
                    content=text, start_char=0, end_char=len(text), parent_chunk_id=chunk_id
                )
            ]

        sub_chunks = []
        start_word_idx = 0

        while start_word_idx < len(words):
            # Get slice of words for this sub-chunk
            end_word_idx = min(start_word_idx + self.word_size, len(words))
            chunk_words = words[start_word_idx:end_word_idx]

            # Reconstruct text from words
            chunk_text = " ".join(chunk_words)

            # Calculate character positions in original text
            # We need to find where this chunk starts and ends in the original text
            words_before = words[:start_word_idx]
            start_char = len(" ".join(words_before)) + (1 if words_before else 0)
            end_char = start_char + len(chunk_text)

            sub_chunks.append(
                SubChunk(
                    content=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    parent_chunk_id=chunk_id,
                )
            )

            # Move forward, accounting for overlap
            if end_word_idx >= len(words):
                break
            start_word_idx = end_word_idx - self.overlap

        return sub_chunks
