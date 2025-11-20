# rag_anywhere/core/splitters/recursive.py

from typing import List

from .base import TextSplitter, TextChunk


class RecursiveTextSplitter(TextSplitter):
    """
    Splits text recursively optimized for 2048 token models.
    Default chunk size: ~1500 tokens (~5000 chars) to stay safely under 2048
    """
    
    def __init__(
        self, 
        chunk_size: int = 5000,      # ~1500 tokens
        chunk_overlap: int = 600,     # ~150 tokens overlap
        token_estimator = None
    ):
        super().__init__(token_estimator)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @property
    def name(self) -> str:
        return "recursive"
    
    def split(self, text: str) -> List[TextChunk]:
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Get chunk text
            chunk_text = text[start:end]

            # Try to break at natural boundaries - NEVER break words mid-word
            if end < len(text):
                # Priority order: paragraph > sentence > space (any whitespace)
                # Try preferred separators first (70% threshold for good breaks)
                found_break = False
                for separator in ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ']:
                    last_sep = chunk_text.rfind(separator)
                    if last_sep > self.chunk_size * 0.7:  # At least 70% of target
                        end = start + last_sep + len(separator)
                        chunk_text = text[start:end]
                        found_break = True
                        break

                # If no good break found, find ANY whitespace to avoid breaking words
                if not found_break:
                    # Search backwards from end for any whitespace
                    last_space = chunk_text.rfind(' ')
                    last_newline = chunk_text.rfind('\n')
                    last_tab = chunk_text.rfind('\t')
                    last_whitespace = max(last_space, last_newline, last_tab)

                    if last_whitespace > 0:  # Found whitespace anywhere in chunk
                        end = start + last_whitespace + 1
                        chunk_text = text[start:end]
                    # If NO whitespace at all (pathological case: 6000+ chars no space)
                    # just hard break but this is extremely rare
            
            # Verify we're under token limit
            estimated_tokens = self.token_estimator(chunk_text)
            while estimated_tokens > 1800:  # Safety margin under 2048
                # Reduce chunk size and find whitespace boundary
                reduction = int(self.chunk_size * 0.1)
                end -= reduction

                # Find last whitespace before new end position
                temp_chunk = text[start:end]
                last_space = temp_chunk.rfind(' ')
                last_newline = temp_chunk.rfind('\n')
                last_tab = temp_chunk.rfind('\t')
                last_whitespace = max(last_space, last_newline, last_tab)

                if last_whitespace > 0:
                    end = start + last_whitespace + 1
                    chunk_text = text[start:end]
                else:
                    # Pathological case: no whitespace found, hard break
                    chunk_text = text[start:end]

                estimated_tokens = self.token_estimator(chunk_text)
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    start_char=start,
                    end_char=end,
                    metadata={
                        'estimated_tokens': estimated_tokens,
                        'splitter': self.name
                    }
                ))
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
