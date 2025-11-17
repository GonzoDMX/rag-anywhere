# rag_anywhere/core/splitters/recursive.py
from typing import List

from .base import TextSplitter, TextChunk


class RecursiveTextSplitter(TextSplitter):
    """
    Splits text recursively optimized for 2048 token models.
    Default chunk size: ~1500 tokens (~6000 chars) to stay safely under 2048
    """
    
    def __init__(
        self, 
        chunk_size: int = 6000,      # ~1500 tokens
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
            
            # Try to break at natural boundaries
            if end < len(text):
                # Priority order: paragraph > sentence > word
                for separator in ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']:
                    last_sep = chunk_text.rfind(separator)
                    if last_sep > self.chunk_size * 0.7:  # At least 70% of target
                        end = start + last_sep + len(separator)
                        chunk_text = text[start:end]
                        break
            
            # Verify we're under token limit
            estimated_tokens = self.token_estimator(chunk_text)
            while estimated_tokens > 1800:  # Safety margin under 2048
                # Reduce chunk size
                reduction = int(self.chunk_size * 0.1)
                end -= reduction
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
