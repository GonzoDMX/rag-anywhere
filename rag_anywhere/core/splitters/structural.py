# rag_anywhere/core/splitters/structural.py

import re
from typing import List, Tuple

from .base import TextSplitter, TextChunk
from .recursive import RecursiveTextSplitter


class StructuralTextSplitter(TextSplitter):
    """
    Splits based on document structure, respecting 2048 token limit
    Identifies headings, paragraphs, and other structural elements
    """
    
    def __init__(
        self, 
        min_chunk_size: int = 1000,   # ~250 tokens minimum
        max_chunk_size: int = 6000,   # ~1500 tokens maximum
        token_estimator = None
    ):
        super().__init__(token_estimator)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    @property
    def name(self) -> str:
        return "structural"
    
    def split(self, text: str) -> List[TextChunk]:
        chunks = []
        
        # Split by structural elements
        sections = self._identify_sections(text)
        
        current_chunk = ""
        current_start = 0
        
        for section_text, section_start in sections:
            estimated_tokens = self.token_estimator(section_text)
            current_tokens = self.token_estimator(current_chunk)
            
            # If section alone is too large, split it
            if estimated_tokens > 1800:
                if current_chunk:
                    # Save current chunk
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={
                            'split_type': 'structural',
                            'estimated_tokens': current_tokens,
                            'splitter': self.name
                        }
                    ))
                
                # Recursively split large section
                recursive = RecursiveTextSplitter(
                    chunk_size=self.max_chunk_size,
                    token_estimator=self.token_estimator
                )
                sub_chunks = recursive.split(section_text)
                chunks.extend(sub_chunks)
                
                current_chunk = ""
                current_start = section_start + len(section_text)
            
            # If adding would exceed max, save current and start new
            elif current_tokens + estimated_tokens > 1800:
                if current_chunk:
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={
                            'split_type': 'structural',
                            'estimated_tokens': current_tokens,
                            'splitter': self.name
                        }
                    ))
                current_chunk = section_text
                current_start = section_start
            
            # Otherwise, accumulate
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section_text
                else:
                    current_chunk = section_text
                    current_start = section_start
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={
                    'split_type': 'structural',
                    'estimated_tokens': self.token_estimator(current_chunk),
                    'splitter': self.name
                }
            ))
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, int]]:
        """Identify structural sections in document"""
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        current_pos = 0
        for para in paragraphs:
            if para.strip():
                sections.append((para, current_pos))
            current_pos += len(para) + 2  # Account for \n\n
        
        return sections
