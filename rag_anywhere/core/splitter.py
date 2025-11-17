# rag_anywhere/core/splitter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TextChunk:
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None

class TextSplitter(ABC):
    @abstractmethod
    def split(self, text: str) -> List[TextChunk]:
        pass

class RecursiveTextSplitter(TextSplitter):
    """Splits text recursively by character count with overlap"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[TextChunk]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # If we're close enough
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunks.append(TextChunk(
                content=chunk_text.strip(),
                start_char=start,
                end_char=end
            ))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

class StructuralTextSplitter(TextSplitter):
    """Splits based on document structure (headings, paragraphs)"""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def split(self, text: str) -> List[TextChunk]:
        import re
        chunks = []
        
        # Detect markdown/text structure patterns
        patterns = [
            r'^#{1,6}\s+.+$',  # Markdown headers
            r'^\d+\.\s+.+$',    # Numbered lists
            r'^\*\s+.+$',       # Bullet points
            r'\n\n+',           # Paragraph breaks
        ]
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds max size
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(TextChunk(
                    content=current_chunk.strip(),
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    metadata={'split_type': 'structural'}
                ))
                current_chunk = para
                start_char += len(current_chunk)
            else:
                current_chunk += "\n\n" + para if current_chunk else para
            
            # If current paragraph alone exceeds max, split it
            if len(current_chunk) > self.max_chunk_size:
                # Fall back to recursive splitter for this chunk
                recursive = RecursiveTextSplitter(
                    chunk_size=self.max_chunk_size, 
                    chunk_overlap=50
                )
                sub_chunks = recursive.split(current_chunk)
                chunks.extend(sub_chunks)
                current_chunk = ""
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata={'split_type': 'structural'}
            ))
        
        return chunks

class SplitterFactory:
    @staticmethod
    def get_splitter(strategy: str, **kwargs) -> TextSplitter:
        if strategy == 'recursive':
            return RecursiveTextSplitter(**kwargs)
        elif strategy == 'structural':
            return StructuralTextSplitter(**kwargs)
        else:
            raise ValueError(f"Unknown splitter strategy: {strategy}")
