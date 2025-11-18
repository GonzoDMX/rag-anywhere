# rag_anywhere/core/splitters/factory.py

from typing import Dict, Any

from .base import TextSplitter
from .recursive import RecursiveTextSplitter
from .structural import StructuralTextSplitter


class SplitterFactory:
    """Factory for creating text splitters"""
    
    @staticmethod
    def create_splitter(strategy: str, **kwargs) -> TextSplitter:
        """
        Create a text splitter based on strategy
        
        Args:
            strategy: 'recursive' or 'structural'
            **kwargs: Additional arguments for the splitter
        """
        if strategy == 'recursive':
            return RecursiveTextSplitter(**kwargs)
        elif strategy == 'structural':
            return StructuralTextSplitter(**kwargs)
        else:
            raise ValueError(
                f"Unknown splitter strategy: {strategy}. "
                f"Available: 'recursive', 'structural'"
            )
    
    @staticmethod
    def list_splitters() -> Dict[str, Dict[str, Any]]:
        """Return info about available splitters"""
        return {
            'recursive': {
                'description': 'Character-based splitting with overlap',
                'best_for': 'General purpose, most document types',
                'parameters': ['chunk_size', 'chunk_overlap']
            },
            'structural': {
                'description': 'Splits on document structure (paragraphs, sections)',
                'best_for': 'Well-formatted documents with clear structure',
                'parameters': ['min_chunk_size', 'max_chunk_size']
            },
        }
