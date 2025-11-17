# rag_anywhere/core/splitters/__init__.py
"""Text splitting strategies"""

from .base import TextSplitter, TextChunk
from .factory import SplitterFactory
from .recursive import RecursiveTextSplitter
from .structural import StructuralTextSplitter

__all__ = [
    'TextSplitter',
    'TextChunk',
    'SplitterFactory',
    'RecursiveTextSplitter',
    'StructuralTextSplitter',
]
