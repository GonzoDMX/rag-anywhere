# rag_anywhere/core/loaders/__init__.py
"""Document loading subsystem"""

from .base import DocumentLoader
from .text import TextLoader
from .pdf import PDFLoader
from .docx import DocxLoader
from .registry import LoaderRegistry

__all__ = [
    'DocumentLoader',
    'TextLoader',
    'PDFLoader',
    'DocxLoader',
    'LoaderRegistry',
]
