# rag_anywhere/core/embeddings/__init__.py
"""Embedding subsystem"""

from .base import EmbeddingProvider
from .factory import EmbeddingProviderFactory
from .providers import (
    EmbeddingGemmaProvider,
    OpenAIEmbeddingProvider,
)

__all__ = [
    'EmbeddingProvider',
    'EmbeddingProviderFactory',
    'EmbeddingGemmaProvider',
    'OpenAIEmbeddingProvider',
]
