# rag_anywhere/core/embeddings/providers/__init__.py
"""Embedding provider implementations"""

from .embedding_gemma import EmbeddingGemmaProvider

__all__ = [
    'EmbeddingGemmaProvider',
]
