# rag_anywhere/core/embeddings/__init__.py
"""Embedding subsystem - EmbeddingGemma only"""

from .providers.embedding_gemma import EmbeddingGemmaProvider

__all__ = [
    'EmbeddingGemmaProvider',
]
