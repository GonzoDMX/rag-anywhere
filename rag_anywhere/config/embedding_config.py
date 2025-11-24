"""Global embedding configuration for RAG Anywhere.

This module defines the embedding model settings that apply to all databases.
The embedding version is used to track breaking changes and enforce update policies.
"""

# Embedding Model Configuration
EMBEDDING_VERSION = "1.0.0"  # Increment when model or dimension changes
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_TOKENS = 2048

# Singleton instance
_embedding_provider = None


def get_embedding_provider():
    """Get or create the global embedding provider singleton.

    Returns:
        EmbeddingGemmaProvider: The global embedding provider instance.
    """
    global _embedding_provider

    if _embedding_provider is None:
        from rag_anywhere.core.embeddings.providers.embedding_gemma import (
            EmbeddingGemmaProvider,
        )
        _embedding_provider = EmbeddingGemmaProvider(model_name=EMBEDDING_MODEL)

    return _embedding_provider


def reset_embedding_provider():
    """Reset the global embedding provider (mainly for testing)."""
    global _embedding_provider
    _embedding_provider = None
