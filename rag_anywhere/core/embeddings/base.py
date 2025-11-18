# rag_anywhere/core/embeddings/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """
    Base interface for all embedding providers.
    All providers MUST output 768-dimensional vectors.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Must return 768"""
        pass
    
    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum token capacity"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier"""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts.
        Returns: (batch_size, 768) numpy array
        """
        pass
    
    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method for single text"""
        return self.embed([text])[0]
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query.

        Default implementation uses the same embedding as a single document
        via embed_single. Providers may override this for specialized
        query/document encoding.
        """
        return self.embed_single(query)
