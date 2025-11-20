# rag_anywhere/core/embeddings/providers/openai.py

import numpy as np
from typing import List
from openai import OpenAI

from ..base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider (text-embedding-3-small)
    Configured to output 768 dimensions
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """ Initialize OpenAI embedding provider. """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 768
    
    @property
    def dimension(self) -> int:
        return 768
    
    @property
    def max_tokens(self) -> int:
        return 8191
    
    @property
    def name(self) -> str:
        return f"openai-{self.model}"
    
    def estimate_tokens(self, text: str) -> int:
        """OpenAI: ~4 chars per token"""
        return len(text) // 4
    
    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=768  # Request 768-dim output
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
