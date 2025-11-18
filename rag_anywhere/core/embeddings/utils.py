# rag_anywhere/core/embeddings/utils.py

import numpy as np

from .base import EmbeddingProvider


class DimensionReducerWrapper(EmbeddingProvider):
    """
    Wraps a provider with higher dimensions and reduces to 768
    Uses simple truncation + re-normalization (works well in practice)
    """
    
    def __init__(self, base_provider: EmbeddingProvider, target_dim: int = 768):
        self.base_provider = base_provider
        self.target_dim = target_dim
        
        if base_provider.dimension < target_dim:
            raise ValueError(
                f"Cannot reduce from {base_provider.dimension}d to {target_dim}d"
            )
    
    @property
    def dimension(self) -> int:
        return self.target_dim
    
    @property
    def max_tokens(self) -> int:
        return self.base_provider.max_tokens
    
    @property
    def name(self) -> str:
        return f"{self.base_provider.name}-reduced-{self.target_dim}d"
    
    def estimate_tokens(self, text: str) -> int:
        return self.base_provider.estimate_tokens(text)
    
    def embed(self, texts: list) -> np.ndarray:
        # Get full-dimension embeddings
        embeddings = self.base_provider.embed(texts)
        
        # Simple truncation (preserve most important dimensions)
        reduced = embeddings[:, :self.target_dim]
        
        # Re-normalize after truncation
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / norms
        
        return reduced
