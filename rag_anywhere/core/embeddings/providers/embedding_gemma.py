# rag_anywhere/core/embeddings/providers/embedding_gemma.py
import numpy as np
from typing import List

from ..base import EmbeddingProvider


class EmbeddingGemmaProvider(EmbeddingProvider):
    """
    Local EmbeddingGemma-300m provider using sentence-transformers
    - 768 dimensions
    - 2048 token context
    - Multi-lingual support
    """
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "EmbeddingGemma requires 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        
        # Handle device parameter
        if device == "auto" or device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        print(f"Loading EmbeddingGemma on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"✓ EmbeddingGemma loaded successfully")
    
    @property
    def dimension(self) -> int:
        return 768
    
    @property
    def max_tokens(self) -> int:
        return 2048
    
    @property
    def name(self) -> str:
        return "embeddinggemma-300m"
    
    def estimate_tokens(self, text: str) -> int:
        """Quick token estimation without full tokenization"""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        Uses encode_document since we're embedding document chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), 768)
        """
        # Use encode_document for document chunks
        # sentence-transformers handles batching, normalization automatically
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Returns L2 normalized embeddings
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Uses the model's encode method with query-specific processing if available.
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (768,)
        """
        # Check if model has encode_query method (some sentence-transformers models do)
        if hasattr(self.model, 'encode_query'):
            embedding = self.model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            # Fallback to regular encode
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embedding
