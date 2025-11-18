# rag_anywhere/core/embeddings/providers/embedding_gemma.py
import numpy as np
import sys
import platform
from typing import List

from ..base import EmbeddingProvider
from ....utils.logging import get_logger

logger = get_logger('core.embeddings.embedding_gemma')


class EmbeddingGemmaProvider(EmbeddingProvider):
    """
    Local EmbeddingGemma-300m provider using sentence-transformers
    - 768 dimensions
    - 2048 token context
    - Multi-lingual support
    """
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = None):
        logger.info(f"Initializing EmbeddingGemmaProvider with model '{model_name}'")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {platform.platform()}")
        logger.debug(f"Machine: {platform.machine()}")

        try:
            from sentence_transformers import SentenceTransformer
            import torch
            logger.debug(f"PyTorch version: {torch.__version__}")
            logger.debug(f"sentence-transformers imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            raise ImportError(
                "EmbeddingGemma requires 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name

        # Handle device parameter with platform-specific logic
        if device == "auto" or device is None:
            logger.debug("Auto-detecting device...")

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            logger.debug(f"CUDA available: {cuda_available}")

            # Check MPS availability (Apple Silicon)
            mps_available = False
            if hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available()
                logger.debug(f"MPS available: {mps_available}")

            # Select device with fallback
            if cuda_available:
                device = "cuda"
                logger.info("Selected CUDA device")
            elif mps_available:
                # Use CPU by default on macOS for stability
                device = "cpu"
                logger.info("MPS available but using CPU for stability")
            else:
                device = "cpu"
                logger.info("Selected CPU device")
        else:
            logger.info(f"Using user-specified device: {device}")

        self.device = device

        try:
            # Check if model is cached in HuggingFace cache
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache_name = f"models--{model_name.replace('/', '--')}"
            model_cached = (cache_dir / model_cache_name).exists()

            if model_cached:
                logger.info(f"Loading cached model '{model_name}' from {cache_dir / model_cache_name}")
                print(f"Loading EmbeddingGemma on {self.device}...")
            else:
                logger.info(f"Downloading model '{model_name}' (~1.2GB for embeddinggemma-300m)")
                logger.info(f"Model will be cached to: {cache_dir / model_cache_name}")
                print(f"Downloading and loading EmbeddingGemma on {self.device}...")

            self.model = SentenceTransformer(model_name, device=self.device)

            logger.info("Model loaded successfully")
            logger.debug(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            print(f"✓ EmbeddingGemma loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {type(e).__name__}: {e}", exc_info=True)
            raise
    
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
        logger.debug(f"Embedding batch of {len(texts)} texts")

        try:
            # Use encode_document for document chunks
            # sentence-transformers handles batching, normalization automatically
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Returns L2 normalized embeddings
            )

            logger.debug(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Uses the model's encode method with query-specific processing if available.

        Args:
            query: Query string

        Returns:
            numpy array of shape (768,)
        """
        logger.debug(f"Embedding query: '{query[:100]}{'...' if len(query) > 100 else ''}'")

        try:
            # Check if model has encode_query method (some sentence-transformers models do)
            if hasattr(self.model, 'encode_query'):
                logger.debug("Using encode_query method")
                embedding = self.model.encode_query(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                # Fallback to regular encode
                logger.debug("Using regular encode method")
                embedding = self.model.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

            logger.debug(f"Generated query embedding with shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {type(e).__name__}: {e}", exc_info=True)
            raise
