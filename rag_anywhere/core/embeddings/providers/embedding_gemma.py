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
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        sys.stderr.write(f"\n[EmbeddingGemma] Initializing with model '{model_name}'\n")
        sys.stderr.flush()

        logger.info(f"Initializing EmbeddingGemmaProvider with model '{model_name}'")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {platform.platform()}")
        logger.debug(f"Machine: {platform.machine()}")

        try:
            sys.stderr.write("[EmbeddingGemma] Importing sentence_transformers and torch...\n")
            sys.stderr.flush()

            from sentence_transformers import SentenceTransformer
            import torch

            sys.stderr.write(f"[EmbeddingGemma] PyTorch version: {torch.__version__}\n")
            sys.stderr.flush()

            logger.debug(f"PyTorch version: {torch.__version__}")
            logger.debug(f"sentence-transformers imported successfully")
        except ImportError as e:
            sys.stderr.write(f"[EmbeddingGemma] IMPORT ERROR: {e}\n")
            sys.stderr.flush()
            logger.error(f"Failed to import required dependencies: {e}")
            raise ImportError(
                "EmbeddingGemma requires 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name

        sys.stderr.write("[EmbeddingGemma] Auto-detecting device...\n")
        sys.stderr.flush()

        # Auto-detect device based on installed packages
        logger.debug("Auto-detecting device...")

        # Check CUDA availability (requires faiss-gpu and torch with CUDA)
        cuda_available = torch.cuda.is_available()
        logger.debug(f"CUDA available: {cuda_available}")

        # Check MPS availability (Apple Silicon)
        mps_available = False
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            logger.debug(f"MPS available: {mps_available}")

        # Select device with fallback
        if cuda_available:
            self.device = "cuda"
            logger.info("Using CUDA device (GPU)")
            sys.stderr.write("[EmbeddingGemma] Selected CUDA device\n")
        elif mps_available:
            # Use CPU by default on macOS for stability
            # MPS can be enabled in future if stability improves
            self.device = "cpu"
            logger.info("Using CPU device (MPS available but preferring CPU for stability)")
            sys.stderr.write("[EmbeddingGemma] Selected CPU device (MPS available but using CPU)\n")
        else:
            self.device = "cpu"
            logger.info("Using CPU device")
            sys.stderr.write("[EmbeddingGemma] Selected CPU device\n")
        sys.stderr.flush()

        try:
            # Check if model is a local path or HuggingFace model
            from pathlib import Path
            is_local_path = (
                model_name.startswith(('.', '/', '~')) or
                Path(model_name).exists()
            )

            if is_local_path:
                # Local model path
                local_path = Path(model_name).expanduser().resolve()
                logger.info(f"Loading local model from: {local_path} on device {self.device}")
                sys.stderr.write(f"[EmbeddingGemma] Loading local model from: {local_path}\n")
                sys.stderr.flush()
            else:
                # HuggingFace model - check cache
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                model_cache_name = f"models--{model_name.replace('/', '--')}"
                model_cached = (cache_dir / model_cache_name).exists()

                if model_cached:
                    logger.info(f"Loading cached model '{model_name}' from {cache_dir / model_cache_name} on device {self.device}")
                    sys.stderr.write(f"[EmbeddingGemma] Loading from cache: {cache_dir / model_cache_name}\n")
                else:
                    logger.info(f"Downloading model '{model_name}' (~1.2GB for embeddinggemma-300m)")
                    logger.info(f"Model will be cached to: {cache_dir / model_cache_name}")
                    logger.info(f"Loading on device {self.device}...")
                    sys.stderr.write(f"[EmbeddingGemma] Downloading model '{model_name}'...\n")
                sys.stderr.flush()

            sys.stderr.write(f"[EmbeddingGemma] About to call SentenceTransformer('{model_name}', device='{self.device}')...\n")
            sys.stderr.flush()

            self.model = SentenceTransformer(model_name, device=self.device)

            sys.stderr.write("[EmbeddingGemma] SentenceTransformer loaded successfully!\n")
            sys.stderr.flush()

            logger.info("✓ EmbeddingGemma loaded successfully")
            logger.debug(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            sys.stderr.write(f"[EmbeddingGemma] EXCEPTION during model load: {type(e).__name__}: {e}\n")
            import traceback
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
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
