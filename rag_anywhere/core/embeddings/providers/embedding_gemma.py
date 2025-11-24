# rag_anywhere/core/embeddings/providers/embedding_gemma.py

import sys
import traceback
import platform
import numpy as np
from typing import List, Literal
from pathlib import Path

from ....utils.logging import get_logger

logger = get_logger('core.embeddings.embedding_gemma')

# Task type definitions for EmbeddingGemma prompts
TaskType = Literal[
    "retrieval",
    "question_answering",
    "fact_checking",
    "classification",
    "clustering",
    "similarity",
    "code_retrieval"
]


class EmbeddingGemmaProvider:
    """
    Local EmbeddingGemma-300m provider using sentence-transformers.

    This provider implements task-specific prompt formatting to optimize
    embeddings for different use cases:
    - Retrieval (document search)
    - Question answering
    - Fact verification
    - Classification
    - Clustering
    - Semantic similarity
    - Code retrieval

    Specifications:
    - 768 dimensions
    - 2048 token context
    - Multi-lingual support
    - Task-optimized embeddings via prompt formatting
    """

    # Task prompt templates
    TASK_PROMPTS = {
        "retrieval": "search result",
        "question_answering": "question answering",
        "fact_checking": "fact checking",
        "classification": "classification",
        "clustering": "clustering",
        "similarity": "sentence similarity",
        "code_retrieval": "code retrieval",
    }

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

            # Keep these imports here to reduce CLI lag on startup
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
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            logger.error(f"Failed to load model: {type(e).__name__}: {e}", exc_info=True)
            raise

    @property
    def dimension(self) -> int:
        """Vector dimension (always 768 for EmbeddingGemma)."""
        return 768

    @property
    def max_tokens(self) -> int:
        """Maximum token context window."""
        return 2048

    @property
    def name(self) -> str:
        """Provider name."""
        return self.model_name

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the actual tokenizer.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        try:
            # Use actual tokenizer for accurate count
            if hasattr(self.model, 'tokenizer'):
                tokens = self.model.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            else:
                # Fallback to rough approximation
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Failed to use tokenizer for estimation: {e}. Using fallback.")
            return len(text) // 4

    def format_document_chunk(self, title: str, content: str) -> str:
        """Format a document chunk with EmbeddingGemma's document prompt.

        Uses the format: "title: {title} | text: {content}"

        Args:
            title: Document title or chunk identifier (e.g., "doc_name_chunk_0")
            content: The actual text content of the chunk

        Returns:
            Formatted string ready for embedding
        """
        return f"title: {title} | text: {content}"

    def format_query(self, query: str, task: TaskType = "retrieval") -> str:
        """Format a query with EmbeddingGemma's task-specific prompt.

        Uses the format: "task: {task_description} | query: {query}"

        Args:
            query: The search query text
            task: Task type (retrieval, question_answering, fact_checking, etc.)

        Returns:
            Formatted query string ready for embedding
        """
        task_prompt = self.TASK_PROMPTS.get(task, "search result")
        return f"task: {task_prompt} | query: {query}"

    def format_code_retrieval(self, query: str) -> str:
        """Format a query for code retrieval.

        Convenience method for code search tasks.

        Args:
            query: Natural language query describing the code to find

        Returns:
            Formatted query for code retrieval
        """
        return self.format_query(query, task="code_retrieval")

    def format_fact_verification(self, query: str) -> str:
        """Format a query for fact verification.

        Convenience method for fact-checking tasks.

        Args:
            query: Statement to verify

        Returns:
            Formatted query for fact verification
        """
        return self.format_query(query, task="fact_checking")

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Note: Texts should be pre-formatted with appropriate prompts using
        format_document_chunk() or format_query() before calling this method.

        Args:
            texts: List of text strings to embed (should be pre-formatted)
            normalize: Whether to L2-normalize embeddings (default: True)

        Returns:
            numpy array of shape (len(texts), 768)
        """
        logger.debug(f"Embedding batch of {len(texts)} texts")

        try:
            # sentence-transformers handles batching, normalization automatically
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize  # L2 normalized for cosine similarity
            )

            logger.debug(f"Generated embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {type(e).__name__}: {e}", exc_info=True)
            raise

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed (should be pre-formatted)
            normalize: Whether to L2-normalize embedding (default: True)

        Returns:
            numpy array of shape (768,)
        """
        return self.embed([text], normalize=normalize)[0]

    def embed_query(self, query: str, task: TaskType = "retrieval") -> np.ndarray:
        """Generate embedding for a search query with task-specific formatting.

        This method automatically formats the query with the appropriate task prompt.

        Args:
            query: Raw query string (will be formatted automatically)
            task: Task type for prompt formatting (default: "retrieval")

        Returns:
            numpy array of shape (768,)
        """
        logger.debug(f"Embedding query with task '{task}': '{query[:100]}{'...' if len(query) > 100 else ''}'")

        try:
            # Format query with task-specific prompt
            formatted_query = self.format_query(query, task=task)

            # Generate embedding
            embedding = self.model.encode(
                formatted_query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Ensure we always return a numpy array
            if not isinstance(embedding, np.ndarray):
                import torch
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                else:
                    embedding = np.array(embedding)

            logger.debug(f"Generated query embedding with shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {type(e).__name__}: {e}", exc_info=True)
            raise
