"""GLiNER entity extraction wrapper.

This module provides a simple interface to GLiNER v2.1 models for named entity recognition.
Supports four model sizes: small, medium, multi (default), and large.
"""

from typing import List, Optional, Dict
import logging

from .models import Entity

logger = logging.getLogger(__name__)


class GLiNERExtractor:
    """Wrapper for GLiNER entity extraction models."""

    MODEL_MAPPING = {
        "small": "urchade/gliner_small-v2.1",
        "medium": "urchade/gliner_medium-v2.1",
        "multi": "urchade/gliner_multi-v2.1",  # Default - multilingual support
        "large": "urchade/gliner_large-v2.1",
    }

    def __init__(
        self,
        model_size: str = "multi",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize GLiNER extractor.

        Args:
            model_size: Model size - one of 'small', 'medium', 'multi', 'large'
            confidence_threshold: Minimum confidence score for entity extraction
            device: Device to run model on ('cpu' or 'cuda')
            cache_dir: Directory to cache downloaded models
        """
        if model_size not in self.MODEL_MAPPING:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Must be one of: {list(self.MODEL_MAPPING.keys())}"
            )

        self.model_size = model_size
        self.model_name = self.MODEL_MAPPING[model_size]
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.cache_dir = cache_dir
        self._model = None  # Lazy loading

    def _load_model(self):
        """Lazy load the GLiNER model."""
        if self._model is not None:
            return

        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "GLiNER is not installed. Install with: pip install gliner"
            )

        logger.info(f"Loading GLiNER model: {self.model_name}")
        self._model = GLiNER.from_pretrained(self.model_name, local_files_only=False)

        # Move to device if specified
        if self.device == "cuda":
            import torch

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
                logger.info("GLiNER model loaded on CUDA")
            else:
                logger.warning("CUDA requested but not available, using CPU")
        else:
            logger.info("GLiNER model loaded on CPU")

    def extract_entities(
        self, texts: List[str], labels: List[str], threshold: Optional[float] = None
    ) -> List[List[Entity]]:
        """
        Extract entities from texts using specified labels.

        Args:
            texts: List of text strings to process
            labels: List of entity labels/types to extract
            threshold: Optional override for confidence threshold

        Returns:
            List of lists of Entity objects (one list per input text)
        """
        self._load_model()

        if not texts or not labels:
            return [[] for _ in texts]

        threshold = threshold if threshold is not None else self.confidence_threshold

        # GLiNER predict_entities expects a single text string and returns a list
        # We need to process each text separately
        all_entities = []
        for text in texts:
            text_entities = self._model.predict_entities(text, labels, threshold=threshold)
            entities = []
            for entity_dict in text_entities:
                entity = Entity(
                    text=entity_dict["text"],
                    label=entity_dict["label"],
                    start_idx=entity_dict["start"],
                    end_idx=entity_dict["end"],
                    score=entity_dict.get("score", 1.0),
                )
                entities.append(entity)
            all_entities.append(entities)

        return all_entities

    def extract_single(self, text: str, labels: List[str]) -> List[Entity]:
        """
        Extract entities from a single text.

        Args:
            text: Text to process
            labels: List of entity labels/types to extract

        Returns:
            List of Entity objects
        """
        results = self.extract_entities([text], labels)
        return results[0] if results else []

    def unload_model(self):
        """Unload the model from memory."""
        if self._model is not None:
            logger.info("Unloading GLiNER model")
            del self._model
            self._model = None

            # Clear CUDA cache if applicable
            if self.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA cache: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()
