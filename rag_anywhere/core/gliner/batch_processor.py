"""Batch processing orchestration for GLiNER entity extraction.

Handles:
1. Sub-chunking of large chunks into GLiNER-compatible sizes
2. Keyword batching (groups of 10 labels per pass)
3. Multi-pass extraction with different label sets
4. Aggregation and deduplication of results
"""

from typing import List, Dict, Optional
import logging

from .gliner_base import GLiNERExtractor
from .sub_chunker import GLiNERSubChunker
from .models import Entity, ChunkEntities, SubChunk

logger = logging.getLogger(__name__)


class GLiNERBatchProcessor:
    """Orchestrates batch entity extraction with sub-chunking and label batching."""

    def __init__(
        self,
        extractor: GLiNERExtractor,
        sub_chunker: Optional[GLiNERSubChunker] = None,
        max_labels_per_pass: int = 10,
    ):
        """
        Initialize batch processor.

        Args:
            extractor: GLiNER extractor instance
            sub_chunker: Sub-chunker instance (creates default if None)
            max_labels_per_pass: Maximum labels to pass to GLiNER per extraction pass
        """
        self.extractor = extractor
        self.sub_chunker = sub_chunker or GLiNERSubChunker()
        self.max_labels_per_pass = max_labels_per_pass

    def _batch_labels(
        self, default_labels: List[str], user_labels: List[str]
    ) -> List[List[str]]:
        """
        Batch labels into groups for multi-pass extraction.

        Strategy:
        - Always include default_labels as first batch (up to max_labels_per_pass)
        - Filter user_labels to remove duplicates from defaults
        - Split remaining into batches of max_labels_per_pass
        - Merge final batch if < 5 labels with previous batch

        Args:
            default_labels: Default entity labels (max 10)
            user_labels: User-provided labels

        Returns:
            List of label batches
        """
        batches = []

        # First batch: default labels (should already be <= max_labels_per_pass)
        if default_labels:
            batches.append(default_labels[: self.max_labels_per_pass])

        # Filter user labels to remove duplicates
        default_set = set(label.lower() for label in default_labels)
        unique_user_labels = [
            label for label in user_labels if label.lower() not in default_set
        ]

        if not unique_user_labels:
            return batches

        # Batch user labels
        remaining = unique_user_labels
        while remaining:
            batch = remaining[: self.max_labels_per_pass]
            remaining = remaining[self.max_labels_per_pass :]

            # Merge small final batch with previous batch
            if remaining and len(remaining) < 5:
                batch.extend(remaining)
                remaining = []

            batches.append(batch)

        return batches

    def process_chunks(
        self,
        chunks: List,  # List of TextChunk objects (from splitter)
        default_labels: List[str],
        user_labels: Optional[List[str]] = None,
    ) -> Dict[str, ChunkEntities]:
        """
        Process chunks with GLiNER entity extraction.

        For each chunk:
        1. Split into sub-chunks (~320 words)
        2. Extract entities with each label batch
        3. Aggregate sub-chunk results
        4. Deduplicate entities

        Args:
            chunks: List of TextChunk objects from document splitter
            default_labels: Default entity labels (10 generic types)
            user_labels: Optional user-provided labels

        Returns:
            Dict mapping chunk_id to ChunkEntities
        """
        user_labels = user_labels or []
        label_batches = self._batch_labels(default_labels, user_labels)

        if not label_batches:
            logger.warning("No labels provided for entity extraction")
            return {}

        logger.info(
            f"Processing {len(chunks)} chunks with {len(label_batches)} label batches"
        )

        all_chunk_entities = {}

        for chunk in chunks:
            chunk_id = f"{chunk.metadata.get('document_id', 'unknown')}_{chunk.metadata.get('chunk_index', 0)}"

            # Sub-chunk the content
            sub_chunks = self.sub_chunker.split(chunk.content, chunk_id)
            logger.debug(f"Chunk {chunk_id}: split into {len(sub_chunks)} sub-chunks")

            # Collect entities from all sub-chunks and label batches
            all_entities = []

            for batch_idx, label_batch in enumerate(label_batches):
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(label_batches)}: {len(label_batch)} labels"
                )

                # Extract from all sub-chunks with this label batch
                sub_chunk_texts = [sc.content for sc in sub_chunks]
                batch_results = self.extractor.extract_entities(
                    sub_chunk_texts, label_batch
                )

                # Aggregate entities from sub-chunks
                for sub_chunk, entities in zip(sub_chunks, batch_results):
                    # Adjust entity positions to be relative to parent chunk
                    for entity in entities:
                        entity.start_idx += sub_chunk.start_char
                        entity.end_idx += sub_chunk.start_char
                    all_entities.extend(entities)

            # Create ChunkEntities and deduplicate
            chunk_entities = ChunkEntities(chunk_id=chunk_id, entities=all_entities)
            chunk_entities.deduplicate()

            all_chunk_entities[chunk_id] = chunk_entities
            logger.debug(
                f"Chunk {chunk_id}: extracted {len(chunk_entities.entities)} unique entities"
            )

        return all_chunk_entities

    def process_single_chunk(
        self, chunk_text: str, chunk_id: str, labels: List[str]
    ) -> ChunkEntities:
        """
        Process a single chunk (convenience method).

        Args:
            chunk_text: Chunk text content
            chunk_id: Chunk identifier
            labels: Entity labels to extract

        Returns:
            ChunkEntities object
        """
        # Create minimal chunk-like object
        class SimpleChunk:
            def __init__(self, content, chunk_id):
                self.content = content
                self.metadata = {"document_id": chunk_id.split("_")[0], "chunk_index": 0}

        chunk = SimpleChunk(chunk_text, chunk_id)
        results = self.process_chunks([chunk], labels, [])
        return results.get(chunk_id, ChunkEntities(chunk_id=chunk_id, entities=[]))
