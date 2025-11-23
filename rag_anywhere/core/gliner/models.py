"""Data models for GLiNER entity extraction."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Entity:
    """Represents an extracted entity from text."""

    text: str
    label: str
    start_idx: int
    end_idx: int
    score: float

    def __hash__(self):
        """Hash based on text and label for deduplication."""
        return hash((self.text.lower(), self.label))

    def __eq__(self, other):
        """Equality based on text and label (case-insensitive)."""
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.label == other.label


@dataclass
class SubChunk:
    """Represents a sub-chunk of text for GLiNER processing."""

    content: str
    start_char: int  # Position in parent chunk
    end_char: int  # Position in parent chunk
    parent_chunk_id: Optional[str] = None


@dataclass
class ChunkEntities:
    """Aggregated entities for a single chunk."""

    chunk_id: str
    entities: List[Entity]

    def deduplicate(self):
        """Remove duplicate entities, keeping highest score for each unique (text, label) pair."""
        entity_map = {}
        for entity in self.entities:
            key = (entity.text.lower(), entity.label)
            if key not in entity_map or entity.score > entity_map[key].score:
                entity_map[key] = entity
        self.entities = list(entity_map.values())
