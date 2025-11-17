# rag_anywhere/core/splitters/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextSplitter(ABC):
    """Base interface for text splitting strategies"""
    
    def __init__(self, token_estimator: Optional[Callable[[str], int]] = None):
        self.token_estimator = token_estimator or self._default_token_estimate
    
    @staticmethod
    def _default_token_estimate(text: str) -> int:
        """Conservative estimate: 4 chars per token"""
        return len(text) // 4
    
    @abstractmethod
    def split(self, text: str) -> List[TextChunk]:
        """Split text into chunks"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Splitter identifier"""
        pass
