# rag_anywhere/core/loaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, ClassVar, List


class DocumentLoader(ABC):
    """Base class for document loaders"""
    
    # Subclasses may override this with a list of supported file extensions
    SUPPORTED_EXTENSIONS: ClassVar[List[str]] = []

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """
        Load document and return text content
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file can't be processed
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Check if this loader supports the file type
        
        Args:
            file_path: Path to check
            
        Returns:
            True if this loader can handle the file
        """
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from the document
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary of metadata (file size, type, etc.)
        """
        pass
