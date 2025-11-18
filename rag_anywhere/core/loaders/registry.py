# rag_anywhere/core/loaders/registry.py

from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import DocumentLoader
from .text import TextLoader
from .pdf import PDFLoader
from .docx import DocxLoader


class LoaderRegistry:
    """
    Registry for document loaders.
    Manages available loaders and routes files to appropriate loader.
    """
    
    def __init__(self):
        self.loaders: List[DocumentLoader] = []
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register built-in loaders"""
        # Order matters - first match wins
        self.loaders.append(TextLoader())
        
        # PDF and DOCX loaders may fail to import if dependencies missing
        try:
            self.loaders.append(PDFLoader())
        except ImportError as e:
            print(f"Warning: PDF loader not available: {e}")
        
        try:
            self.loaders.append(DocxLoader())
        except ImportError as e:
            print(f"Warning: DOCX loader not available: {e}")
    
    def get_loader(self, file_path: Path) -> Optional[DocumentLoader]:
        """
        Get appropriate loader for file
        
        Args:
            file_path: Path to file
            
        Returns:
            DocumentLoader instance or None if no loader supports the file
        """
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None
    
    def register(self, loader: DocumentLoader, prepend: bool = True):
        """
        Register a custom loader
        
        Args:
            loader: DocumentLoader instance
            prepend: If True, add to beginning (higher priority)
        """
        if prepend:
            self.loaders.insert(0, loader)
        else:
            self.loaders.append(loader)
    
    def load_document(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """
        Load document using appropriate loader
        
        Args:
            file_path: Path to document
            
        Returns:
            Tuple of (content, metadata)
            
        Raises:
            ValueError: If no loader supports the file type
        """
        file_path = Path(file_path)
        
        loader = self.get_loader(file_path)
        if loader is None:
            raise ValueError(
                f"No loader available for file type: {file_path.suffix}. "
                f"Supported types: {self.get_supported_extensions()}"
            )
        
        content = loader.load(file_path)
        metadata = loader.get_metadata(file_path)
        
        return content, metadata
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        extensions = set()
        for loader in self.loaders:
            if hasattr(loader, 'SUPPORTED_EXTENSIONS'):
                extensions.update(loader.SUPPORTED_EXTENSIONS)
            elif isinstance(loader, PDFLoader):
                extensions.add('.pdf')
            elif isinstance(loader, DocxLoader):
                extensions.update(['.docx', '.doc'])
        return sorted(extensions)
