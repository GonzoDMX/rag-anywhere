# rag_anywhere/core/loaders/text.py
from pathlib import Path
from typing import Dict, Any

from .base import DocumentLoader


class TextLoader(DocumentLoader):
    """Loader for plain text and markdown files"""
    
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.markdown', '.rst', '.text']
    
    def load(self, file_path: Path) -> str:
        """Load text file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Could not decode file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is a supported text format"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from text file"""
        stat = file_path.stat()
        return {
            'filename': file_path.name,
            'file_size': stat.st_size,
            'file_type': file_path.suffix,
            'mime_type': 'text/plain',
        }
