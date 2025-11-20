# rag_anywhere/core/loaders/pdf.py

from pathlib import Path
from typing import Dict, Any

from .base import DocumentLoader


class PDFLoader(DocumentLoader):
    """Loader for PDF documents"""
    
    def __init__(self):
        try:
            import pypdf
            self.pypdf = pypdf
        except ImportError as e:
            raise ImportError(
                "PDFLoader requires 'pypdf' package. "
                "Install with: pip install pypdf"
            ) from e
        
    def load(self, file_path: Path) -> str:
        """Load PDF file and extract text"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                reader = self.pypdf.PdfReader(f)
                
                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                
                return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Error loading PDF {file_path}: {e}")
    
    def supports(self, file_path: Path) -> bool:
        """Check if file is PDF"""
        return file_path.suffix.lower() == '.pdf'
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        stat = file_path.stat()
        metadata = {
            'filename': file_path.name,
            'file_size': stat.st_size,
            'file_type': '.pdf',
            'mime_type': 'application/pdf',
        }
        
        try:
            with open(file_path, 'rb') as f:
                reader = self.pypdf.PdfReader(f)
                metadata['num_pages'] = len(reader.pages)
                
                # Extract PDF metadata if available
                if reader.metadata:
                    if reader.metadata.title:
                        metadata['title'] = reader.metadata.title
                    if reader.metadata.author:
                        metadata['author'] = reader.metadata.author
                    if reader.metadata.subject:
                        metadata['subject'] = reader.metadata.subject
        except Exception:
            pass  # If metadata extraction fails, just return basic info
        
        return metadata
