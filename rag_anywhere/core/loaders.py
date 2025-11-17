# rag_anywhere/core/loaders.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class DocumentLoader(ABC):
    """Base class for document loaders"""
    
    @abstractmethod
    def load(self, file_path: Path) -> str:
        """Load document and return text content"""
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if this loader supports the file type"""
        pass

class TextLoader(DocumentLoader):
    def load(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix in ['.txt', '.md', '.markdown']

class PDFLoader(DocumentLoader):
    def load(self, file_path: Path) -> str:
        import pypdf
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix == '.pdf'

class DocxLoader(DocumentLoader):
    def load(self, file_path: Path) -> str:
        import docx
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix in ['.docx', '.doc']

# Future: CodeLoader, CSVLoader, etc.
class CodeLoader(DocumentLoader):
    """For future: handle code files with syntax awareness"""
    pass

class LoaderRegistry:
    """Registry pattern for extensibility"""
    def __init__(self):
        self.loaders = [
            TextLoader(),
            PDFLoader(),
            DocxLoader(),
        ]
    
    def get_loader(self, file_path: Path) -> Optional[DocumentLoader]:
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None
    
    def register(self, loader: DocumentLoader):
        """Allow users to add custom loaders"""
        self.loaders.insert(0, loader)
