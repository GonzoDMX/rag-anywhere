# rag_anywhere/core/indexer.py

from pathlib import Path
from typing import Optional, Dict, Any, List

from .loaders import LoaderRegistry
from .splitters import SplitterFactory
from .embeddings import EmbeddingProvider
from .document_store import DocumentStore
from .vector_store import VectorStore
from .keyword_search import KeywordSearcher


class Indexer:
    """
    Orchestrates the document ingestion pipeline:
    Load -> Split -> Embed -> Store (+ FTS5 Index)
    """

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        keyword_searcher: Optional[KeywordSearcher] = None,
        loader_registry: Optional[LoaderRegistry] = None,
        splitter_strategy: str = "recursive",
        splitter_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.keyword_searcher = keyword_searcher
        self.loader_registry = loader_registry or LoaderRegistry()

        # Create splitter
        splitter_kwargs = splitter_kwargs or {}
        self.splitter = SplitterFactory.create_splitter(
            splitter_strategy,
            token_estimator=embedding_provider.estimate_tokens,
            **splitter_kwargs
        )
    
    def index_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a single document
        
        Args:
            file_path: Path to document
            metadata: Optional additional metadata
            
        Returns:
            Document ID
        """
        file_path = Path(file_path)
        
        # Check if document already exists
        existing = self.document_store.get_document_by_filename(file_path.name)
        if existing:
            raise ValueError(
                f"Document '{file_path.name}' already exists with ID {existing['id']}. "
                "Please remove it first if you want to re-index."
            )
        
        print(f"Loading document: {file_path.name}")
        # Load document
        content, file_metadata = self.loader_registry.load_document(file_path)
        
        # Merge metadata
        if metadata:
            file_metadata.update(metadata)
        
        print(f"Splitting document into chunks...")
        # Split into chunks
        chunks = self.splitter.split(content)
        print(f"Created {len(chunks)} chunks")
        
        print(f"Storing document and chunks...")
        # Store document and chunks
        doc_id = self.document_store.add_document(
            filename=file_path.name,
            content=content,
            chunks=chunks,
            metadata=file_metadata
        )
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed(chunk_texts)
        
        print(f"Storing vectors...")
        # Store vectors
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.vector_store.add_batch(chunk_ids, embeddings)

        # Index in FTS5 for keyword search
        if self.keyword_searcher:
            print(f"Indexing for keyword search...")
            fts_chunks = [(chunk_id, chunk.content, "") for chunk_id, chunk in zip(chunk_ids, chunks)]
            self.keyword_searcher.index_chunks_batch(fts_chunks)

        print(f"✓ Successfully indexed document '{file_path.name}' (ID: {doc_id})")
        return doc_id
    
    def index_directory(
        self, 
        directory_path: Path,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Index all documents in a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            metadata: Optional metadata to apply to all documents
            
        Returns:
            List of document IDs
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Get supported extensions
        supported_exts = set(self.loader_registry.get_supported_extensions())
        
        # Find all supported files
        if recursive:
            files = [
                f for f in directory_path.rglob('*') 
                if f.is_file() and f.suffix.lower() in supported_exts
            ]
        else:
            files = [
                f for f in directory_path.glob('*') 
                if f.is_file() and f.suffix.lower() in supported_exts
            ]
        
        if not files:
            print(f"No supported documents found in {directory_path}")
            return []
        
        print(f"Found {len(files)} documents to index")
        
        doc_ids = []
        for file_path in files:
            try:
                doc_id = self.index_document(file_path, metadata)
                doc_ids.append(doc_id)
            except Exception as e:
                print(f"✗ Error indexing {file_path.name}: {e}")
        
        print(f"\n✓ Successfully indexed {len(doc_ids)}/{len(files)} documents")
        return doc_ids
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and its vectors
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was removed
        """
        # Get chunk IDs before deletion
        chunks = self.document_store.get_chunks_by_document(doc_id)
        chunk_ids = [chunk['id'] for chunk in chunks]
        
        # Delete from document store (cascades to chunks)
        if not self.document_store.delete_document(doc_id):
            return False
        
        # Delete vectors
        if chunk_ids:
            self.vector_store.delete(chunk_ids)

        # Delete from FTS5 index
        if self.keyword_searcher and chunk_ids:
            self.keyword_searcher.delete_chunks_batch(chunk_ids)

        print(f"✓ Removed document {doc_id} and {len(chunk_ids)} vectors")
        return True
