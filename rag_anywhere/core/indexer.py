# rag_anywhere/core/indexer.py

from pathlib import Path
from typing import Optional, Dict, Any, List

from .loaders import LoaderRegistry
from .splitters import SplitterFactory
from .embeddings import EmbeddingProvider
from .document_store import DocumentStore
from .vector_store import VectorStore
from .keyword_search import KeywordSearcher
from .entity_store import EntityStore
from .gliner import GLiNERBatchProcessor


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
        entity_store: Optional[EntityStore] = None,
        gliner_processor: Optional[GLiNERBatchProcessor] = None,
        gliner_config: Optional[Dict[str, Any]] = None,
        loader_registry: Optional[LoaderRegistry] = None,
        splitter_strategy: str = "recursive",
        splitter_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.keyword_searcher = keyword_searcher
        self.entity_store = entity_store
        self.gliner_processor = gliner_processor
        self.gliner_config = gliner_config or {}
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
        # Store document and chunks (need doc_id for GLiNER processing)
        doc_id = self.document_store.add_document(
            filename=file_path.name,
            content=content,
            chunks=chunks,
            metadata=file_metadata
        )

        # Track what we've indexed for rollback on failure
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        vectors_stored = False
        fts_indexed = False

        try:
            # Update chunks with document metadata for GLiNER processing
            for i, chunk in enumerate(chunks):
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata['document_id'] = doc_id
                chunk.metadata['chunk_index'] = i

            print(f"Generating embeddings for {len(chunks)} chunks...")
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_provider.embed(chunk_texts)

            print(f"Storing vectors...")
            # Store vectors
            self.vector_store.add_batch(chunk_ids, embeddings)
            vectors_stored = True

            # Index in FTS5 for keyword search
            if self.keyword_searcher:
                print(f"Indexing for keyword search...")
                fts_chunks = [(chunk_id, chunk.content, "") for chunk_id, chunk in zip(chunk_ids, chunks)]
                self.keyword_searcher.index_chunks_batch(fts_chunks)
                fts_indexed = True

            # Extract entities with GLiNER
            if self.gliner_processor and self.entity_store:
                gliner_enabled = self.gliner_config.get('enabled', True)
                if gliner_enabled:
                    print(f"Extracting entities with GLiNER...")
                    default_labels = self.gliner_config.get('default_labels', [])
                    user_labels = file_metadata.get('gliner_labels', [])

                    # Process chunks to extract entities
                    chunk_entities_map = self.gliner_processor.process_chunks(
                        chunks,
                        default_labels,
                        user_labels
                    )

                    # Store entities in entity store
                    total_entities = 0
                    for chunk_id, chunk_entities in chunk_entities_map.items():
                        num_entities = self.entity_store.add_entities(
                            chunk_id,
                            chunk_entities.entities,
                            source='gliner'
                        )
                        total_entities += num_entities

                    print(f"✓ Extracted and stored {total_entities} entities")

            print(f"✓ Successfully indexed document '{file_path.name}' (ID: {doc_id})")
            return doc_id

        except Exception as e:
            # Rollback: Clean up partial indexing
            print(f"✗ Error during indexing, rolling back changes for '{file_path.name}'...")

            # Delete entities if any were stored
            if self.entity_store:
                for chunk_id in chunk_ids:
                    try:
                        self.entity_store.delete_chunk_entities(chunk_id)
                    except Exception:
                        pass  # Best effort cleanup

            # Delete from FTS5 if indexed
            if fts_indexed and self.keyword_searcher:
                try:
                    self.keyword_searcher.delete_chunks_batch(chunk_ids)
                except Exception:
                    pass  # Best effort cleanup

            # Delete vectors if stored
            if vectors_stored:
                try:
                    self.vector_store.delete(chunk_ids)
                except Exception:
                    pass  # Best effort cleanup

            # Delete document and chunks (cascades)
            try:
                self.document_store.delete_document(doc_id)
            except Exception:
                pass  # Best effort cleanup

            # Re-raise the original exception
            raise e
    
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

        # Delete entities from knowledge graph
        if self.entity_store and chunk_ids:
            for chunk_id in chunk_ids:
                self.entity_store.delete_chunk_entities(chunk_id)

        print(f"✓ Removed document {doc_id} and {len(chunk_ids)} vectors")
        return True
