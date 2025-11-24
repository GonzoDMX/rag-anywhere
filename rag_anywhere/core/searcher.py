# rag_anywhere/core/searcher.py

from typing import List, Dict, Any, Optional, Literal

from .embeddings.providers.embedding_gemma import EmbeddingGemmaProvider, TaskType
from .vector_store import VectorStore
from .document_store import DocumentStore


class SearchResult:
    """Represents a search result with context"""
    
    def __init__(
        self,
        chunk_id: str,
        chunk_content: str,
        similarity_score: float,
        document_id: str,
        document_filename: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Dict[str, Any]
    ):
        self.chunk_id = chunk_id
        self.chunk_content = chunk_content
        self.similarity_score = similarity_score
        self.document_id = document_id
        self.document_filename = document_filename
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.chunk_content,
            'similarity_score': self.similarity_score,
            'document': {
                'id': self.document_id,
                'filename': self.document_filename,
            },
            'position': {
                'chunk_index': self.chunk_index,
                'start_char': self.start_char,
                'end_char': self.end_char,
            },
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return (
            f"SearchResult(doc='{self.document_filename}', "
            f"chunk={self.chunk_index}, score={self.similarity_score:.3f})"
        )


class Searcher:
    """
    Handles search queries and retrieval with task-specific embeddings
    """

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingGemmaProvider
    ):
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        task: TaskType = "retrieval"
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using task-specific query embedding

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            task: Task type for query embedding (retrieval, fact_checking, code_retrieval, etc.)

        Returns:
            List of SearchResult objects
        """
        if self.vector_store.count() == 0:
            print("No documents in the index")
            return []

        # Generate task-specific query embedding
        query_vector = self.embedding_provider.embed_query(query, task=task)
        
        # Search vector store
        raw_results = self.vector_store.search(query_vector, k=top_k)
        
        # Enrich with document context
        results = []
        for chunk_id, score in raw_results:
            # Apply score threshold if specified
            if min_score is not None and score < min_score:
                continue
            
            # Get chunk details
            chunk = self.document_store.get_chunk(chunk_id)
            if chunk is None:
                continue
            
            # Get document details
            document = self.document_store.get_document(chunk['document_id'])
            if document is None:
                continue
            
            result = SearchResult(
                chunk_id=chunk_id,
                chunk_content=chunk['content'],
                similarity_score=score,
                document_id=document['id'],
                document_filename=document['filename'],
                chunk_index=chunk['chunk_index'],
                start_char=chunk['start_char'],
                end_char=chunk['end_char'],
                metadata=chunk['metadata']
            )
            results.append(result)
        
        return results
    
    def get_document_context(
        self,
        doc_id: str,
        chunk_index: int,
        context_chunks: int = 1
    ) -> str:
        """
        Get a chunk with surrounding context
        
        Args:
            doc_id: Document ID
            chunk_index: Index of target chunk
            context_chunks: Number of chunks to include before/after
            
        Returns:
            Combined text with context
        """
        chunks = self.document_store.get_chunks_by_document(doc_id)
        
        # Get range of chunks to include
        start_idx = max(0, chunk_index - context_chunks)
        end_idx = min(len(chunks), chunk_index + context_chunks + 1)
        
        context_texts = []
        for i in range(start_idx, end_idx):
            if i < len(chunks):
                context_texts.append(chunks[i]['content'])
        
        return "\n\n".join(context_texts)
