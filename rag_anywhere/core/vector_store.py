# rag_anywhere/core/vector_store.py
import sqlite3
import numpy as np
import faiss
from typing import List, Tuple, Optional


class VectorStore:
    """
    Hybrid vector store: FAISS for fast search + SQLite for persistence
    """
    
    def __init__(self, db_path: str, dimension: int = 768):
        self.db_path = db_path
        self.dimension = dimension
        self.index = None
        self.id_map = {}  # Maps FAISS index positions to chunk IDs
        self._load_or_create()
    
    def _load_or_create(self):
        """Load vectors from SQLite into FAISS on startup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all vectors from database
        cursor.execute("SELECT chunk_id, vector FROM chunk_vectors")
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            # Load existing vectors into FAISS
            vectors = []
            chunk_ids = []
            
            for chunk_id, vector_blob in rows:
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                vectors.append(vector)
                chunk_ids.append(chunk_id)
            
            # Create FAISS index with inner product (for cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            vectors_array = np.array(vectors, dtype=np.float32)
            self.index.add(vectors_array)
            
            # Build ID mapping
            self.id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
            print(f"Loaded {len(vectors)} vectors into FAISS index")
        else:
            # Create empty index
            self.index = faiss.IndexFlatIP(self.dimension)
            print("Created new empty FAISS index")
    
    def add(self, chunk_id: str, vector: np.ndarray):
        """
        Add a single vector to both FAISS and SQLite
        
        Args:
            chunk_id: Unique chunk identifier
            vector: Embedding vector (must be 768-dimensional and normalized)
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
        
        # Ensure vector is normalized for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Add to FAISS
        faiss_id = len(self.id_map)
        self.index.add(np.array([vector], dtype=np.float32))
        self.id_map[faiss_id] = chunk_id
        
        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO chunk_vectors (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, vector.astype(np.float32).tobytes())
        )
        conn.commit()
        conn.close()
    
    def add_batch(self, chunk_ids: List[str], vectors: np.ndarray):
        """
        Add multiple vectors at once (more efficient)
        
        Args:
            chunk_ids: List of chunk identifiers
            vectors: Array of shape (n, 768)
        """
        if len(chunk_ids) != len(vectors):
            raise ValueError("Number of chunk IDs must match number of vectors")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms
        
        # Add to FAISS
        start_id = len(self.id_map)
        self.index.add(vectors.astype(np.float32))
        
        # Update ID mapping
        for i, chunk_id in enumerate(chunk_ids):
            self.id_map[start_id + i] = chunk_id
        
        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk_id, vector in zip(chunk_ids, vectors):
            cursor.execute(
                "INSERT OR REPLACE INTO chunk_vectors (chunk_id, vector) VALUES (?, ?)",
                (chunk_id, vector.astype(np.float32).tobytes())
            )
        
        conn.commit()
        conn.close()
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding (768-dimensional)
            k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
        """
        if self.index.ntotal == 0:
            return []
        
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vector.shape[0]}")
        
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Search FAISS index
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            k
        )
        
        # Map FAISS indices back to chunk IDs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_map:  # -1 means no result
                chunk_id = self.id_map[idx]
                # Convert inner product back to similarity score (already normalized, so IP = cosine similarity)
                similarity = float(dist)
                results.append((chunk_id, similarity))
        
        return results
    
    def delete(self, chunk_ids: List[str]):
        """
        Delete vectors from store
        Note: This requires rebuilding the FAISS index
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        # Delete from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.commit()
        conn.close()
        
        # Rebuild FAISS index from SQLite
        self._load_or_create()
    
    def count(self) -> int:
        """Get total number of vectors in the index"""
        return self.index.ntotal
