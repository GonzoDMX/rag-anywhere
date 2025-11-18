# rag_anywhere/core/vector_store.py
import sqlite3
from typing import Dict, List, Tuple

import faiss
import numpy as np

from ..utils.logging import get_logger

logger = get_logger('core.vector_store')


class VectorStore:
    """
    Hybrid vector store: FAISS for fast search + SQLite for persistence
    """
    
    def __init__(self, db_path: str, dimension: int = 768):
        logger.info(f"Initializing VectorStore with dimension={dimension}")
        logger.debug(f"Database path: {db_path}")
        logger.debug(f"FAISS version: {faiss.__version__}")

        self.db_path = db_path
        self.dimension = dimension

        try:
            # FAISS index for inner-product similarity search. Always initialized
            # to a valid (possibly empty) index so type checkers know it's not None.
            logger.debug(f"Creating FAISS IndexFlatIP with dimension {self.dimension}")
            self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
            logger.debug("FAISS index created successfully")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {type(e).__name__}: {e}", exc_info=True)
            raise

        # Maps FAISS index positions to chunk IDs
        self.id_map: Dict[int, str] = {}

        self._load_or_create()
    
    def _load_or_create(self):
        """Load vectors from SQLite into FAISS on startup"""
        logger.debug("Loading vectors from SQLite into FAISS")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all vectors from database
            logger.debug("Querying chunk_vectors table")
            cursor.execute("SELECT chunk_id, vector FROM chunk_vectors")
            rows = cursor.fetchall()
            conn.close()

            logger.debug(f"Found {len(rows)} vectors in database")

            # Reset index and mapping to ensure we are always in a consistent state
            logger.debug("Resetting FAISS index and ID mapping")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_map = {}

            if rows:
                # Load existing vectors into FAISS
                logger.info(f"Loading {len(rows)} vectors into FAISS index...")
                vectors = []
                chunk_ids = []

                for chunk_id, vector_blob in rows:
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    vectors.append(vector)
                    chunk_ids.append(chunk_id)

                if vectors:
                    # FAISS' Python bindings expose multiple index types; IndexFlatIP
                    # in this project expects a 2D float32 array of shape (n, d) as
                    # the first positional argument.
                    vectors_array = np.asarray(vectors, dtype=np.float32)
                    logger.debug(f"Vectors array shape: {vectors_array.shape}")

                    # Guard against any shape mismatch at runtime
                    if vectors_array.ndim != 2 or vectors_array.shape[1] != self.dimension:
                        error_msg = f"Stored vectors have wrong shape {vectors_array.shape}, expected (*, {self.dimension})"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Populate FAISS index and ID mapping. The IndexFlatIP.add
                    # binding takes the 2D float32 array as its single argument.
                    # FAISS stubs show C-style API, but runtime uses Pythonic API (ignore)
                    logger.debug("Adding vectors to FAISS index")
                    self.index.add(vectors_array.astype(np.float32))  # type: ignore[call-arg]
                    self.id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
                    logger.info(f"Successfully loaded {len(vectors)} vectors into FAISS index")

                print(f"Loaded {len(vectors)} vectors into FAISS index")
            else:
                # Already initialized to an empty index above
                logger.info("No existing vectors found, created new empty FAISS index")
                print("Created new empty FAISS index")

        except Exception as e:
            logger.error(f"Failed to load vectors: {type(e).__name__}: {e}", exc_info=True)
            raise
    
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
        to_add = np.array([vector], dtype=np.float32, copy=False)
        # FAISS stubs show C-style API, but runtime uses Pythonic API (ignore)
        self.index.add(to_add)  # type: ignore[call-arg]
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
        to_add_batch = vectors.astype(np.float32, copy=False)
        # FAISS stubs show C-style API, but runtime uses Pythonic API (ignore)
        self.index.add(to_add_batch)  # type: ignore[call-arg]
        
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
        # FAISS stubs show C-style API, but runtime uses Pythonic API (ignore)
        distances, indices = self.index.search(  # type: ignore[call-arg]
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
