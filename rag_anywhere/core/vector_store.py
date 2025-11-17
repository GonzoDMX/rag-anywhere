# rag_anywhere/core/vector_store.py
class HybridVectorStore:
    """FAISS for search + SQLite for persistence"""
    
    def __init__(self, db_path: str, dimension: int = 768):
        self.db_path = db_path
        self.dimension = dimension
        self.index = None  # FAISS index (in-memory)
        self.id_map = {}   # Maps FAISS IDs to chunk IDs
        self._load_or_create()
    
    def _load_or_create(self):
        """Load vectors from SQLite into FAISS on startup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT id, vector FROM chunk_vectors")
        
        vectors = []
        ids = []
        for chunk_id, vector_blob in cursor:
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            vectors.append(vector)
            ids.append(chunk_id)
        
        if vectors:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            self.index.add(np.array(vectors))
            self.id_map = {i: chunk_id for i, chunk_id in enumerate(ids)}
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def add(self, chunk_id: str, vector: np.ndarray):
        """Add vector to both FAISS and SQLite"""
        # Add to FAISS
        faiss_id = len(self.id_map)
        self.index.add(np.array([vector]))
        self.id_map[faiss_id] = chunk_id
        
        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO chunk_vectors (id, vector) VALUES (?, ?)",
            (chunk_id, vector.tobytes())
        )
        conn.commit()
    
    def search(self, query_vector: np.ndarray, k: int = 5):
        """Search FAISS index"""
        distances, indices = self.index.search(np.array([query_vector]), k)
        results = [
            (self.id_map[idx], float(dist)) 
            for dist, idx in zip(distances[0], indices[0])
            if idx in self.id_map
        ]
        return results
    
    def delete(self, chunk_ids: List[str]):
        """Remove from SQLite and rebuild FAISS"""
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(chunk_ids))
        conn.execute(f"DELETE FROM chunk_vectors WHERE id IN ({placeholders})", chunk_ids)
        conn.commit()
        
        # Rebuild FAISS index
        self._load_or_create()
