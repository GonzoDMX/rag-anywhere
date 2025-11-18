# rag_anywhere/core/document_store.py

import sqlite3
import json
import uuid
from typing import List, Optional, Dict, Any

from .splitters import TextChunk


class DocumentStore:
    """
    SQLite-based document and chunk storage.
    Manages documents, their chunks, and metadata.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        # Chunk vectors table (for persistence)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)
        
        # Config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)")
        
        conn.commit()
        conn.close()
    
    def add_document(
        self, 
        filename: str, 
        content: str, 
        chunks: List[TextChunk],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document and its chunks to the store
        
        Args:
            filename: Document filename
            content: Full document content
            chunks: List of text chunks
            metadata: Optional document metadata
            
        Returns:
            Document ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Insert document
        cursor.execute(
            """
            INSERT INTO documents (id, filename, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (doc_id, filename, content, json.dumps(metadata or {}))
        )
        
        # Insert chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{idx}"
            chunk_metadata = chunk.metadata or {}
            chunk_metadata.update({
                'document_id': doc_id,
                'chunk_index': idx
            })
            
            cursor.execute(
                """
                INSERT INTO chunks (id, document_id, chunk_index, content, start_char, end_char, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    doc_id,
                    idx,
                    chunk.content,
                    chunk.start_char,
                    chunk.end_char,
                    json.dumps(chunk_metadata)
                )
            )
        
        conn.commit()
        conn.close()
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row['id'],
            'filename': row['filename'],
            'content': row['content'],
            'metadata': json.loads(row['metadata']),
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document by filename"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM documents WHERE filename = ?",
            (filename,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row['id'],
            'filename': row['filename'],
            'content': row['content'],
            'metadata': json.loads(row['metadata']),
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, filename, metadata, created_at FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row['id'],
                'filename': row['filename'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at']
            }
            for row in rows
        ]
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document and its chunks
        
        Returns:
            True if document was deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if document exists
        cursor.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
        if cursor.fetchone() is None:
            conn.close()
            return False
        
        # Delete document (cascades to chunks and vectors)
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        return True
    
    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row['id'],
                'document_id': row['document_id'],
                'chunk_index': row['chunk_index'],
                'content': row['content'],
                'start_char': row['start_char'],
                'end_char': row['end_char'],
                'metadata': json.loads(row['metadata'])
            }
            for row in rows
        ]
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row['id'],
            'document_id': row['document_id'],
            'chunk_index': row['chunk_index'],
            'content': row['content'],
            'start_char': row['start_char'],
            'end_char': row['end_char'],
            'metadata': json.loads(row['metadata'])
        }
    
    def get_all_chunk_ids(self) -> List[str]:
        """Get all chunk IDs in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chunks")
        chunk_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return chunk_ids
    
    def get_config(self, key: str) -> Optional[str]:
        """Get configuration value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    def set_config(self, key: str, value: str):
        """Set configuration value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()
        conn.close()
