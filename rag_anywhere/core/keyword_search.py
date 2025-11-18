# rag_anywhere/core/keyword_search.py
import sqlite3
import re
from typing import List, Tuple, Optional


class KeywordSearcher:
    """
    Keyword-based search using SQLite FTS5 (Full-Text Search).

    Features:
    - Fast inverted index with BM25 ranking
    - Porter stemming for better matching
    - Boolean operators (AND, OR, NOT)
    - Phrase queries
    - Prefix matching
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_fts()

    @staticmethod
    def _escape_fts5_special_chars(text: str) -> str:
        """
        Sanitize special characters for FTS5 queries.

        Strategy: Remove or replace problematic punctuation to avoid FTS5 syntax errors
        without splitting words.
        """
        # Replace double quotes with spaces (to avoid "phrase" syntax)
        text = text.replace('"', ' ')
        
        # Remove apostrophes/single quotes *entirely*, not with a space
        text = text.replace("'", '')   

        text = text.replace('\\', ' ')  # Remove backslashes
        text = text.replace('/', ' ')   # Remove slashes
        text = text.replace('(', ' ')   # Remove opening parenthesis
        text = text.replace(')', ' ')   # Remove closing parenthesis

        # Collapse multiple spaces into single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @staticmethod
    def _make_exact_match_query(query: str) -> str:
        """
        Convert a query to exact match (phrase query) format.

        Args:
            query: The search query

        Returns:
            FTS5 phrase query format
        """
        # Remove any existing quotes
        query = query.replace('"', '')

        # Wrap in quotes for exact phrase matching
        return f'"{query}"'

    def _init_fts(self):
        """Create FTS5 virtual table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create FTS5 table with porter stemming and unicode support
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                metadata UNINDEXED,
                tokenize='porter unicode61 remove_diacritics 1'
            )
        """)

        conn.commit()
        conn.close()

    def index_chunk(self, chunk_id: str, content: str, metadata: str = ""):
        """
        Add a chunk to the FTS index

        Args:
            chunk_id: Unique chunk identifier
            content: Text content to index
            metadata: Optional metadata (as JSON string)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO chunks_fts (chunk_id, content, metadata) VALUES (?, ?, ?)",
            (chunk_id, content, metadata)
        )

        conn.commit()
        conn.close()

    def index_chunks_batch(self, chunks: List[Tuple[str, str, str]]):
        """
        Add multiple chunks to FTS index efficiently

        Args:
            chunks: List of (chunk_id, content, metadata) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany(
            "INSERT OR REPLACE INTO chunks_fts (chunk_id, content, metadata) VALUES (?, ?, ?)",
            chunks
        )

        conn.commit()
        conn.close()

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_terms: Optional[List[str]] = None,
        exact_match: bool = False,
        escape_special_chars: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search using FTS5 with BM25 ranking

        Args:
            query: Search query. Supports:
                - Simple: "machine learning"
                - Phrase: '"machine learning"'
                - Boolean: "machine AND learning"
                - NOT: "machine NOT cat"
                - Prefix: "mach*"
            top_k: Number of results to return
            exclude_terms: Terms to exclude from results
            exact_match: If True, treat query as exact phrase match
            escape_special_chars: If True, escape special FTS5 characters

        Returns:
            List of (chunk_id, score) tuples, sorted by relevance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if exact_match:
            # 1. Handle exact match FIRST.
            # This converts the query to a phrase, escaping internal quotes.
            fts_query = f'"{query.replace('"', '""')}"'
        
        elif escape_special_chars:
            # 2. Handle standard queries
            # Sanitize the *entire* query string.
            fts_query = self._escape_fts5_special_chars(query)
        
        else:
            # 3. No escaping (e.g., from search_with_keywords). Pass raw query.
            fts_query = query
        
        # Add exclusions
        if exclude_terms:
            for term in exclude_terms:
                # Escape exclude terms as well *if* the main query was escaped
                if escape_special_chars:
                    term = self._escape_fts5_special_chars(term)
                
                # Append the NOT operator
                if fts_query.strip():
                    fts_query += f" NOT {term}"
                else:
                    # This would be a query of *only* exclusions, 
                    # which FTS5 doesn't support well.
                    pass

        try:
            # FTS5 query with BM25 ranking (rank is negative, lower is better)
            cursor.execute("""
                SELECT
                    chunk_id,
                    rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, top_k))

            results = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Query syntax error
            conn.close()
            raise ValueError(f"Invalid FTS5 query: {e}")

        conn.close()

        # Convert rank to similarity score (BM25 rank is negative)
        # Normalize to positive scores
        return [(chunk_id, abs(rank)) for chunk_id, rank in results]

    def search_with_keywords(
        self,
        required_keywords: List[str],
        optional_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search with explicit keyword control

        Args:
            required_keywords: All these terms must be present (AND)
            optional_keywords: At least one should be present (OR)
            exclude_keywords: None of these should be present (NOT)
            top_k: Number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        # Build query
        query_parts = []

        # Required terms (AND)
        if required_keywords:
            # Sanitize *each* keyword *before* joining
            sanitized_required = [self._escape_fts5_special_chars(kw) for kw in required_keywords]
            required = " AND ".join(sanitized_required)
            query_parts.append(f"({required})")

        # Optional terms (OR)
        if optional_keywords:
            # Sanitize *each* keyword *before* joining
            sanitized_optional = [self._escape_fts5_special_chars(kw) for kw in optional_keywords]
            optional = " OR ".join(sanitized_optional)
            query_parts.append(f"({optional})")

        # Combine with AND
        query = " AND ".join(query_parts) if query_parts else ""

        if not query:
            return []

        # Sanitize *exclude* terms as well
        sanitized_exclude = None
        if exclude_keywords:
            sanitized_exclude = [self._escape_fts5_special_chars(kw) for kw in exclude_keywords]

        # Use standard search with exclusions
        # We set escape_special_chars=False because we have *already*
        # built a perfectly-formed, sanitized FTS query.
        return self.search(
            query,
            top_k=top_k,
            exclude_terms=sanitized_exclude,
            escape_special_chars=False
        )

    def highlight(self, chunk_id: str, query: str,
                  start_tag: str = "<mark>", end_tag: str = "</mark>") -> Optional[str]:
        """
        Return chunk content with matched terms highlighted

        Args:
            chunk_id: Chunk identifier
            query: Search query used
            start_tag: HTML/marker to start highlight
            end_tag: HTML/marker to end highlight

        Returns:
            Highlighted content, or None if chunk not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(f"""
                SELECT highlight(chunks_fts, 1, '{start_tag}', '{end_tag}')
                FROM chunks_fts
                WHERE chunk_id = ? AND chunks_fts MATCH ?
            """, (chunk_id, query))

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None
        except sqlite3.OperationalError:
            conn.close()
            return None

    def delete_chunk(self, chunk_id: str):
        """Remove a chunk from the FTS index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))

        conn.commit()
        conn.close()

    def delete_chunks_batch(self, chunk_ids: List[str]):
        """Remove multiple chunks from FTS index"""
        if not chunk_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)

        conn.commit()
        conn.close()

    def count(self) -> int:
        """Get total number of indexed chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks_fts")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def rebuild_index(self):
        """
        Rebuild the FTS index from the main chunks table
        Useful if FTS table gets out of sync
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear FTS table
        cursor.execute("DELETE FROM chunks_fts")

        # Repopulate from chunks table
        cursor.execute("""
            INSERT INTO chunks_fts (chunk_id, content, metadata)
            SELECT id, content, json_object() FROM chunks
        """)

        conn.commit()
        conn.close()