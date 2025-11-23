"""Entity storage and knowledge graph operations.

This module manages the knowledge graph layer on top of the document/chunk storage.
It stores entities (nodes) and their relationships to chunks (edges).
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from .gliner.models import Entity

logger = logging.getLogger(__name__)


class EntityStore:
    """Manages entity and knowledge graph storage in SQLite."""

    def __init__(self, db_path: str):
        """
        Initialize entity store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize database schema for knowledge graph."""
        cursor = self.conn.cursor()

        # Graph nodes table - stores unique entities
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                category TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                is_public INTEGER DEFAULT 1,
                UNIQUE(name, category)
            )
        """
        )

        # Chunk edges table - links chunks to entities
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_edges (
                chunk_id TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT DEFAULT 'gliner',
                PRIMARY KEY (chunk_id, node_id),
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                FOREIGN KEY(node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
            )
        """
        )

        # Create indices for fast graph traversal
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_node
            ON chunk_edges(node_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_chunk
            ON chunk_edges(chunk_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_cat
            ON graph_nodes(category)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_name
            ON graph_nodes(name)
        """
        )

        self.conn.commit()
        logger.info("Entity store database initialized")

    def add_entities(
        self, chunk_id: str, entities: List[Entity], source: str = "gliner"
    ) -> int:
        """
        Add entities for a chunk and create edges.

        This method:
        1. Inserts or updates nodes in graph_nodes
        2. Creates edges linking chunk_id to node_ids
        3. Updates frequency counts

        Args:
            chunk_id: Chunk identifier (format: doc_id_chunk_index)
            entities: List of Entity objects extracted from chunk
            source: Source of entities ('gliner', 'user', etc.)

        Returns:
            Number of entities added
        """
        if not entities:
            return 0

        cursor = self.conn.cursor()

        for entity in entities:
            # Normalize name for deduplication (lowercase, stripped)
            normalized_name = entity.text.strip().lower()
            display_name = entity.text.strip()

            # Insert or update node
            cursor.execute(
                """
                INSERT INTO graph_nodes (name, display_name, category, frequency)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(name, category) DO UPDATE SET
                    frequency = frequency + 1,
                    display_name = CASE
                        WHEN length(excluded.display_name) > length(display_name)
                        THEN excluded.display_name
                        ELSE display_name
                    END
            """,
                (normalized_name, display_name, entity.label),
            )

            # Get node_id
            cursor.execute(
                """
                SELECT id FROM graph_nodes
                WHERE name = ? AND category = ?
            """,
                (normalized_name, entity.label),
            )
            node_id = cursor.fetchone()[0]

            # Create edge (insert or replace to handle duplicates)
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunk_edges (chunk_id, node_id, weight, source)
                VALUES (?, ?, ?, ?)
            """,
                (chunk_id, node_id, entity.score, source),
            )

        self.conn.commit()
        return len(entities)

    def get_chunk_entities(self, chunk_id: str) -> List[Dict]:
        """
        Get all entities for a specific chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            List of entity dicts with node info and edge weight
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                n.id, n.name, n.display_name, n.category,
                n.frequency, e.weight, e.source
            FROM chunk_edges e
            JOIN graph_nodes n ON e.node_id = n.id
            WHERE e.chunk_id = ?
            ORDER BY e.weight DESC
        """,
            (chunk_id,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_entity_chunks(self, entity_name: str, category: Optional[str] = None) -> List[str]:
        """
        Get all chunks that mention a specific entity.

        Args:
            entity_name: Entity name (case-insensitive)
            category: Optional entity category filter

        Returns:
            List of chunk IDs
        """
        cursor = self.conn.cursor()
        normalized_name = entity_name.strip().lower()

        if category:
            cursor.execute(
                """
                SELECT DISTINCT e.chunk_id
                FROM chunk_edges e
                JOIN graph_nodes n ON e.node_id = n.id
                WHERE n.name = ? AND n.category = ?
            """,
                (normalized_name, category),
            )
        else:
            cursor.execute(
                """
                SELECT DISTINCT e.chunk_id
                FROM chunk_edges e
                JOIN graph_nodes n ON e.node_id = n.id
                WHERE n.name = ?
            """,
                (normalized_name,),
            )

        return [row[0] for row in cursor.fetchall()]

    def query_entities(
        self,
        category: Optional[str] = None,
        min_frequency: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Query entities with optional filters.

        Args:
            category: Filter by entity category
            min_frequency: Minimum frequency (number of mentions)
            limit: Maximum number of results

        Returns:
            List of entity dicts
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM graph_nodes WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if min_frequency is not None:
            query += " AND frequency >= ?"
            params.append(min_frequency)

        query += " ORDER BY frequency DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_entity_by_id(self, entity_id: int) -> Optional[Dict]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity node ID

        Returns:
            Entity dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM graph_nodes WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_entity_by_name(
        self, name: str, category: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get entity by name and optional category.

        Args:
            name: Entity name (case-insensitive)
            category: Optional category filter

        Returns:
            Entity dict or None if not found
        """
        cursor = self.conn.cursor()
        normalized_name = name.strip().lower()

        if category:
            cursor.execute(
                "SELECT * FROM graph_nodes WHERE name = ? AND category = ?",
                (normalized_name, category),
            )
        else:
            cursor.execute(
                "SELECT * FROM graph_nodes WHERE name = ?", (normalized_name,)
            )

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_related_entities(
        self, entity_id: int, limit: Optional[int] = None
    ) -> List[Tuple[Dict, int]]:
        """
        Get entities that co-occur with the given entity (share chunks).

        Args:
            entity_id: Source entity ID
            limit: Maximum number of related entities

        Returns:
            List of tuples (entity_dict, co_occurrence_count)
        """
        cursor = self.conn.cursor()

        # Find entities that appear in the same chunks
        query = """
            SELECT
                n.*,
                COUNT(DISTINCT e2.chunk_id) as co_occurrence_count
            FROM chunk_edges e1
            JOIN chunk_edges e2 ON e1.chunk_id = e2.chunk_id
            JOIN graph_nodes n ON e2.node_id = n.id
            WHERE e1.node_id = ? AND e2.node_id != ?
            GROUP BY n.id
            ORDER BY co_occurrence_count DESC
        """

        params = [entity_id, entity_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [(dict(row), row["co_occurrence_count"]) for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """
        Get knowledge graph statistics.

        Returns:
            Dict with entity counts, category breakdown, etc.
        """
        cursor = self.conn.cursor()

        # Total entities
        cursor.execute("SELECT COUNT(*) as count FROM graph_nodes")
        total_entities = cursor.fetchone()["count"]

        # Total edges
        cursor.execute("SELECT COUNT(*) as count FROM chunk_edges")
        total_edges = cursor.fetchone()["count"]

        # Entities by category
        cursor.execute(
            """
            SELECT category, COUNT(*) as count
            FROM graph_nodes
            GROUP BY category
            ORDER BY count DESC
        """
        )
        by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

        # Top entities
        cursor.execute(
            """
            SELECT display_name, category, frequency
            FROM graph_nodes
            ORDER BY frequency DESC
            LIMIT 10
        """
        )
        top_entities = [dict(row) for row in cursor.fetchall()]

        return {
            "total_entities": total_entities,
            "total_edges": total_edges,
            "by_category": by_category,
            "top_entities": top_entities,
        }

    def delete_chunk_entities(self, chunk_id: str) -> int:
        """
        Delete all entities for a chunk (cleanup).

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of edges deleted
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunk_edges WHERE chunk_id = ?", (chunk_id,))
        deleted = cursor.rowcount
        self.conn.commit()

        # Optionally clean up orphaned nodes (nodes with no edges)
        cursor.execute(
            """
            DELETE FROM graph_nodes
            WHERE id NOT IN (SELECT DISTINCT node_id FROM chunk_edges)
        """
        )
        self.conn.commit()

        return deleted

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Entity store connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
