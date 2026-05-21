"""SQLite storage backend for knowledge graph persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from agentic_os.core.graph.edge import Edge, EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import MemoryNode, NodeType
from agentic_os.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class SqliteStorage(StorageBackend):
    """SQLite-backed persistent storage for knowledge graphs.

    Uses three tables:
        - ``nodes``: node data with JSON-serialized content and metadata
        - ``edges``: directed edges between nodes
        - ``metadata``: key-value store for graph-level metadata

    Thread-safe via an internal ``threading.Lock``.

    Args:
        path: Path to the SQLite database file. Use ``":memory:"`` for
            in-memory databases (useful for testing).

    Example::

        with SqliteStorage("agent_memory.db") as store:
            store.save_graph(kg)
            loaded = store.load_graph()
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()
        logger.info("SqliteStorage opened: %s", path)

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    keywords_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (source_id, target_id)
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            self._conn.commit()

    def save_graph(self, graph: KnowledgeGraph) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM nodes")
            self._conn.execute("DELETE FROM edges")
            for node in graph.nodes:
                self._upsert_node(node)
            for sid in graph._out_edges:
                for edge in graph._out_edges[sid].values():
                    self._upsert_edge(edge)
            self._conn.commit()
        logger.info("save_graph: saved %d nodes", graph.node_count)

    def load_graph(self) -> KnowledgeGraph | None:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
            if row is None or row[0] == 0:
                return None

        kg = KnowledgeGraph()
        with self._lock:
            for row in self._conn.execute("SELECT * FROM nodes"):
                node = self._row_to_node(row)
                kg._nodes[node.id] = node
                kg._type_index[node.type].add(node.id)
                for kw in kg._tokenize(node.content):
                    kg._keyword_index[kw].add(node.id)

            for row in self._conn.execute("SELECT * FROM edges"):
                edge = self._row_to_edge(row)
                kg._out_edges[edge.source_id][edge.target_id] = edge
                kg._in_edges[edge.target_id][edge.source_id] = edge

        logger.info("load_graph: loaded %d nodes", kg.node_count)
        kg._sync_edge_count()
        return kg

    def save_node(self, node: MemoryNode) -> None:
        with self._lock:
            self._upsert_node(node)
            self._conn.commit()
        logger.debug("save_node: %s", node.id)

    def load_node(self, node_id: str) -> MemoryNode | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def delete_node(self, node_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                               (node_id, node_id))
            self._conn.commit()
            return cursor.rowcount > 0

    def query_nodes(self, filter_fn: Callable[[str, dict[str, Any]], bool]) -> list[MemoryNode]:
        results: list[MemoryNode] = []
        with self._lock:
            for row in self._conn.execute("SELECT * FROM nodes"):
                node = self._row_to_node(row)
                value_dict: dict[str, Any] = {
                    "type": node.type.value,
                    "content": node.content,
                    "importance": node.importance,
                    "access_count": node.access_count,
                    "metadata": node.metadata,
                }
                if filter_fn(node.id, value_dict):
                    results.append(node)
        return results

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("SqliteStorage closed: %s", self._path)

    def __enter__(self) -> SqliteStorage:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Internal helpers ──

    def _upsert_node(self, node: MemoryNode) -> None:
        keywords = list(KnowledgeGraph._tokenize(node.content))
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, type, content, importance, access_count, created_at, updated_at,
                metadata_json, keywords_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.id, node.type.value, node.content, node.importance,
             node.access_count, node.created_at, node.updated_at,
             json.dumps(node.metadata), json.dumps(keywords)),
        )

    def _upsert_edge(self, edge: Edge) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO edges (source_id, target_id, type, weight, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (edge.source_id, edge.target_id, edge.type.value, edge.weight,
             json.dumps(edge.metadata)),
        )

    @staticmethod
    def _row_to_node(row: tuple[Any, ...]) -> MemoryNode:
        return MemoryNode(
            id=row[0],
            type=NodeType(row[1]),
            content=row[2],
            importance=row[3],
            access_count=row[4],
            created_at=row[5],
            updated_at=row[6],
            metadata=json.loads(row[7]),
        )

    @staticmethod
    def _row_to_edge(row: tuple[Any, ...]) -> Edge:
        return Edge(
            source_id=row[0],
            target_id=row[1],
            type=EdgeType(row[2]),
            weight=row[3],
            metadata=json.loads(row[4]),
        )
