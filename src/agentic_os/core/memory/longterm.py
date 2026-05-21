"""Long-term memory backed by a knowledge graph for persistent agent memory."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import (
    MemoryNode,
    create_episode,
    create_fact,
    create_goal,
    create_reflection,
)
from agentic_os.core.graph.scoring import get_most_important
from agentic_os.core.graph.traversal import bfs, shortest_path
from agentic_os.core.vector.base import VectorStore

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enum defining supported retrieval strategies for long-term memory.

    Attributes:
        RECENCY: Return the most recently added memories.
        IMPORTANCE: Return memories with the highest importance scores.
        ASSOCIATION: Traverse the graph outward from a seed node.
        KEYWORD: Match memories by keyword against the inverted index.
    """

    RECENCY = "recency"
    IMPORTANCE = "importance"
    ASSOCIATION = "association"
    KEYWORD = "keyword"


class LongTermMemory:
    """Persistent long-term memory layer backed by a KnowledgeGraph.

    Supports multiple retrieval strategies (keyword, importance, recency,
    association) and provides convenience helpers for storing episodes,
    facts, reflections, and goals.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        """Initialize with an empty knowledge graph and timeline.

        Args:
            vector_store: Optional vector store for similarity search.
                When provided together with *embedder*, enables
                :meth:`retrieve_similar` and :meth:`index_all`.
            embedder: Optional callable that converts text to embedding vectors.
        """
        self._graph = KnowledgeGraph()
        self._timeline: list[str] = []  # chronologically ordered node IDs
        self._vector_store = vector_store
        self._embedder = embedder

    @property
    def graph(self) -> KnowledgeGraph:
        """KnowledgeGraph: The underlying knowledge graph instance."""
        return self._graph

    def store(self, node: MemoryNode) -> str:
        """Store a memory node in the knowledge graph.

        Args:
            node: A ``MemoryNode`` instance to persist.

        Returns:
            The node ID.
        """
        self._graph.add_node(node)
        self._timeline.append(node.id)
        logger.debug("store: node %s", node.id)
        return node.id

    def store_episode(self, content: str, **metadata: Any) -> str:
        """Create and store an episode node.

        Args:
            content: Textual description of the episode.
            **metadata: Additional metadata attached to the node.

        Returns:
            The new node ID.
        """
        node_id = self.store(create_episode(content, **metadata))
        logger.debug("store_episode: %s", node_id)
        return node_id

    def store_fact(self, content: str, importance: float = 0.6, **metadata: Any) -> str:
        """Create and store a fact node.

        Args:
            content: Fact text.
            importance: Initial importance score in [0, 1].
            **metadata: Additional metadata.

        Returns:
            The new node ID.
        """
        node_id = self.store(create_fact(content, importance, **metadata))
        logger.debug("store_fact: %s", node_id)
        return node_id

    def store_reflection(self, content: str, importance: float = 0.7, **metadata: Any) -> str:
        """Create and store a reflection node.

        Args:
            content: Reflection text.
            importance: Initial importance score in [0, 1].
            **metadata: Additional metadata.

        Returns:
            The new node ID.
        """
        node_id = self.store(create_reflection(content, importance, **metadata))
        logger.debug("store_reflection: %s", node_id)
        return node_id

    def store_goal(self, content: str, importance: float = 0.8, **metadata: Any) -> str:
        """Create and store a goal node.

        Args:
            content: Goal description text.
            importance: Initial importance score in [0, 1].
            **metadata: Additional metadata.

        Returns:
            The new node ID.
        """
        return self.store(create_goal(content, importance, **metadata))

    def link(self, source_id: str, target_id: str,
             edge_type: EdgeType = EdgeType.ASSOCIATIVE, weight: float = 1.0) -> None:
        """Create an edge between two memory nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Semantic type of the edge. Defaults to ASSOCIATIVE.
            weight: Edge weight. Defaults to 1.0.
        """
        self._graph.add_edge(source_id, target_id, edge_type, weight)

    def retrieve(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.KEYWORD,
                 top_k: int = 10, seed_id: str | None = None) -> list[MemoryNode]:
        """Retrieve memories using the specified strategy.

        Args:
            query: Search text (used by KEYWORD strategy).
            strategy: Retrieval strategy to use.
            top_k: Maximum number of results to return.
            seed_id: Starting node ID for the ASSOCIATION strategy.

        Returns:
            List of matching ``MemoryNode`` instances, up to *top_k*.

        Examples:
            >>> ltm = LongTermMemory()
            >>> ltm.store_fact("Paris is the capital of France")
            >>> ltm.retrieve("Paris", strategy=RetrievalStrategy.KEYWORD, top_k=5)
        """
        logger.debug("retrieve: strategy=%s query=%r top_k=%d", strategy.value, query[:40], top_k)
        if strategy == RetrievalStrategy.KEYWORD:
            return self._graph.find_nodes(query)[:top_k]
        elif strategy == RetrievalStrategy.IMPORTANCE:
            important = get_most_important(self._graph, top_k)
            nodes = []
            for nid, _ in important:
                node = self._graph.get_node(nid)
                if node:
                    nodes.append(node)
            return nodes
        elif strategy == RetrievalStrategy.RECENCY:
            recent_ids = self._timeline[-top_k:]
            nodes = []
            for nid in reversed(recent_ids):
                node = self._graph.get_node(nid)
                if node:
                    nodes.append(node)
            return nodes
        elif strategy == RetrievalStrategy.ASSOCIATION:
            if seed_id is None:
                return []
            layers = bfs(self._graph, seed_id, max_depth=2)
            all_ids = []
            for layer_ids in layers.values():
                all_ids.extend(layer_ids)
            nodes = []
            for nid in all_ids[:top_k]:
                node = self._graph.get_node(nid)
                if node:
                    nodes.append(node)
            return nodes
        return []

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a single node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            The ``MemoryNode`` if found, else ``None``.
        """
        return self._graph.get_node(node_id)

    def find_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Find the shortest path between two nodes.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.

        Returns:
            Ordered list of node IDs along the path, or ``None`` if no path.
        """
        return shortest_path(self._graph, from_id, to_id)

    def recompute_importance(self) -> None:
        """Re-run PageRank to update all node importance scores."""
        from agentic_os.core.graph.scoring import recompute_importance
        recompute_importance(self._graph)

    def stats(self) -> dict[str, Any]:
        """Return graph statistics (node count, edge count, etc.).

        Returns:
            Dict of statistical key-value pairs.
        """
        return self._graph.stats()

    def retrieve_similar(self, query: str, top_k: int = 10) -> list[MemoryNode]:
        """Retrieve memory nodes whose content is semantically similar to *query*.

        Requires both ``vector_store`` and ``embedder`` to have been provided
        at construction time.  Falls back to an empty list when either is
        missing.

        Args:
            query: Text to search for.
            top_k: Maximum number of results.

        Returns:
            List of :class:`~agentic_os.core.graph.node.MemoryNode` sorted by
            descending similarity.

        Raises:
            RuntimeError: If ``vector_store`` or ``embedder`` is not configured.
        """
        if self._vector_store is None or self._embedder is None:
            raise RuntimeError(
                "retrieve_similar requires both vector_store and embedder "
                "to be provided at construction time."
            )
        query_vec = self._embedder(query)
        hits = self._vector_store.search(query_vec, top_k=top_k)
        nodes: list[MemoryNode] = []
        for hit in hits:
            node = self._graph.get_node(hit.id)
            if node is not None:
                nodes.append(node)
        logger.debug("retrieve_similar: query=%r top_k=%d returned=%d", query[:40], top_k, len(nodes))
        return nodes

    def index_all(self) -> int:
        """Build vector indices for all existing memory nodes.

        Requires both ``vector_store`` and ``embedder`` to have been provided
        at construction time.

        Returns:
            The number of nodes indexed.

        Raises:
            RuntimeError: If ``vector_store`` or ``embedder`` is not configured.
        """
        if self._vector_store is None or self._embedder is None:
            raise RuntimeError(
                "index_all requires both vector_store and embedder "
                "to be provided at construction time."
            )
        count = 0
        for node in self._graph.nodes:
            vec = self._embedder(node.content)
            self._vector_store.add(node.id, vec, {"content": node.content, "type": node.type.value})
            count += 1
        logger.debug("index_all: indexed %d nodes", count)
        return count
