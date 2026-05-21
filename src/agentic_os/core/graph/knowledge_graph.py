"""Knowledge graph core -- adjacency list + inverted index implementation."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from agentic_os.core.graph.edge import Edge, EdgeType
from agentic_os.core.graph.node import MemoryNode, NodeType

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Adjacency-list knowledge graph with inverted indexes.

    Internal data structures (all O(1) average access):
        _out_edges:      ``dict[src_id, dict[tgt_id, Edge]]`` -- outgoing edges.
        _in_edges:       ``dict[tgt_id, dict[src_id, Edge]]`` -- incoming edges.
        _nodes:          ``dict[node_id, MemoryNode]`` -- node store.
        _keyword_index:  ``dict[keyword, set[node_id]]`` -- inverted index.
        _type_index:     ``dict[NodeType, set[node_id]]`` -- type filter index.

    Time complexity notes:
        - Node lookup: O(1)
        - Edge lookup: O(1)
        - Keyword search: O(k) where k = matching nodes, falls back to O(V)
          for substring scan.
        - Subgraph extraction: O(V' + E') where V', E' are subgraph size.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._out_edges: dict[str, dict[str, Edge]] = defaultdict(dict)
        self._in_edges: dict[str, dict[str, Edge]] = defaultdict(dict)
        self._keyword_index: dict[str, set[str]] = defaultdict(set)
        self._type_index: dict[NodeType, set[str]] = defaultdict(set)
        self._edge_count: int = 0

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.node_count}, edges={self.edge_count})"

    # ── Node operations ──

    def add_node(self, node: MemoryNode) -> None:
        """Insert a node and update keyword/type indexes.

        Args:
            node: The ``MemoryNode`` to add. If a node with the same ID
                already exists it will be overwritten.

        Example::

            kg.add_node(create_fact("Earth orbits the Sun"))
        """
        logger.debug("add_node id=%s type=%s", node.id, node.type.value)
        old = self._nodes.get(node.id)
        if old is not None:
            self._type_index[old.type].discard(node.id)
            for kw in self._tokenize(old.content):
                self._keyword_index[kw].discard(node.id)
        self._nodes[node.id] = node
        self._type_index[node.type].add(node.id)
        for kw in self._tokenize(node.content):
            self._keyword_index[kw].add(node.id)

    def remove_node(self, node_id: str) -> MemoryNode | None:
        """Remove a node and all its incident edges.

        Args:
            node_id: ID of the node to remove.

        Returns:
            The removed ``MemoryNode``, or ``None`` if not found.

        Example::

            removed = kg.remove_node("fact_abc123")
        """
        node = self._nodes.pop(node_id, None)
        if node is None:
            logger.warning("remove_node node not found: %s", node_id)
            return None
        self._type_index[node.type].discard(node_id)
        for kw in self._tokenize(node.content):
            self._keyword_index[kw].discard(node_id)
        # Remove associated edges
        out_count = len(self._out_edges.get(node_id, {}))
        in_count = len(self._in_edges.get(node_id, {}))
        self._edge_count -= out_count + in_count
        for tgt_id in list(self._out_edges.get(node_id, {})):
            self._in_edges[tgt_id].pop(node_id, None)
        self._out_edges.pop(node_id, None)
        for src_id in list(self._in_edges.get(node_id, {})):
            self._out_edges[src_id].pop(node_id, None)
        self._in_edges.pop(node_id, None)
        logger.debug("remove_node removed id=%s", node_id)
        return node

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Retrieve a node by ID and mark it as accessed (``touch``).

        Args:
            node_id: ID of the node.

        Returns:
            The ``MemoryNode`` if found (with updated access count), or
            ``None``.

        Example::

            node = kg.get_node("fact_abc123")
        """
        node = self._nodes.get(node_id)
        if node is not None:
            node.touch()
        else:
            logger.warning("get_node node not found: %s", node_id)
        return node

    def has_node(self, node_id: str) -> bool:
        """Check whether a node exists.

        Args:
            node_id: ID to check.

        Returns:
            ``True`` if the node is present.

        Example::

            if kg.has_node("n1"):
                ...
        """
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        """Return the total number of nodes. O(1)."""
        return len(self._nodes)

    @property
    def nodes(self) -> Iterator[MemoryNode]:
        """Iterate over all nodes in the graph.

        Yields:
            ``MemoryNode`` instances.
        """
        return iter(self._nodes.values())

    def get_nodes_by_type(self, node_type: NodeType) -> list[MemoryNode]:
        """Return all nodes of a given type using the type index.

        Args:
            node_type: The ``NodeType`` to filter by.

        Returns:
            A list of matching ``MemoryNode`` instances.

        Example::

            facts = kg.get_nodes_by_type(NodeType.FACT)
        """
        ids = self._type_index.get(node_type, set())
        return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def find_nodes(self, keyword: str, node_type: NodeType | None = None) -> list[MemoryNode]:
        """Search nodes by keyword with optional type filter.

        First attempts exact keyword index lookup; falls back to O(V)
        substring scan for languages without whitespace (e.g. Chinese).

        Args:
            keyword: Search term (case-insensitive).
            node_type: Optional ``NodeType`` filter.

        Returns:
            A list of matching ``MemoryNode`` instances.

        Example::

            results = kg.find_nodes("python", node_type=NodeType.FACT)
        """
        kw = keyword.lower()
        # Try exact keyword match first
        matched = self._keyword_index.get(kw)
        if matched:
            ids = set(matched)
            if node_type is not None:
                ids &= self._type_index.get(node_type, set())
            return [self._nodes[nid] for nid in ids if nid in self._nodes]
        # Fall back to substring match (e.g. Chinese without spaces)
        node_ids: set[str] = set()
        for node in self._nodes.values():
            if kw in node.content.lower():
                node_ids.add(node.id)
        if node_type is not None:
            node_ids &= self._type_index.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    # ── Edge operations ──

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0,
                 **metadata: Any) -> Edge | None:
        """Create a directed edge between two existing nodes.

        Args:
            source_id: ID of the source node (must exist).
            target_id: ID of the target node (must exist).
            edge_type: Semantic ``EdgeType`` for the relationship.
            weight: Edge weight in [0, 1]. Defaults to 1.0.
            **metadata: Arbitrary key-value pairs attached to the edge.

        Returns:
            The created ``Edge``, or ``None`` if either endpoint is missing.

        Example::

            kg.add_edge("n1", "n2", EdgeType.CAUSAL, weight=0.9)
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning("add_edge endpoints missing: %s -> %s", source_id, target_id)
            return None
        edge = Edge(source_id=source_id, target_id=target_id,
                    type=edge_type, weight=weight, metadata=metadata)
        existing = self._out_edges[source_id].get(target_id)
        self._out_edges[source_id][target_id] = edge
        self._in_edges[target_id][source_id] = edge
        if existing is None:
            self._edge_count += 1
        logger.debug("add_edge %s -> %s type=%s", source_id, target_id, edge_type.value)
        return edge

    def remove_edge(self, source_id: str, target_id: str) -> Edge | None:
        """Remove a directed edge.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            The removed ``Edge``, or ``None`` if no such edge exists.

        Example::

            kg.remove_edge("n1", "n2")
        """
        edge = self._out_edges.get(source_id, {}).pop(target_id, None)
        if edge is not None:
            self._in_edges.get(target_id, {}).pop(source_id, None)
            self._edge_count -= 1
        else:
            logger.warning("remove_edge edge not found: %s -> %s", source_id, target_id)
        return edge

    def get_edge(self, source_id: str, target_id: str) -> Edge | None:
        """Look up an edge by its endpoints.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            The ``Edge`` if found, else ``None``.
        """
        return self._out_edges.get(source_id, {}).get(target_id)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check whether a directed edge exists.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            ``True`` if the edge exists.
        """
        return target_id in self._out_edges.get(source_id, {})

    @property
    def edge_count(self) -> int:
        """Return the total number of directed edges. O(1)."""
        return self._edge_count

    def _sync_edge_count(self) -> None:
        """Recompute _edge_count from scratch (used after bulk internal mutations)."""
        self._edge_count = sum(len(targets) for targets in self._out_edges.values())

    # ── Neighbour queries ──

    def out_neighbors(self, node_id: str) -> list[str]:
        """Return IDs of all outgoing neighbours.

        Args:
            node_id: ID of the node.

        Returns:
            A list of target node IDs.
        """
        return list(self._out_edges.get(node_id, {}).keys())

    def in_neighbors(self, node_id: str) -> list[str]:
        """Return IDs of all incoming neighbours.

        Args:
            node_id: ID of the node.

        Returns:
            A list of source node IDs.
        """
        return list(self._in_edges.get(node_id, {}).keys())

    def out_edges(self, node_id: str) -> list[Edge]:
        """Return all outgoing edges for a node.

        Args:
            node_id: ID of the node.

        Returns:
            A list of ``Edge`` instances.
        """
        return list(self._out_edges.get(node_id, {}).values())

    def in_edges(self, node_id: str) -> list[Edge]:
        """Return all incoming edges for a node.

        Args:
            node_id: ID of the node.

        Returns:
            A list of ``Edge`` instances.
        """
        return list(self._in_edges.get(node_id, {}).values())

    def degree(self, node_id: str) -> int:
        """Return the total (in + out) degree of a node.

        Args:
            node_id: ID of the node.

        Returns:
            The sum of incoming and outgoing edge counts.
        """
        return len(self._out_edges.get(node_id, {})) + len(self._in_edges.get(node_id, {}))

    # ── Subgraph & merge ──

    def subgraph(self, center_id: str, depth: int = 2) -> KnowledgeGraph:
        """Extract a neighbourhood subgraph around ``center_id``.

        Expands bidirectionally (both in- and out-edges) up to ``depth``
        hops, then copies all edges whose endpoints are within the
        extracted node set.

        Args:
            center_id: ID of the central node.
            depth: Number of hops to expand. Defaults to 2.

        Returns:
            A new ``KnowledgeGraph`` containing the extracted subgraph.

        Example::

            local = kg.subgraph("event_42", depth=3)
        """
        sub = KnowledgeGraph()
        logger.debug("subgraph center=%s depth=%s", center_id, depth)
        visited: set[str] = set()
        frontier = {center_id}
        for _ in range(depth + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self._nodes.get(nid)
                if node:
                    sub.add_node(node)
                    for tgt in self.out_neighbors(nid):
                        next_frontier.add(tgt)
                    for src in self.in_neighbors(nid):
                        next_frontier.add(src)
            frontier = next_frontier - visited
        # Add edges within the subgraph
        for sid in visited:
            for edge in self.out_edges(sid):
                if edge.target_id in visited:
                    sub._out_edges[sid][edge.target_id] = edge
                    sub._in_edges[edge.target_id][sid] = edge
        return sub

    def merge(self, other: KnowledgeGraph) -> int:
        """Merge another graph into this one, skipping duplicates.

        Nodes with existing IDs are silently skipped. Edges whose both
        endpoints exist in the current graph are added if not already
        present.

        Args:
            other: The ``KnowledgeGraph`` to merge from.

        Returns:
            The number of newly added nodes.

        Example::

            added = kg.merge(other_kg)
        """
        added = 0
        for node in other.nodes:
            if node.id not in self._nodes:
                self.add_node(node)
                added += 1
        for sid in other._out_edges:
            for edge in other._out_edges[sid].values():
                if not self.has_edge(edge.source_id, edge.target_id):
                    existing_src = self._nodes.get(edge.source_id)
                    existing_tgt = self._nodes.get(edge.target_id)
                    if existing_src and existing_tgt:
                        self.add_edge(edge.source_id, edge.target_id,
                                      edge.type, edge.weight, **edge.metadata)
        logger.info("merge added %d nodes", added)
        return added

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a JSON-friendly dictionary.

        Returns:
            A dict with ``"nodes"`` and ``"edges"`` lists.

        Example::

            data = kg.to_dict()
            import json
            json_str = json.dumps(data)
        """
        nodes = []
        for n in self._nodes.values():
            nodes.append({
                "id": n.id, "type": n.type.value, "content": n.content,
                "importance": n.importance, "access_count": n.access_count,
                "created_at": n.created_at, "updated_at": n.updated_at,
                "metadata": n.metadata,
            })
        edges = []
        for sid in self._out_edges:
            for edge in self._out_edges[sid].values():
                edges.append({
                    "source_id": edge.source_id, "target_id": edge.target_id,
                    "type": edge.type.value, "weight": edge.weight,
                    "metadata": edge.metadata,
                })
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Deserialize a graph from a dictionary produced by ``to_dict``.

        Args:
            data: A dict with ``"nodes"`` and ``"edges"`` entries.

        Returns:
            A reconstructed ``KnowledgeGraph`` instance.

        Example::

            kg2 = KnowledgeGraph.from_dict(data)
        """
        kg = cls()
        for nd in data.get("nodes", []):
            node = MemoryNode(
                id=nd["id"], type=NodeType(nd["type"]), content=nd["content"],
                importance=nd.get("importance", 0.5),
                access_count=nd.get("access_count", 0),
                created_at=nd.get("created_at", ""),
                updated_at=nd.get("updated_at", ""),
                metadata=nd.get("metadata", {}),
            )
            kg._nodes[node.id] = node
            kg._type_index[node.type].add(node.id)
            for kw in kg._tokenize(node.content):
                kg._keyword_index[kw].add(node.id)
        for ed in data.get("edges", []):
            kg.add_edge(
                ed["source_id"], ed["target_id"],
                EdgeType(ed["type"]), ed.get("weight", 1.0),
                **ed.get("metadata", {}),
            )
        return kg

    # ── Statistics ──

    def stats(self) -> dict[str, Any]:
        """Return a summary of graph size and index coverage.

        Returns:
            A dict with keys ``"nodes"``, ``"edges"``, ``"types"``,
            ``"keywords"``.

        Example::

            info = kg.stats()
            # {"nodes": 42, "edges": 67, "types": {...}, "keywords": 128}
        """
        type_counts = {t.value: len(ids) for t, ids in self._type_index.items() if ids}
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "types": type_counts,
            "keywords": len(self._keyword_index),
        }

    # ── Internal helpers ──

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into lowercase keywords by whitespace splitting.

        Tokens shorter than 2 characters are discarded.

        Args:
            text: Input text.

        Returns:
            A set of lowercase tokens.
        """
        return {w.lower() for w in text.split() if len(w) > 1}
