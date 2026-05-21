"""Redis storage backend for knowledge graph persistence."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any

from agentic_os.core.graph.edge import Edge, EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import MemoryNode, NodeType
from agentic_os.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_PREFIX = "agentic"
_NODE_KEY = f"{_PREFIX}:node:{{}}"
_OUT_EDGES_KEY = f"{_PREFIX}:edges:out:{{}}"
_IN_EDGES_KEY = f"{_PREFIX}:edges:in:{{}}"
_META_KEY = f"{_PREFIX}:meta"


def _get_redis() -> ModuleType:
    """Lazy import redis, raising a helpful error if not installed."""
    try:
        import redis
        return redis
    except ImportError as err:
        raise ImportError(
            "redis package is required for RedisStorage. "
            "Install it with: pip install agentic-os-core[redis]"
        ) from err


class RedisStorage(StorageBackend):
    """Redis-backed persistent storage for knowledge graphs.

    Data layout:
        - ``agentic:node:{id}`` — Hash with serialized node fields
        - ``agentic:edges:out:{src_id}`` — Set of serialized outgoing edges
        - ``agentic:edges:in:{tgt_id}`` — Set of source node IDs (for reverse lookup)
        - ``agentic:meta`` — Hash for graph-level metadata

    Args:
        url: Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        prefix: Key prefix for namespacing. Defaults to ``"agentic"``.
        **kwargs: Additional keyword arguments passed to ``redis.Redis()``.

    Example::

        with RedisStorage("redis://localhost:6379/0") as store:
            store.save_graph(kg)
            loaded = store.load_graph()
    """

    def __init__(self, url: str = "redis://localhost:6379/0",
                 prefix: str = _PREFIX, **kwargs: Any) -> None:
        redis_mod = _get_redis()
        self._client = redis_mod.Redis.from_url(url, **kwargs)
        self._prefix = prefix
        self._lock = threading.Lock()
        logger.info("RedisStorage connected: %s", url)

    def _node_key(self, node_id: str) -> str:
        return f"{self._prefix}:node:{node_id}"

    def _out_key(self, node_id: str) -> str:
        return f"{self._prefix}:edges:out:{node_id}"

    def _in_key(self, node_id: str) -> str:
        return f"{self._prefix}:edges:in:{node_id}"

    def save_graph(self, graph: KnowledgeGraph) -> None:
        pipe = self._client.pipeline()
        # Clear existing data with this prefix
        self._clear_prefix(pipe)

        for node in graph.nodes:
            self._write_node(pipe, node)
        for sid in graph._out_edges:
            for edge in graph._out_edges[sid].values():
                self._write_edge(pipe, edge)

        pipe.execute()
        logger.info("save_graph: saved %d nodes", graph.node_count)

    def load_graph(self) -> KnowledgeGraph | None:
        kg = KnowledgeGraph()
        pattern = f"{self._prefix}:node:*"
        keys = list(self._client.scan_iter(match=pattern))
        if not keys:
            return None

        pipe = self._client.pipeline()
        for key in keys:
            pipe.hgetall(key)
        results = pipe.execute()

        for raw in results:
            if not raw:
                continue
            node = self._parse_node(raw)
            kg._nodes[node.id] = node
            kg._type_index[node.type].add(node.id)
            for kw in kg._tokenize(node.content):
                kg._keyword_index[kw].add(node.id)

        # Load edges
        for node_id in list(kg._nodes.keys()):
            out_key = self._out_key(node_id)
            raw_edges = self._client.smembers(out_key)
            for raw_edge in raw_edges:
                data = json.loads(raw_edge)
                edge = Edge(
                    source_id=data["source_id"],
                    target_id=data["target_id"],
                    type=EdgeType(data["type"]),
                    weight=data["weight"],
                    metadata=data.get("metadata", {}),
                )
                kg._out_edges[edge.source_id][edge.target_id] = edge
                kg._in_edges[edge.target_id][edge.source_id] = edge

        logger.info("load_graph: loaded %d nodes", kg.node_count)
        return kg

    def save_node(self, node: MemoryNode) -> None:
        pipe = self._client.pipeline()
        self._write_node(pipe, node)
        pipe.execute()
        logger.debug("save_node: %s", node.id)

    def load_node(self, node_id: str) -> MemoryNode | None:
        raw = self._client.hgetall(self._node_key(node_id))
        if not raw:
            return None
        return self._parse_node(raw)

    def delete_node(self, node_id: str) -> bool:
        existed = self._client.exists(self._node_key(node_id))
        if not existed:
            return False
        pipe = self._client.pipeline()
        pipe.delete(self._node_key(node_id))
        # Remove outgoing edges
        out_edges = self._client.smembers(self._out_key(node_id))
        for raw_edge in out_edges:
            data = json.loads(raw_edge)
            pipe.srem(self._in_key(data["target_id"]), node_id)
        pipe.delete(self._out_key(node_id))
        # Remove from incoming edge sets
        in_sources = self._client.smembers(self._in_key(node_id))
        for src_id in in_sources:
            src_id_str = src_id if isinstance(src_id, str) else src_id.decode()
            out_key = self._out_key(src_id_str)
            members = self._client.smembers(out_key)
            for raw_edge in members:
                edge_data = json.loads(raw_edge)
                if edge_data["target_id"] == node_id:
                    pipe.srem(out_key, raw_edge)
        pipe.delete(self._in_key(node_id))
        pipe.execute()
        return True

    def query_nodes(self, filter_fn: Callable[[str, dict[str, Any]], bool]) -> list[MemoryNode]:
        results: list[MemoryNode] = []
        pattern = f"{self._prefix}:node:*"
        pipe = self._client.pipeline()
        keys = list(self._client.scan_iter(match=pattern))
        for key in keys:
            pipe.hgetall(key)
        raw_results = pipe.execute()

        for raw in raw_results:
            if not raw:
                continue
            node = self._parse_node(raw)
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
        self._client.close()
        logger.info("RedisStorage closed")

    def __enter__(self) -> RedisStorage:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Internal helpers ──

    def _clear_prefix(self, pipe: Any) -> None:
        for pattern in [f"{self._prefix}:node:*", f"{self._prefix}:edges:*"]:
            for key in self._client.scan_iter(match=pattern):
                pipe.delete(key)

    def _write_node(self, pipe: Any, node: MemoryNode) -> None:
        pipe.hset(self._node_key(node.id), mapping={
            "id": node.id,
            "type": node.type.value,
            "content": node.content,
            "importance": str(node.importance),
            "access_count": str(node.access_count),
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "metadata": json.dumps(node.metadata),
        })

    def _write_edge(self, pipe: Any, edge: Edge) -> None:
        edge_data = json.dumps({
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type.value,
            "weight": edge.weight,
            "metadata": edge.metadata,
        })
        pipe.sadd(self._out_key(edge.source_id), edge_data)
        pipe.sadd(self._in_key(edge.target_id), edge.source_id)

    @staticmethod
    def _parse_node(raw: dict[bytes, bytes] | dict[str, str]) -> MemoryNode:
        # Normalize to str-keyed dict
        decoded: dict[str, str] = {}
        for k, v in raw.items():
            ks = k.decode() if isinstance(k, bytes) else k
            vs = v.decode() if isinstance(v, bytes) else v
            decoded[ks] = vs

        def _get(key: str) -> str:
            return decoded.get(key, "")

        return MemoryNode(
            id=_get("id"),
            type=NodeType(_get("type")),
            content=_get("content"),
            importance=float(_get("importance") or "0.5"),
            access_count=int(_get("access_count") or "0"),
            created_at=_get("created_at"),
            updated_at=_get("updated_at"),
            metadata=json.loads(_get("metadata") or "{}"),
        )
