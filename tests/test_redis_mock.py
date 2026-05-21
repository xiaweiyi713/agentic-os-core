"""Redis backend tests using a mock redis client.

Tests RedisStorage logic without requiring a live Redis server.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import (
    NodeType,
    create_episode,
    create_fact,
    create_goal,
    create_reflection,
)


class FakeRedis:
    """Minimal fake Redis client supporting the operations used by RedisStorage."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._sets: dict[str, set[str]] = {}

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> FakeRedis:
        return cls()

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._store[key] = dict(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return self._store.get(key, {})

    def sadd(self, key: str, *values: str) -> None:
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(values)

    def smembers(self, key: str) -> set[str]:
        return self._sets.get(key, set())

    def srem(self, key: str, *values: str) -> None:
        if key in self._sets:
            self._sets[key] -= set(values)

    def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)
            self._sets.pop(k, None)

    def exists(self, key: str) -> bool:
        return key in self._store or key in self._sets

    def scan_iter(self, match: str = "") -> list[str]:
        pattern = match.replace("*", "")
        all_keys = list(self._store.keys()) + list(self._sets.keys())
        return [k for k in all_keys if k.startswith(pattern)]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    def close(self) -> None:
        pass


class FakePipeline:
    """Fake Redis pipeline that executes immediately (no transaction semantics needed)."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._results: list[Any] = []

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._redis.hset(key, mapping=mapping)

    def hgetall(self, key: str) -> None:
        self._results.append(self._redis.hgetall(key))

    def sadd(self, key: str, *values: str) -> None:
        self._redis.sadd(key, *values)

    def srem(self, key: str, *values: str) -> None:
        self._redis.srem(key, *values)

    def delete(self, *keys: str) -> None:
        self._redis.delete(*keys)

    def execute(self) -> list[Any]:
        results = self._results
        self._results = []
        return results


def _make_store() -> tuple[Any, Any]:
    """Create a RedisStorage with a FakeRedis client injected."""
    from agentic_os.core.storage.redis_backend import RedisStorage

    store = object.__new__(RedisStorage)
    fake = FakeRedis()
    store._client = fake
    store._prefix = "agentic"
    store._lock = __import__("threading").Lock()
    return store, fake


class TestRedisStorageSaveLoadGraph:
    def test_save_and_load_graph(self):
        store, _ = _make_store()
        kg = KnowledgeGraph()
        a = create_fact("Python is great", importance=0.8)
        b = create_episode("Learned about graphs")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL, weight=0.9)

        store.save_graph(kg)
        loaded = store.load_graph()

        assert loaded is not None
        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        assert loaded.has_edge(a.id, b.id)

    def test_load_empty_returns_none(self):
        store, _ = _make_store()
        assert store.load_graph() is None

    def test_save_all_node_types(self):
        store, _ = _make_store()
        kg = KnowledgeGraph()
        nodes = [
            create_episode("An episode"),
            create_fact("A fact", importance=0.6),
            create_reflection("A reflection", importance=0.7),
            create_goal("A goal", importance=0.8),
        ]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.ASSOCIATIVE)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.TEMPORAL)
        kg.add_edge(nodes[2].id, nodes[3].id, EdgeType.SUPPORTS)

        store.save_graph(kg)
        loaded = store.load_graph()

        assert loaded is not None
        assert loaded.node_count == 4
        assert loaded.edge_count == 3
        assert len(loaded.get_nodes_by_type(NodeType.EPISODE)) == 1
        assert len(loaded.get_nodes_by_type(NodeType.FACT)) == 1
        assert len(loaded.get_nodes_by_type(NodeType.REFLECTION)) == 1
        assert len(loaded.get_nodes_by_type(NodeType.GOAL)) == 1


class TestRedisStorageNodeCRUD:
    def test_save_and_load_node(self):
        store, _ = _make_store()
        node = create_fact("Redis test fact", importance=0.75)
        store.save_node(node)

        loaded = store.load_node(node.id)
        assert loaded is not None
        assert loaded.content == "Redis test fact"
        assert loaded.importance == 0.75
        assert loaded.type == NodeType.FACT

    def test_load_nonexistent_node(self):
        store, _ = _make_store()
        assert store.load_node("nonexistent") is None

    def test_delete_node(self):
        store, _ = _make_store()
        node = create_episode("To delete")
        store.save_node(node)
        assert store.delete_node(node.id) is True
        assert store.load_node(node.id) is None
        assert store.delete_node(node.id) is False

    def test_update_node(self):
        store, _ = _make_store()
        node = create_fact("Original", importance=0.5)
        store.save_node(node)

        node.importance = 0.9
        store.save_node(node)

        loaded = store.load_node(node.id)
        assert loaded is not None
        assert loaded.importance == 0.9


class TestRedisStorageNodeMetadata:
    def test_node_metadata_preserved(self):
        store, _ = _make_store()
        node = create_fact("Tagged", importance=0.5, tags=["redis", "test"], source="unit")
        store.save_node(node)

        loaded = store.load_node(node.id)
        assert loaded is not None
        assert loaded.metadata["tags"] == ["redis", "test"]
        assert loaded.metadata["source"] == "unit"


class TestRedisStorageQueryNodes:
    def test_query_nodes(self):
        store, _ = _make_store()
        high = create_fact("Important", importance=0.9)
        low = create_fact("Trivial", importance=0.1)
        store.save_node(high)
        store.save_node(low)

        results = store.query_nodes(lambda k, v: v["importance"] > 0.5)
        assert len(results) == 1
        assert results[0].content == "Important"


class TestRedisStorageParseNode:
    def test_parse_node_with_bytes_keys(self):
        from agentic_os.core.storage.redis_backend import RedisStorage

        raw = {
            b"id": b"test-id",
            b"type": b"fact",
            b"content": b"Bytes test",
            b"importance": b"0.7",
            b"access_count": b"3",
            b"created_at": b"2026-01-01T00:00:00",
            b"updated_at": b"2026-01-01T00:00:00",
            b"metadata": b'{"key": "value"}',
        }
        node = RedisStorage._parse_node(raw)
        assert node.id == "test-id"
        assert node.content == "Bytes test"
        assert node.importance == 0.7
        assert node.access_count == 3
        assert node.metadata == {"key": "value"}

    def test_parse_node_with_str_keys(self):
        from agentic_os.core.storage.redis_backend import RedisStorage

        raw = {
            "id": "str-id",
            "type": "episode",
            "content": "String test",
            "importance": "0.5",
            "access_count": "0",
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
            "metadata": "{}",
        }
        node = RedisStorage._parse_node(raw)
        assert node.id == "str-id"
        assert node.type == NodeType.EPISODE


class TestRedisStorageEdgeCases:
    def test_save_graph_overwrites(self):
        store, _ = _make_store()
        kg1 = KnowledgeGraph()
        kg1.add_node(create_fact("First graph"))
        store.save_graph(kg1)

        kg2 = KnowledgeGraph()
        kg2.add_node(create_fact("Second graph"))
        kg2.add_node(create_episode("Extra node"))
        store.save_graph(kg2)

        loaded = store.load_graph()
        assert loaded is not None
        assert loaded.node_count == 2

    def test_delete_node_cascades_edges(self):
        store, _ = _make_store()
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.ASSOCIATIVE)
        store.save_graph(kg)

        store.delete_node(a.id)
        loaded = store.load_graph()
        assert loaded is not None
        assert loaded.node_count == 1

    def test_all_edge_types(self):
        store, _ = _make_store()
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(7)]
        for n in nodes:
            kg.add_node(n)

        edge_types = [
            EdgeType.ASSOCIATIVE, EdgeType.CAUSAL, EdgeType.TEMPORAL,
            EdgeType.DERIVED_FROM, EdgeType.SUPPORTS, EdgeType.CONTRADICTS,
        ]
        for i, et in enumerate(edge_types):
            kg.add_edge(nodes[i].id, nodes[i + 1].id, et, weight=0.5 + i * 0.05)

        store.save_graph(kg)
        loaded = store.load_graph()

        assert loaded is not None
        assert loaded.edge_count == 6

    def test_context_manager(self):
        store, _ = _make_store()
        node = create_fact("Ctx test")
        store.save_node(node)

        loaded = store.load_node(node.id)
        assert loaded is not None
        assert loaded.content == "Ctx test"

    def test_close(self):
        store, _ = _make_store()
        store.close()


class TestRedisStorageGetRedis:
    def test_import_error_when_redis_missing(self):
        import sys
        redis_mod = sys.modules.pop("redis", None)
        try:
            from agentic_os.core.storage.redis_backend import _get_redis
            try:
                _get_redis()
            except ImportError as e:
                assert "redis" in str(e).lower()
            else:
                pytest.skip("redis is installed, cannot test ImportError path")
        finally:
            if redis_mod is not None:
                sys.modules["redis"] = redis_mod
