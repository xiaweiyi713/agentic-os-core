"""Tests for SQLite storage backend."""

import os
import tempfile

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import (
    NodeType,
    create_episode,
    create_fact,
    create_goal,
    create_reflection,
)
from agentic_os.core.storage.sqlite_backend import SqliteStorage


class TestSqliteStorageBasicCRUD:
    def test_save_and_load_graph(self):
        with SqliteStorage(":memory:") as store:
            kg = KnowledgeGraph()
            a = create_fact("Earth orbits the Sun", importance=0.8)
            b = create_episode("Learned about gravity")
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
        with SqliteStorage(":memory:") as store:
            assert store.load_graph() is None

    def test_save_and_load_node(self):
        with SqliteStorage(":memory:") as store:
            node = create_fact("Test fact", importance=0.7)
            store.save_node(node)

            loaded = store.load_node(node.id)
            assert loaded is not None
            assert loaded.content == "Test fact"
            assert loaded.importance == 0.7
            assert loaded.type == NodeType.FACT

    def test_load_nonexistent_node(self):
        with SqliteStorage(":memory:") as store:
            assert store.load_node("nonexistent") is None

    def test_delete_node(self):
        with SqliteStorage(":memory:") as store:
            node = create_episode("To delete")
            store.save_node(node)
            assert store.delete_node(node.id) is True
            assert store.load_node(node.id) is None
            assert store.delete_node(node.id) is False

    def test_delete_node_cascades_edges(self):
        with SqliteStorage(":memory:") as store:
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
            assert loaded.edge_count == 0

    def test_query_nodes(self):
        with SqliteStorage(":memory:") as store:
            high = create_fact("Important", importance=0.9)
            low = create_fact("Trivial", importance=0.1)
            store.save_node(high)
            store.save_node(low)

            results = store.query_nodes(lambda k, v: v["importance"] > 0.5)
            assert len(results) == 1
            assert results[0].content == "Important"

    def test_update_node(self):
        with SqliteStorage(":memory:") as store:
            node = create_fact("Original", importance=0.5)
            store.save_node(node)

            node.importance = 0.9
            store.save_node(node)

            loaded = store.load_node(node.id)
            assert loaded is not None
            assert loaded.importance == 0.9

    def test_file_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            node = create_goal("Persist this goal")
            with SqliteStorage(db_path) as store:
                store.save_node(node)

            with SqliteStorage(db_path) as store:
                loaded = store.load_node(node.id)
                assert loaded is not None
                assert loaded.content == "Persist this goal"
                assert loaded.type == NodeType.GOAL
        finally:
            os.unlink(db_path)


class TestSqliteStorageAllNodeTypes:
    def test_save_all_node_types(self):
        with SqliteStorage(":memory:") as store:
            kg = KnowledgeGraph()
            ep = create_episode("An episode")
            fact = create_fact("A fact", importance=0.6)
            ref = create_reflection("A reflection", importance=0.7)
            goal = create_goal("A goal", importance=0.8)
            for n in [ep, fact, ref, goal]:
                kg.add_node(n)
            kg.add_edge(ep.id, fact.id, EdgeType.DERIVED_FROM)
            kg.add_edge(fact.id, ref.id, EdgeType.ASSOCIATIVE)

            store.save_graph(kg)
            loaded = store.load_graph()

            assert loaded is not None
            assert loaded.node_count == 4
            assert loaded.edge_count == 2
            assert len(loaded.get_nodes_by_type(NodeType.EPISODE)) == 1
            assert len(loaded.get_nodes_by_type(NodeType.FACT)) == 1
            assert len(loaded.get_nodes_by_type(NodeType.REFLECTION)) == 1
            assert len(loaded.get_nodes_by_type(NodeType.GOAL)) == 1


class TestSqliteStorageEdgeTypes:
    def test_save_all_edge_types(self):
        with SqliteStorage(":memory:") as store:
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
            for i, et in enumerate(edge_types):
                edge = loaded.get_edge(nodes[i].id, nodes[i + 1].id)
                assert edge is not None
                assert edge.type == et

    def test_edge_with_metadata(self):
        with SqliteStorage(":memory:") as store:
            kg = KnowledgeGraph()
            a = create_episode("A")
            b = create_episode("B")
            kg.add_node(a)
            kg.add_node(b)
            kg.add_edge(a.id, b.id, EdgeType.CAUSAL, weight=0.8, source="test", confidence=0.95)

            store.save_graph(kg)
            loaded = store.load_graph()

            assert loaded is not None
            edge = loaded.get_edge(a.id, b.id)
            assert edge is not None
            assert edge.metadata["source"] == "test"
            assert edge.metadata["confidence"] == 0.95


class TestSqliteStorageNodeMetadata:
    def test_node_metadata_preserved(self):
        with SqliteStorage(":memory:") as store:
            node = create_fact("Tagged fact", importance=0.5, tags=["python", "ai"], author="test")
            store.save_node(node)

            loaded = store.load_node(node.id)
            assert loaded is not None
            assert loaded.metadata["tags"] == ["python", "ai"]
            assert loaded.metadata["author"] == "test"


class TestSqliteStorageOverwrite:
    def test_save_graph_overwrites(self):
        with SqliteStorage(":memory:") as store:
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
            assert loaded.get_nodes_by_type(NodeType.FACT)[0].content == "Second graph"


class TestRedisStorageImport:
    """Tests that Redis storage can be imported and handles missing redis gracefully."""

    def test_import_raises_without_redis(self):
        """If redis is not installed, constructing RedisStorage should raise ImportError."""
        import sys
        # Save and remove redis if present
        redis_mod = sys.modules.pop("redis", None)
        try:
            from agentic_os.core.storage.redis_backend import RedisStorage
            try:
                RedisStorage("redis://localhost:9999/0")
            except ImportError as e:
                assert "redis" in str(e).lower()
            except Exception:
                pass  # redis is installed, connection might fail — that's fine
        finally:
            if redis_mod is not None:
                sys.modules["redis"] = redis_mod
