"""Tests for the LlamaIndex adapter — no llama-index installation required."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_os.core.vector.base import SearchResult
from agentic_os.ext.llamaindex.vector_store import (
    AgenticOSVectorStore,
    _HAS_LLAMAINDEX,
)


# ---------------------------------------------------------------------------
# Lightweight mocks that mimic LlamaIndex objects without requiring the package
# ---------------------------------------------------------------------------


class MockVectorStore:
    """Minimal AgenticOS VectorStore implementation for testing."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def add(
        self, id: str, vector: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        self._data[id] = {"vector": vector, "metadata": metadata or {}}

    def add_batch(
        self, items: list[tuple[str, list[float], dict[str, Any]]]
    ) -> None:
        for id_, vector, metadata in items:
            self.add(id_, vector, metadata)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]:
        results: list[SearchResult] = []
        for id_, data in self._data.items():
            # Simple dot-product similarity
            score = sum(a * b for a, b in zip(query_vector, data["vector"]))
            results.append(
                SearchResult(id=id_, score=score, metadata=data["metadata"])
            )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete(self, id: str) -> bool:
        if id in self._data:
            del self._data[id]
            return True
        return False

    def count(self) -> int:
        return len(self._data)


class MockNode:
    """Mimics a LlamaIndex TextNode with the attributes we access."""

    def __init__(
        self,
        id: str,
        text: str = "",
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.node_id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}


class MockQuery:
    """Mimics a LlamaIndex VectorStoreQuery."""

    def __init__(
        self,
        query_embedding: list[float] | None = None,
        similarity_top_k: int = 10,
    ) -> None:
        self.query_embedding = query_embedding
        self.similarity_top_k = similarity_top_k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgenticOSVectorStoreInit:
    """Initialization tests."""

    def test_basic_init(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)
        assert adapter._store is store
        assert adapter._embedder is None

    def test_init_with_embedder(self) -> None:
        store = MockVectorStore()
        embedder = MagicMock(return_value=[0.1, 0.2, 0.3])
        adapter = AgenticOSVectorStore(vector_store=store, embedder=embedder)
        assert adapter._embedder is embedder

    def test_stores_text_attribute(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)
        assert adapter.stores_text is True


class TestAgenticOSVectorStoreAdd:
    """Tests for the add() method."""

    def test_add_nodes_with_embeddings(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        nodes = [
            MockNode(id="n1", embedding=[1.0, 0.0, 0.0], metadata={"source": "a"}),
            MockNode(id="n2", embedding=[0.0, 1.0, 0.0], metadata={"source": "b"}),
        ]
        ids = adapter.add(nodes)

        assert ids == ["n1", "n2"]
        assert store.count() == 2

    def test_add_nodes_with_embedder_fallback(self) -> None:
        store = MockVectorStore()
        embedder = MagicMock(return_value=[0.5, 0.5, 0.5])
        adapter = AgenticOSVectorStore(vector_store=store, embedder=embedder)

        nodes = [MockNode(id="n1", text="hello")]
        ids = adapter.add(nodes)

        assert ids == ["n1"]
        assert store.count() == 1
        embedder.assert_called_once_with("hello")

    def test_add_node_without_embedding_skipped(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        nodes = [MockNode(id="n1", text="hello")]
        ids = adapter.add(nodes)

        assert ids == []
        assert store.count() == 0

    def test_add_empty_list(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        ids = adapter.add([])
        assert ids == []
        assert store.count() == 0


class TestAgenticOSVectorStoreQuery:
    """Tests for the query() method."""

    @pytest.mark.skipif(_HAS_LLAMAINDEX, reason="Test covers no-llamaindex path")
    def test_query_without_llamaindex_raises(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        with pytest.raises(ImportError, match="llama-index is required"):
            adapter.query(MockQuery(query_embedding=[1.0, 0.0]))

    @pytest.mark.skipif(not _HAS_LLAMAINDEX, reason="Requires llama-index installed")
    def test_query_with_llamaindex(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        adapter.add([MockNode(id="n1", embedding=[1.0, 0.0])])
        adapter.add([MockNode(id="n2", embedding=[0.0, 1.0])])

        result = adapter.query(MockQuery(query_embedding=[0.9, 0.1], similarity_top_k=2))
        assert len(result.ids) == 2
        assert result.ids[0] == "n1"  # closer to query

    @pytest.mark.skipif(not _HAS_LLAMAINDEX, reason="Requires llama-index installed")
    def test_query_with_no_embedding_returns_empty(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        result = adapter.query(MockQuery(query_embedding=None))
        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []


class TestAgenticOSVectorStoreDelete:
    """Tests for the delete() method."""

    def test_delete_existing_node(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        adapter.add([MockNode(id="n1", embedding=[1.0, 0.0])])
        assert store.count() == 1

        adapter.delete("n1")
        assert store.count() == 0

    def test_delete_nonexistent_node_no_error(self) -> None:
        store = MockVectorStore()
        adapter = AgenticOSVectorStore(vector_store=store)

        # Should not raise
        adapter.delete("nonexistent")
