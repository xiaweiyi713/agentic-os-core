"""向量相似度检索系统测试"""

import math
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from agentic_os.core.memory.longterm import LongTermMemory
from agentic_os.core.vector.base import SearchResult, VectorStore
from agentic_os.core.vector.numpy_backend import NumpyVectorStore


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _simple_embedder(text: str) -> list[float]:
    """将文本转换为一个简单的伪向量（基于字符 ASCII 值）。"""
    dim = 8
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[i % dim] += float(ord(ch))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ---------------------------------------------------------------------------
# NumpyVectorStore 基础测试
# ---------------------------------------------------------------------------

class TestNumpyVectorStore:
    def test_add_and_count(self) -> None:
        store = NumpyVectorStore(dimension=3)
        assert store.count() == 0
        store.add("v1", [1.0, 0.0, 0.0])
        assert store.count() == 1
        store.add("v2", [0.0, 1.0, 0.0])
        assert store.count() == 2

    def test_add_dimension_mismatch(self) -> None:
        store = NumpyVectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add("v1", [1.0, 0.0])

    def test_add_with_metadata(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0], {"label": "x-axis"})
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].metadata == {"label": "x-axis"}

    def test_add_batch(self) -> None:
        store = NumpyVectorStore(dimension=3)
        items = [
            ("v1", [1.0, 0.0, 0.0], {"label": "x"}),
            ("v2", [0.0, 1.0, 0.0], {"label": "y"}),
            ("v3", [0.0, 0.0, 1.0], {"label": "z"}),
        ]
        store.add_batch(items)
        assert store.count() == 3

    def test_search_cosine_similarity(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        store.add("v2", [0.0, 1.0, 0.0])
        store.add("v3", [0.0, 0.0, 1.0])

        # 查询与 v1 最相似
        results = store.search([0.9, 0.1, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0].id == "v1"
        assert results[0].score > results[1].score

    def test_search_top_k(self) -> None:
        store = NumpyVectorStore(dimension=3)
        for i in range(10):
            vec = [0.0] * 3
            vec[i % 3] = 1.0
            store.add(f"v{i}", vec)
        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self) -> None:
        store = NumpyVectorStore(dimension=3)
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    def test_search_zero_query(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        results = store.search([0.0, 0.0, 0.0])
        assert results == []

    def test_search_zero_vector_skipped(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        store.add("v2", [0.0, 0.0, 0.0])  # 零向量
        results = store.search([1.0, 0.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].id == "v1"

    def test_delete_existing(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        assert store.delete("v1") is True
        assert store.count() == 0

    def test_delete_nonexistent(self) -> None:
        store = NumpyVectorStore()
        assert store.delete("ghost") is False

    def test_search_result_is_search_result(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0])
        assert isinstance(results[0], SearchResult)

    def test_overwrite_by_id(self) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])
        store.add("v1", [0.0, 1.0, 0.0])  # overwrite
        assert store.count() == 1
        results = store.search([0.0, 1.0, 0.0])
        assert results[0].id == "v1"

    def test_search_scores_cosine_range(self) -> None:
        """确保分数在 [-1, 1] 范围内（余弦相似度）。"""
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 2.0, 3.0])
        store.add("v2", [-1.0, -2.0, -3.0])
        results = store.search([1.0, 0.5, 0.1], top_k=2)
        assert len(results) == 2
        for r in results:
            assert -1.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# save / load 测试
# ---------------------------------------------------------------------------

class TestNumpyVectorStorePersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0], {"label": "x"})
        store.add("v2", [0.0, 1.0, 0.0], {"label": "y"})

        path = tmp_path / "vectors"
        store.save(path)

        loaded = NumpyVectorStore.load(path)
        assert loaded.count() == 2

        results = loaded.search([1.0, 0.0, 0.0], top_k=2)
        assert results[0].id == "v1"
        assert results[0].metadata == {"label": "x"}

    def test_save_and_load_empty(self, tmp_path: Path) -> None:
        store = NumpyVectorStore(dimension=3)
        path = tmp_path / "empty"
        store.save(path)

        loaded = NumpyVectorStore.load(path)
        assert loaded.count() == 0

    def test_save_creates_npz_and_json(self, tmp_path: Path) -> None:
        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0])

        path = tmp_path / "test"
        store.save(path)

        assert path.with_suffix(".npz").exists()
        assert path.with_suffix(".json").exists()


# ---------------------------------------------------------------------------
# LongTermMemory 向量检索集成测试
# ---------------------------------------------------------------------------

class TestLongTermMemoryVectorSearch:
    def _make_ltm(self) -> LongTermMemory:
        store = NumpyVectorStore()
        return LongTermMemory(vector_store=store, embedder=_simple_embedder)

    def test_retrieve_similar_basic(self) -> None:
        ltm = self._make_ltm()
        ltm.store_fact("Python 是解释型语言")
        ltm.store_fact("Rust 是编译型语言")
        ltm.store_fact("今天天气晴朗")

        ltm.index_all()
        results = ltm.retrieve_similar("Python 编程", top_k=2)
        assert len(results) >= 1
        # Python 相关事实应在前面
        assert any("Python" in n.content for n in results)

    def test_index_all_returns_count(self) -> None:
        ltm = self._make_ltm()
        ltm.store_fact("事实 A")
        ltm.store_fact("事实 B")
        ltm.store_fact("事实 C")
        count = ltm.index_all()
        assert count == 3

    def test_retrieve_similar_without_store_raises(self) -> None:
        ltm = LongTermMemory()
        with pytest.raises(RuntimeError, match="vector_store"):
            ltm.retrieve_similar("test")

    def test_index_all_without_store_raises(self) -> None:
        ltm = LongTermMemory()
        with pytest.raises(RuntimeError, match="vector_store"):
            ltm.index_all()

    def test_backward_compatible_no_vector_args(self) -> None:
        """确保不传 vector_store/embedder 时旧功能不受影响。"""
        ltm = LongTermMemory()
        nid = ltm.store_episode("普通记忆存储")
        assert ltm.get_node(nid) is not None

    def test_retrieve_similar_returns_memory_nodes(self) -> None:
        ltm = self._make_ltm()
        ltm.store_fact("机器学习是人工智能的子领域")
        ltm.index_all()
        results = ltm.retrieve_similar("深度学习")
        for node in results:
            assert hasattr(node, "content")
            assert hasattr(node, "id")

    def test_search_result_metadata_preserved(self) -> None:
        ltm = self._make_ltm()
        ltm.store_fact("测试元数据")
        ltm.index_all()

        # 验证 vector store 中包含正确的元数据
        results = ltm._vector_store.search(_simple_embedder("测试"), top_k=1)  # type: ignore[union-attr]
        assert len(results) == 1
        assert "content" in results[0].metadata
        assert "type" in results[0].metadata


# ---------------------------------------------------------------------------
# VectorStore 抽象基类测试
# ---------------------------------------------------------------------------

class TestVectorStoreABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self) -> None:
        class IncompleteStore(VectorStore):
            def add(self, id: str, vector: list[float],
                    metadata: dict | None = None) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStore()  # type: ignore[abstract]
