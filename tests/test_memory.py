"""记忆管理系统测试"""


from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.memory.consolidation import (
    importance_consolidation,
    pattern_consolidation,
    simple_consolidation,
)
from agentic_os.core.memory.longterm import LongTermMemory, RetrievalStrategy
from agentic_os.core.memory.manager import MemoryManager
from agentic_os.core.memory.working import WorkingMemory


class TestWorkingMemory:
    def test_put_and_get(self):
        wm = WorkingMemory(capacity=10)
        key = wm.put_content("测试内容", tag="test")
        result = wm.get(key)
        assert result is not None
        assert result["content"] == "测试内容"

    def test_lru_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.put("a", {"content": "A"})
        wm.put("b", {"content": "B"})
        wm.put("c", {"content": "C"})
        wm.put("d", {"content": "D"})  # 应淘汰 A

        assert wm.get("a") is None
        assert wm.get("d") is not None

    def test_lru_access_refreshes(self):
        wm = WorkingMemory(capacity=3)
        wm.put("a", {"content": "A"})
        wm.put("b", {"content": "B"})
        wm.put("c", {"content": "C"})
        wm.get("a")  # 刷新 A
        wm.put("d", {"content": "D"})  # 应淘汰 B

        assert wm.get("a") is not None
        assert wm.get("b") is None

    def test_recent(self):
        wm = WorkingMemory(capacity=10)
        for i in range(5):
            wm.put(f"k{i}", {"content": f"C{i}"})
        recent = wm.recent(3)
        assert len(recent) == 3

    def test_capacity(self):
        wm = WorkingMemory(capacity=5)
        assert wm.capacity == 5
        assert wm.size == 0


class TestLongTermMemory:
    def test_store_and_retrieve(self):
        ltm = LongTermMemory()
        nid = ltm.store_episode("用户问了关于 Python 的问题")
        node = ltm.get_node(nid)
        assert node is not None
        assert "Python" in node.content

    def test_store_different_types(self):
        ltm = LongTermMemory()
        eid = ltm.store_episode("经历")
        fid = ltm.store_fact("事实")
        rid = ltm.store_reflection("反思")
        gid = ltm.store_goal("目标")

        assert ltm.get_node(eid) is not None
        assert ltm.get_node(fid) is not None
        assert ltm.get_node(rid) is not None
        assert ltm.get_node(gid) is not None

    def test_keyword_retrieval(self):
        ltm = LongTermMemory()
        ltm.store_episode("今天学习了 Python 装饰器")
        ltm.store_episode("明天要学习 Rust")

        results = ltm.retrieve("Python", strategy=RetrievalStrategy.KEYWORD)
        assert len(results) >= 1
        assert any("Python" in n.content for n in results)

    def test_importance_retrieval(self):
        ltm = LongTermMemory()
        ltm.store_fact("低重要性事实", importance=0.1)
        ltm.store_fact("高重要性事实", importance=0.9)

        results = ltm.retrieve("", strategy=RetrievalStrategy.IMPORTANCE, top_k=2)
        assert len(results) >= 1

    def test_recency_retrieval(self):
        ltm = LongTermMemory()
        ltm.store_episode("旧记忆")
        ltm.store_episode("新记忆")

        results = ltm.retrieve("", strategy=RetrievalStrategy.RECENCY, top_k=1)
        assert len(results) == 1
        assert "新记忆" in results[0].content

    def test_association_retrieval(self):
        ltm = LongTermMemory()
        nid1 = ltm.store_episode("Python 基础")
        nid2 = ltm.store_episode("Python 进阶")
        _nid3 = ltm.store_episode("Rust basics")
        ltm.link(nid1, nid2, EdgeType.CAUSAL)

        results = ltm.retrieve("", strategy=RetrievalStrategy.ASSOCIATION,
                               top_k=5, seed_id=nid1)
        assert len(results) >= 1

    def test_find_path(self):
        ltm = LongTermMemory()
        n1 = ltm.store_episode("A")
        n2 = ltm.store_episode("B")
        n3 = ltm.store_episode("C")
        ltm.link(n1, n2, EdgeType.CAUSAL)
        ltm.link(n2, n3, EdgeType.CAUSAL)

        path = ltm.find_path(n1, n3)
        assert path is not None
        assert len(path) == 3

    def test_stats(self):
        ltm = LongTermMemory()
        ltm.store_episode("E1")
        ltm.store_fact("F1")
        stats = ltm.stats()
        assert stats["nodes"] == 2


class TestMemoryManager:
    def test_add_and_recall(self):
        mm = MemoryManager()
        mm.add_experience("学习了图论算法")
        mm.add_experience("研究了 MCTS 搜索")

        results = mm.recall("图论")
        assert len(results) >= 1

    def test_store_fact_and_recall(self):
        mm = MemoryManager()
        mm.store_fact("Python GIL 限制了多线程性能", importance=0.8)

        results = mm.recall("Python")
        assert len(results) >= 1

    def test_consolidate(self):
        mm = MemoryManager()
        mm.add_experience("短期记忆A")
        mm.add_experience("短期记忆B")

        count = mm.consolidate()
        assert count == 2
        assert mm.working.size == 0

    def test_reflect(self):
        mm = MemoryManager()
        mm.add_experience("调试了图论算法")
        mm.add_experience("完成了图论测试")
        mm.add_experience("编写了图论文档")

        ids = mm.reflect()
        assert len(ids) >= 1

    def test_forget(self):
        mm = MemoryManager()
        mm.store_fact("不重要", importance=0.01)
        mm.store_fact("重要", importance=0.9)

        removed = mm.forget(min_importance=0.5)
        assert removed >= 1

    def test_link_memories(self):
        mm = MemoryManager()
        fid1 = mm.store_fact("前提条件")
        fid2 = mm.store_fact("结论")
        mm.link_memories(fid1, fid2, EdgeType.CAUSAL)

        results = mm.recall_associated(fid1, top_k=5)
        assert len(results) >= 1

    def test_stats(self):
        mm = MemoryManager()
        mm.add_experience("test")
        stats = mm.stats()
        assert stats["working_memory"] == 1


class TestConsolidation:
    def test_simple_consolidation(self):
        wm = WorkingMemory()
        ltm = LongTermMemory()
        wm.put_content("内容A")
        wm.put_content("内容B")

        count = simple_consolidation(wm, ltm)
        assert count == 2
        assert wm.size == 0

    def test_importance_consolidation(self):
        wm = WorkingMemory()
        ltm = LongTermMemory()
        wm.put("a", {"content": "重要", "metadata": {"importance": 0.9}})
        wm.put("b", {"content": "不重要", "metadata": {"importance": 0.1}})

        count = importance_consolidation(wm, ltm, threshold=0.5)
        assert count == 1

    def test_pattern_consolidation(self):
        wm = WorkingMemory()
        ltm = LongTermMemory()
        wm.put_content("Python 是解释型语言")
        wm.put_content("Python 支持面向对象")
        wm.put_content("今天天气晴朗")

        count = pattern_consolidation(wm, ltm)
        assert count == 3
        assert ltm.stats()["nodes"] >= 3
