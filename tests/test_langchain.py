"""LangChain 适配器测试

所有测试均不依赖 LangChain 安装，直接测试核心逻辑。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agentic_os.core.memory.manager import MemoryManager
from agentic_os.core.vector.base import SearchResult
from agentic_os.ext.langchain.memory import AgenticOSMemory
from agentic_os.ext.langchain.retriever import AgenticOSRetriever
from agentic_os.ext.langchain.tool import AgenticOSGraphTool


# ---------------------------------------------------------------------------
# AgenticOSMemory
# ---------------------------------------------------------------------------
class TestAgenticOSMemory:
    def test_memory_variables(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm, memory_key="chat_history")
        assert memory.memory_variables == ["chat_history"]

    def test_memory_variables_default_key(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)
        assert memory.memory_variables == ["history"]

    def test_save_context(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        memory.save_context(
            {"input": "什么是 MCTS?"},
            {"output": "MCTS 是蒙特卡洛树搜索算法"},
        )

        # 工作记忆中应有两条记录
        assert mm.working.size == 2

    def test_save_context_user_only(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        memory.save_context({"input": "你好"}, {})

        assert mm.working.size == 1

    def test_save_context_ai_only(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        memory.save_context({}, {"output": "你好!"})

        assert mm.working.size == 1

    def test_save_context_empty(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        memory.save_context({}, {})

        assert mm.working.size == 0

    def test_load_memory_variables(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        mm.add_experience("今天学习了 Python 装饰器")
        mm.add_experience("明天要学习 Rust")

        result = memory.load_memory_variables({"history": "Python"})
        assert "Python" in result["history"]

    def test_load_memory_variables_fallback_input_key(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        mm.add_experience("图论是计算机科学的基础")

        result = memory.load_memory_variables({"input": "图论"})
        assert "图论" in result["history"]

    def test_load_memory_variables_empty_query(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        mm.add_experience("某些内容")

        result = memory.load_memory_variables({})
        assert result["history"] == ""

    def test_clear(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        mm.add_experience("内容A")
        mm.add_experience("内容B")
        assert mm.working.size == 2

        memory.clear()
        assert mm.working.size == 0

    def test_clear_preserves_longterm(self):
        mm = MemoryManager()
        memory = AgenticOSMemory(mm)

        mm.store_fact("长期事实", importance=0.9)
        mm.add_experience("短期记忆")

        memory.clear()
        assert mm.working.size == 0
        assert mm.longterm.graph.node_count == 1


# ---------------------------------------------------------------------------
# AgenticOSRetriever
# ---------------------------------------------------------------------------
class TestAgenticOSRetriever:
    def test_invoke(self):
        # Mock VectorStore
        store = MagicMock()
        store.search.return_value = [
            SearchResult(
                id="doc1",
                score=0.95,
                metadata={"content": "MCTS 是一种搜索算法"},
            ),
            SearchResult(
                id="doc2",
                score=0.80,
                metadata={"content": "AlphaGo 使用了 MCTS"},
            ),
        ]

        def embed_fn(text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        retriever = AgenticOSRetriever(store, embed_fn, top_k=3)
        docs = retriever.invoke("MCTS 搜索算法")

        assert len(docs) == 2
        assert docs[0]["page_content"] == "MCTS 是一种搜索算法"
        assert docs[0]["metadata"]["id"] == "doc1"
        assert docs[0]["metadata"]["score"] == 0.95

    def test_get_relevant_documents(self):
        store = MagicMock()
        store.search.return_value = [
            SearchResult(
                id="r1",
                score=0.9,
                metadata={"content": "结果文本", "tag": "test"},
            ),
        ]

        def embed_fn(text: str) -> list[float]:
            return [1.0]

        retriever = AgenticOSRetriever(store, embed_fn, top_k=5)
        docs = retriever._get_relevant_documents("query")

        assert len(docs) == 1
        assert docs[0]["page_content"] == "结果文本"
        assert docs[0]["metadata"]["tag"] == "test"

    def test_invoke_empty_results(self):
        store = MagicMock()
        store.search.return_value = []

        def embed_fn(text: str) -> list[float]:
            return []

        retriever = AgenticOSRetriever(store, embed_fn)
        docs = retriever.invoke("空查询")

        assert docs == []

    def test_embedder_called_with_query(self):
        store = MagicMock()
        store.search.return_value = []

        embedder = MagicMock(return_value=[0.5, 0.6])

        retriever = AgenticOSRetriever(store, embedder, top_k=3)
        retriever.invoke("测试查询")

        embedder.assert_called_once_with("测试查询")
        store.search.assert_called_once_with([0.5, 0.6], top_k=3)


# ---------------------------------------------------------------------------
# AgenticOSGraphTool
# ---------------------------------------------------------------------------
class TestAgenticOSGraphTool:
    def test_default_name_and_description(self):
        mm = MemoryManager()
        tool = AgenticOSGraphTool(mm)

        assert tool.name == "agentic_graph_query"
        assert "knowledge graph" in tool.description.lower()

    def test_custom_name_and_description(self):
        mm = MemoryManager()
        tool = AgenticOSGraphTool(
            mm,
            name="my_tool",
            description="Custom tool description",
        )

        assert tool.name == "my_tool"
        assert tool.description == "Custom tool description"

    def test_run_returns_json(self):
        mm = MemoryManager()
        mm.add_experience("部署了 v2.0 到生产环境")
        mm.store_fact("生产环境运行在 k8s 上", importance=0.8)

        tool = AgenticOSGraphTool(mm)
        result = tool._run("生产环境")

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        assert any("生产环境" in item["content"] for item in parsed)

    def test_run_result_structure(self):
        mm = MemoryManager()
        mm.store_fact("测试事实", importance=0.7)

        tool = AgenticOSGraphTool(mm)
        result = tool._run("测试")
        parsed = json.loads(result)

        assert len(parsed) >= 1
        item = parsed[0]
        assert "id" in item
        assert "type" in item
        assert "content" in item
        assert "importance" in item

    def test_run_no_results(self):
        mm = MemoryManager()
        tool = AgenticOSGraphTool(mm)
        result = tool._run("不存在的查询")

        parsed = json.loads(result)
        assert parsed == []

    def test_run_result_is_valid_json_utf8(self):
        mm = MemoryManager()
        mm.add_experience("中文内容测试：图论算法、MCTS 搜索")

        tool = AgenticOSGraphTool(mm)
        result = tool._run("图论")

        # 确保是合法 JSON 且中文不转义
        assert "\\u" not in result
        parsed = json.loads(result)
        assert any("图论" in item["content"] for item in parsed)
