<div align="center">

# Agentic OS Core

**A hardcore memory & planning engine for AI Agents**

Graph-based knowledge storage · MCTS tree-of-thoughts reasoning · Hierarchical goal planning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/xuwenyao/agentic-os-core/actions/workflows/ci.yml/badge.svg)](https://github.com/xuwenyao/agentic-os-core/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-330%20passed-brightgreen.svg)](./tests)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](./tests)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](./docs)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](./pyproject.toml)

[English](#overview) · [中文](#概述)

</div>

---

## Overview

Current Agent frameworks (LangChain, AutoGen) treat memory as a flat vector store — append-only, no structure, no reasoning over past experiences. **Agentic OS Core** takes a different approach: it provides **low-level data structure primitives** purpose-built for Agent cognition.

Instead of dumping everything into a vector DB, you get:

- **Knowledge Graph** — memories as typed nodes with weighted, labeled edges (causal, temporal, associative). Supports PageRank importance scoring, Dijkstra shortest-path, topological sort, and subgraph extraction.
- **MCTS Tree-of-Thoughts** — a full Monte Carlo Tree Search engine with UCB1 selection, configurable exploration/exploitation, and multiple pruning strategies (score threshold, depth limit, redundancy removal).
- **Hierarchical Memory** — a working memory (LRU, O(1) ops) that consolidates into a long-term graph store, with pluggable consolidation strategies (simple, importance-filtered, pattern-extracting).
- **Goal Planner** — decomposes goals into sub-goals, generates topologically-sorted execution plans, evaluates plans with ToT, and supports retry/replanning on failure.
- **Persistent Storage** — SQLite built-in for production use, Redis optional for distributed deployments.
- **Vector Similarity** — NumPy-based cosine similarity built-in, with adapters for FAISS/Milvus/ChromaDB.
- **Multi-Agent Shared Memory** — namespace-isolated shared knowledge graph with agent-level access control.
- **Interactive Visualization** — D3.js-powered HTML visualizations for knowledge graphs and thought trees.

**Zero external core dependencies.** Pure Python 3.10+, using only `dataclasses`, `collections`, `heapq`, and `json`.

## 概述

当前 Agent 框架的记忆管理大多是简单的向量数据库拼接，缺乏结构化推理能力。**Agentic OS Core** 提供了专为 Agent 认知设计的底层数据结构原语：

- **知识图谱** — 邻接表 + 倒排索引，支持 PageRank、Dijkstra、拓扑排序
- **MCTS 思维树** — 完整的蒙特卡洛树搜索引擎，UCB1 + 多种剪枝策略
- **层级记忆** — LRU 工作记忆 → 图存储长期记忆，可插拔巩固策略
- **目标规划器** — 目标分解、拓扑排序执行计划、失败重试/重规划
- **持久化存储** — 内置 SQLite，可选 Redis 分布式后端
- **向量检索** — 内置 NumPy 余弦相似度，支持 FAISS/Milvus/ChromaDB
- **多 Agent 共享记忆** — 命名空间隔离的共享知识图谱
- **交互式可视化** — 基于 D3.js 的知识图谱和思维树 HTML 可视化

**零核心外部依赖**，纯 Python 3.10+ 标准库实现。

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       Agent Loop                          │
│   Perceive → Remember → Plan → Execute → Reflect         │
└───────────┬──────────┬──────────┬────────────────────────┘
            │          │          │
  ┌─────────▼──┐  ┌────▼────┐  ┌─▼──────────┐
  │  Memory    │  │ Planning│  │  Plugins    │
  │  Manager   │  │ Engine  │  │  (LLM I/O)  │
  ├────────────┤  ├─────────┤  └─────────────┘
  │ Working    │  │ Goals   │
  │ Memory(LRU)│  │ Planner │
  │     ↕      │  │ Executor│
  │ Long-Term  │  └────┬────┘
  │ Memory(KG) │       │
  └─────┬──────┘  ┌────▼────┐
        │         │   MCTS  │
  ┌─────▼──────┐  │ Engine  │     ┌───────────────┐
  │ Knowledge  │  │ (ToT)   │     │  Extensions    │
  │ Graph      │  └─────────┘     ├───────────────┤
  │ ┌────────┐ │                  │  LangChain     │
  │ │ Nodes  │ │  Node Types:     │  LlamaIndex    │
  │ │ Edges  │ │  Episode/Fact/   │  Visualization │
  │ │ Index  │ │  Reflection/Goal │  Shared Memory │
  │ └────────┘ │                  │  Vector Store  │
  └────────────┘                  └───────────────┘
        ↕
  ┌────────────┐
  │  Storage   │  SQLite (built-in) / Redis (optional)
  └────────────┘
```

---

## Installation

```bash
pip install agentic-os-core
```

Optional extras:

```bash
pip install agentic-os-core[numpy]       # Vector similarity search
pip install agentic-os-core[langchain]    # LangChain integration
pip install agentic-os-core[llamaindex]   # LlamaIndex integration
pip install agentic-os-core[dev]          # Test + lint tools
```

Or from source:

```bash
git clone https://github.com/xuwenyao/agentic-os-core.git
cd agentic-os-core
pip install -e ".[dev]"
```

## Quick Start

### 1. Knowledge Graph

```python
from agentic_os import KnowledgeGraph, EdgeType, create_episode, create_fact, create_reflection

kg = KnowledgeGraph()

# Add typed memory nodes
e1 = create_episode("用户询问了 Python 装饰器")
f1 = create_fact("装饰器是高阶函数的语法糖", importance=0.8)
r1 = create_reflection("理解闭包是掌握装饰器的关键", importance=0.9)

for node in [e1, f1, r1]:
    kg.add_node(node)

# Link with typed, weighted edges
kg.add_edge(e1.id, f1.id, EdgeType.DERIVED_FROM, weight=0.8)
kg.add_edge(f1.id, r1.id, EdgeType.DERIVED_FROM, weight=0.9)

# Query
results = kg.find_nodes("装饰器")  # substring-aware search
path = kg.subgraph(e1.id, depth=2)  # extract neighborhood
```

### 2. MCTS Reasoning

```python
from agentic_os import MCTS

mcts = MCTS(exploration_weight=1.414, max_iterations=200, max_depth=4)

def generator(state):
    return ["方案A: 优化数据库", "方案B: 增加缓存", "方案C: 升级硬件"]

def evaluator(state, thought):
    return {"方案A": 0.85, "方案B": 0.9, "方案C": 0.5}.get(thought[:3], 0.3)

tree = mcts.search("如何提升系统性能?", {}, generator, evaluator)
best_path = tree.get_best_path()
print(tree.visualize())  # ASCII tree visualization
```

### 3. Memory Manager

```python
from agentic_os import MemoryManager, EdgeType

memory = MemoryManager(working_capacity=50)

# Store experiences
memory.add_experience("学习了 MCTS 算法")
memory.add_experience("MCTS 用于决策优化")

# Consolidate short-term → long-term
memory.consolidate()

# Recall with multiple strategies
results = memory.recall("MCTS")                     # keyword search
important = memory.recall_by_importance(top_k=5)     # PageRank-ranked
recent = memory.recall_by_recency(top_k=5)           # timeline

# Reflect on experiences → generate insight nodes
memory.reflect()

# Build causal chains
f1 = memory.store_fact("索引缺失导致查询慢")
f2 = memory.store_fact("添加索引后查询快了 10 倍")
memory.link_memories(f1, f2, EdgeType.CAUSAL, weight=0.95)
```

### 4. Goal Planning

```python
from agentic_os import Planner, Executor, GoalPriority, create_planning_goal

planner = Planner()
root = create_planning_goal("构建 Web 服务", priority=GoalPriority.HIGH)
planner.add_goal(root)

# Decompose
planner.decompose(root, ["设计 API", "实现认证", "编写测试", "部署"])

# Generate execution plan (topologically sorted)
plan = planner.create_plan(root)

# Execute
executor = Executor(planner)
logs = executor.execute_plan(plan, lambda g: (True, f"Done: {g.description}"))
print(f"Success: {executor.success_count}, Failed: {executor.failure_count}")
```

### 5. Persistent Storage

```python
from agentic_os import KnowledgeGraph, create_fact
from agentic_os.core.storage import SqliteStorage

kg = KnowledgeGraph()
kg.add_node(create_fact("Python is great", importance=0.9))

# Save to SQLite
with SqliteStorage("my_agent.db") as store:
    store.save_graph(kg)

# Load later
with SqliteStorage("my_agent.db") as store:
    loaded = store.load_graph()
```

### 6. Multi-Agent Shared Memory

```python
from agentic_os import SharedMemoryGraph

shared = SharedMemoryGraph()
alice = shared.register_agent("alice")
bob = shared.register_agent("bob")

# Alice stores and shares knowledge
nid = alice.store_fact("Redis supports pub/sub")
alice.share_with(nid, "bob")

# Bob retrieves shared knowledge
bob_knowledge = bob.retrieve("Redis", include_shared=True)
```

---

## Core Concepts

### Node Types

| Type | Purpose | Auto-prefix |
|------|---------|-------------|
| `EPISODE` | Raw experience / interaction | `ep_` |
| `FACT` | Extracted knowledge | `fact_` |
| `REFLECTION` | Agent's insight / synthesis | `ref_` |
| `GOAL` | Objective / intent | `goal_` |

### Edge Types

| Type | Semantics |
|------|-----------|
| `CAUSAL` | A caused B |
| `TEMPORAL` | A happened before B |
| `ASSOCIATIVE` | A is related to B |
| `DERIVED_FROM` | B was derived from A |
| `SUPPORTS` | A supports B |
| `CONTRADICTS` | A contradicts B |

### Retrieval Strategies

| Strategy | Description |
|----------|-------------|
| `KEYWORD` | Substring-aware content search |
| `IMPORTANCE` | PageRank-scored ranking |
| `RECENCY` | Timeline-ordered |
| `ASSOCIATION` | Graph traversal from seed node |

### MCTS Phases

1. **Selection** — traverse tree using UCB1: `Q/N + c·√(ln(N_parent)/N)`
2. **Expansion** — add candidate thoughts as children
3. **Simulation** — evaluate new node quality
4. **Backpropagation** — update scores from leaf to root

---

## Performance

Benchmarks on Apple M1, Python 3.13:

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Add node | O(1) | HashMap + index update |
| Add/remove edge | O(1) | Adjacency list |
| Keyword search | O(n) | Linear scan with early exit |
| BFS/DFS traversal | O(V+E) | Standard graph traversal |
| Shortest path | O((V+E) log V) | Dijkstra with binary heap |
| PageRank | O(k·(V+E)) | k iterations, typically 10-20 |
| MCTS iteration | O(depth) | Per iteration |
| Working memory put/get | O(1) | OrderedDict LRU |
| Consolidation | O(n) | n = working memory size |

Run your own benchmarks:

```bash
python benchmarks/run_benchmarks.py
```

---

## Project Structure

```
src/agentic_os/
├── core/
│   ├── graph/          # Knowledge Graph (adjacency list + inverted index)
│   │   ├── node.py     # MemoryNode with 4 types
│   │   ├── edge.py     # Edge with 6 types + weight
│   │   ├── knowledge_graph.py  # Core graph with CRUD + serialization
│   │   ├── traversal.py        # BFS, DFS, Dijkstra, topo-sort
│   │   └── scoring.py          # PageRank, decay, association
│   ├── tree/           # Tree of Thoughts + MCTS
│   │   ├── thought_node.py     # ThoughtNode with UCB1 stats
│   │   ├── thought_tree.py     # Tree management + visualization
│   │   ├── mcts.py             # Full MCTS implementation
│   │   ├── async_mcts.py       # Async MCTS with concurrent rollouts
│   │   ├── search.py           # UCB1, Best-First, Beam Search
│   │   └── pruning.py          # Score/depth/redundancy pruning
│   ├── memory/         # Memory Management
│   │   ├── working.py          # LRU short-term memory
│   │   ├── longterm.py         # Graph-backed long-term memory
│   │   ├── manager.py          # Unified memory orchestrator
│   │   ├── consolidation.py    # 3 consolidation strategies
│   │   └── shared.py           # Multi-agent shared memory graph
│   ├── planning/       # Planning Engine
│   │   ├── goal.py             # Goal with priority + state machine
│   │   ├── planner.py          # Decomposition + ToT evaluation
│   │   └── executor.py         # Execution with retry + logging
│   ├── storage/        # Persistent Storage
│   │   ├── base.py             # StorageBackend ABC
│   │   ├── sqlite_backend.py   # SQLite storage (built-in)
│   │   └── redis_backend.py    # Redis storage (optional)
│   └── vector/         # Vector Similarity
│       ├── base.py             # VectorStore ABC + SearchResult
│       └── numpy_backend.py    # NumPy cosine similarity (built-in)
├── ext/                # Extension Adapters
│   ├── langchain/      # LangChain integration
│   │   ├── memory.py           # AgenticOSMemory
│   │   ├── retriever.py        # AgenticOSRetriever
│   │   └── tool.py             # AgenticOSGraphTool
│   ├── llamaindex/     # LlamaIndex integration
│   │   └── vector_store.py     # AgenticOSVectorStore
│   └── visualization/  # Interactive HTML Visualization
│       ├── graph_visualizer.py # D3.js knowledge graph
│       └── tree_visualizer.py  # D3.js thought tree
├── plugins/            # Plugin Interface
│   ├── base.py                 # Abstract interfaces (LLM, Evaluator, Action)
│   └── mock.py                 # Mock implementations for testing
└── utils/              # Utilities
    ├── hashing.py              # FNV-1a content hashing
    └── serialization.py        # JSON serialization protocol
```

---

## Plugin Integration

Agentic OS Core is designed to be plugged into any LLM framework. Implement the abstract interfaces:

```python
from agentic_os.plugins.base import LLMBackend, Evaluator

class MyLLM(LLMBackend):
    def generate(self, prompt, **kwargs):
        # Call your LLM API here
        return call_api(prompt)

    def embed(self, text):
        # Return embedding vector
        return get_embedding(text)

class MyEvaluator(Evaluator):
    def evaluate(self, state, thought):
        # Score a thought given state, return [0, 1]
        return score
```

### LangChain Integration

```python
from agentic_os.ext.langchain import AgenticOSMemory, AgenticOSRetriever, AgenticOSGraphTool

memory = AgenticOSMemory(memory_manager=my_mm)
retriever = AgenticOSRetriever(memory_manager=my_mm)
tool = AgenticOSGraphTool(memory_manager=my_mm)
```

### LlamaIndex Integration

```python
from agentic_os.ext.llamaindex import AgenticOSVectorStore

vector_store = AgenticOSVectorStore(wrapped_store=my_numpy_store)
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

[MIT License](./LICENSE)

## Roadmap

- [x] Persistent storage backends (SQLite, Redis)
- [x] Vector similarity retrieval (NumPy built-in, FAISS/Milvus/ChromaDB adapters)
- [x] Async API for concurrent MCTS rollouts
- [x] LangChain / LlamaIndex integration adapters
- [x] Visualization tools (interactive D3.js HTML)
- [x] Multi-agent shared memory graph
- [ ] Temporal decay with configurable curves
- [ ] Streaming event system for real-time Agent coordination
