# Agentic OS Core v2 — 扩展设计文档

> 日期: 2026-05-21 | 版本: 0.2.0

## 概述

在 v0.1.0 核心引擎基础上，分 4 个阶段扩展：发布准备 → 存储检索基础设施 → 能力扩展 → 生态集成。

**核心原则：**
- 核心包保持零外部依赖，所有新依赖通过 `[extra]` 可选安装
- 新模块放在 `src/agentic_os/ext/` 下（生态集成），存储和向量放在 `core/` 下
- 每个阶段独立可测试、独立可发布

---

## 阶段 0：发布准备

### 0.1 首次 commit + push
- `git add` 所有文件，创建初始 commit
- 创建 GitHub 远程仓库并 push
- 创建 `v0.1.0` tag

### 0.2 覆盖率补全
- `mcts.py` (91%) — 补测试：max_depth 限制、空候选、全部候选失败
- `traversal.py` (94%) — 补测试：空图、单节点、不连通图、权重为 0 的边
- `hashing.py` (94%) — 补测试：空字符串、超长字符串、二进制输入

### 0.3 Examples 验证
- 运行并修复 `04_custom_evaluator.py`
- 运行并修复 `05_persistence_and_restoration.py`
- 运行并修复 `06_multi_turn_memory.py`

---

## 阶段 1：存储与检索基础设施

### 1a. 持久化存储后端

**文件结构：**
```
src/agentic_os/core/storage/
├── __init__.py
├── base.py              # StorageBackend 抽象基类
├── sqlite_backend.py    # SQLite 实现
└── redis_backend.py     # Redis 实现（仅当 redis 可用时导入）
```

**StorageBackend 接口：**
```python
class StorageBackend(ABC):
    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> None: ...
    @abstractmethod
    def load_graph(self) -> KnowledgeGraph | None: ...
    @abstractmethod
    def save_node(self, node: MemoryNode) -> None: ...
    @abstractmethod
    def load_node(self, node_id: str) -> MemoryNode | None: ...
    @abstractmethod
    def delete_node(self, node_id: str) -> bool: ...
    @abstractmethod
    def query_nodes(self, filter_fn: Callable) -> list[MemoryNode]: ...
    @abstractmethod
    def close(self) -> None: ...
```

**SQLite 后端：**
- 3 张表：`nodes`（id, type, content JSON, importance, access_count, timestamps, metadata JSON）、`edges`（source_id, target_id, type, weight, metadata JSON）、`metadata`（key, value）
- 关键词索引用 JSON 数组存储在 nodes 表的 keywords 列
- 支持上下文管理器 `with SqliteStorage("path.db") as store:`
- 线程安全（每个操作用 `threading.Lock` 保护）

**Redis 后端：**
- 节点：Hash `agentic:node:{id}` — 存储序列化 JSON
- 出边：Set `agentic:edges:out:{src_id}` — 存储目标 ID
- 入边：Set `agentic:edges:in:{tgt_id}` — 存储源 ID
- 元数据：Hash `agentic:meta`
- 需要 `redis>=4.0`，通过 `[redis]` extra 安装
- 使用 lazy import，不可用时抛出 `ImportError`

### 1b. 向量相似度检索

**文件结构：**
```
src/agentic_os/core/vector/
├── __init__.py
├── base.py              # VectorStore 抽象基类 + SearchResult
├── numpy_backend.py     # NumPy 内置实现
├── faiss_backend.py     # FAISS 实现（可选）
├── milvus_backend.py    # Milvus 实现（可选）
└── chroma_backend.py    # Chroma 实现（可选）
```

**VectorStore 接口：**
```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: list[float], metadata: dict) -> None: ...
    @abstractmethod
    def add_batch(self, items: list[tuple[str, list[float], dict]]) -> None: ...
    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]: ...
    @abstractmethod
    def delete(self, id: str) -> bool: ...
    @abstractmethod
    def count(self) -> int: ...

@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
```

**NumPy 后端：**
- 向量存储在 `dict[str, np.ndarray]`，元数据存储在 `dict[str, dict]`
- 检索：余弦相似度线性扫描 `cos_sim = dot(q, v) / (norm(q) * norm(v))`
- 持久化：`np.savez()` 保存向量 + JSON 保存元数据
- 需要 `numpy>=1.24`，通过 `[numpy]` extra 安装

**FAISS 后端：**
- 用 `faiss.IndexFlatIP`（内积索引）
- 支持 `IndexIVFFlat` 加速大规模检索
- 需要 `faiss-cpu>=1.7`，通过 `[faiss]` extra 安装

**Milvus 后端：**
- 连接 Milvus 实例，创建 Collection
- 支持 ANN 搜索
- 需要 `pymilvus>=2.3`，通过 `[milvus]` extra 安装

**Chroma 后端：**
- 用 Chroma Collection 管理
- 需要通过 `[chroma]` 额外安装

**集成到 LongTermMemory：**
```python
class LongTermMemory:
    def __init__(self, vector_store: VectorStore | None = None,
                 embedder: Callable[[str], list[float]] | None = None): ...
    def retrieve_similar(self, query: str, top_k: int = 10) -> list[MemoryNode]: ...
    def index_all(self) -> int: ...  # 对所有节点建立向量索引
```

---

## 阶段 2：能力扩展

### 2a. 异步 MCTS

**新文件：** `src/agentic_os/core/tree/async_mcts.py`

```python
class AsyncMCTS:
    def __init__(self, exploration_weight: float = 1.414,
                 max_iterations: int = 1000,
                 max_depth: int = 10,
                 concurrency: int = 8): ...

    async def search(self, root_thought: str, root_state: dict,
                     async_generator: AsyncCandidateGenerator,
                     async_evaluator: AsyncEvaluator) -> ThoughtTree: ...
```

**关键设计：**
- `_expand_batch` 并发生成多个子节点：用 `asyncio.gather` 同时评估所有候选项
- `_simulate_batch` 并发 rollout：同时评估多个叶节点
- `concurrency` 参数控制 `asyncio.Semaphore` 并发上限
- `AsyncEvaluator = Callable[[dict, str], Awaitable[float]]`
- `AsyncCandidateGenerator = Callable[[dict], Awaitable[list[str]]]`

### 2b. 可视化工具

**文件结构：**
```
src/agentic_os/ext/visualization/
├── __init__.py
├── graph_visualizer.py
├── tree_visualizer.py
└── templates/
    ├── graph.html      # D3.js 力导向图
    └── tree.html       # D3.js 树形布局
```

**KnowledgeGraphVisualizer：**
- `to_html(graph, output_path, title="Knowledge Graph")` — 生成交互式 HTML
- D3.js force-directed graph 布局
- 节点：颜色 = NodeType（4种），大小 = importance * scale
- 边：颜色 = EdgeType（6种），粗细 = weight
- 交互：点击节点显示详情面板、缩放、拖拽、搜索框过滤
- CDN：`https://cdn.jsdelivr.net/npm/d3@7`

**ThoughtTreeVisualizer：**
- `to_html(tree, output_path, title="Thought Tree")` — 生成树形 HTML
- D3.js tree layout（自上而下）
- 节点：大小 = log(visits + 1)，颜色 = avg_score 渐变（红→绿）
- 最佳路径高亮
- 悬停显示 thought text、score、visits、UCB1

---

## 阶段 3：生态集成

### 3a. LangChain 适配器

**文件结构：**
```
src/agentic_os/ext/langchain/
├── __init__.py
├── memory.py       # AgenticOSMemory
├── retriever.py    # AgenticOSRetriever
└── tool.py         # AgenticOSGraphTool
```

**适配器：**
- `AgenticOSMemory(BaseMemory)` — 包装 MemoryManager
  - `save_context(inputs, outputs)` → `memory.add_episode(outputs)`
  - `load_memory_variables(inputs)` → `memory.recall(inputs["query"])`
  - `clear()` → 清空工作记忆
- `AgenticOSRetriever(VectorStoreRetriever)` — 包装 VectorStore
  - `_get_relevant_documents(query)` → 向量检索
- `AgenticOSGraphTool(BaseTool)` — 图谱查询工具
  - `_run(query)` → `memory.recall(query)`

### 3b. LlamaIndex 适配器

**文件结构：**
```
src/agentic_os/ext/llamaindex/
├── __init__.py
└── vector_store.py     # AgenticOSVectorStore
```

- 实现 LlamaIndex `VectorStore` 接口
- 委托给内部 `VectorStore` 实例

### 3c. 多 Agent 共享记忆

**新文件：** `src/agentic_os/core/memory/shared.py`

```python
class SharedMemoryGraph:
    def __init__(self):
        self._graph = KnowledgeGraph()
        self._namespaces: dict[str, set[str]]   # agent_id → node_ids
        self._lock = threading.RLock()

    def register_agent(self, agent_id: str) -> AgentMemoryHandle: ...
    def unregister_agent(self, agent_id: str) -> None: ...
    def query_shared(self, query: str, top_k: int = 10,
                     exclude_agent: str | None = None) -> list[MemoryNode]: ...
    def list_agents(self) -> list[str]: ...
    def stats(self) -> dict[str, Any]: ...

class AgentMemoryHandle:
    def store(self, node: MemoryNode) -> str: ...
    def retrieve(self, query: str, top_k: int = 10,
                 include_shared: bool = True) -> list[MemoryNode]: ...
    def share_with(self, node_id: str, target_agent_id: str) -> None: ...
    def get_shared_from(self, source_agent_id: str) -> list[MemoryNode]: ...
```

---

## pyproject.toml 更新

```toml
[project.optional-dependencies]
dev = [...]
redis = ["redis>=4.0"]
numpy = ["numpy>=1.24"]
faiss = ["faiss-cpu>=1.7"]
milvus = ["pymilvus>=2.3"]
chroma = ["chromadb>=0.4"]
langchain = ["langchain>=0.1", "langchain-core>=0.1"]
llamaindex = ["llama-index>=0.10"]
all = ["agentic-os-core[redis,numpy,faiss,milvus,chroma,langchain,llamaindex]"]
```

---

## 公共 API 导出

新增 `__all__` 导出（按阶段）：

**阶段 1：** `StorageBackend`, `SqliteStorage`, `RedisStorage`, `VectorStore`, `SearchResult`, `NumpyVectorStore`, `FaissVectorStore`
**阶段 2：** `AsyncMCTS`, `KnowledgeGraphVisualizer`, `ThoughtTreeVisualizer`
**阶段 3：** `SharedMemoryGraph`, `AgentMemoryHandle`
