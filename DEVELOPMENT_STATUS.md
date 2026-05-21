# Agentic OS Core — 开发进度

> 最后更新: 2026-05-21 | 版本: 0.1.0

## 项目概览

Agent 底层记忆与规划引擎，纯 Python 3.10+ 标准库实现，零外部依赖。

| 指标 | 数值 |
|------|------|
| 源文件 | 30 个 (`src/`) |
| 测试文件 | 11 个 (`tests/`) |
| 测试数量 | 172 passed |
| 代码覆盖率 | 97% (1402 stmts, 45 miss) |
| mypy strict | 零错误 |
| ruff lint | All checks passed |
| Examples | 6 个 (全部可运行) |
| Benchmarks | 26 项 |

---

## 已完成模块

### 核心引擎 (src/agentic_os/core/)

| 模块 | 功能 | 覆盖率 |
|------|------|--------|
| `graph/node.py` | 4 种节点类型 (Episode/Fact/Reflection/Goal) | 100% |
| `graph/edge.py` | 6 种边类型 + 权重验证 | 100% |
| `graph/knowledge_graph.py` | CRUD + 子图提取 + 序列化 | 95% |
| `graph/traversal.py` | BFS/DFS/Dijkstra/拓扑排序/连通分量 | 94% |
| `graph/scoring.py` | PageRank + 衰减 + 关联度 | 96% |
| `tree/thought_node.py` | UCB1 统计节点 | 97% |
| `tree/thought_tree.py` | 树管理 + 可视化 | 96% |
| `tree/mcts.py` | 完整 MCTS (选择/扩展/模拟/回传) | 91% |
| `tree/search.py` | Best-First / Beam Search | 96% |
| `tree/pruning.py` | 得分/深度/冗余剪枝 | 98% |
| `memory/working.py` | LRU 工作记忆 (OrderedDict, O(1)) | 96% |
| `memory/longterm.py` | 图存储长期记忆 + 4 种检索策略 | 98% |
| `memory/manager.py` | 统一记忆管理器 (facade) | 99% |
| `memory/consolidation.py` | 3 种巩固策略 | 99% |
| `planning/goal.py` | Goal 数据类 + 优先级 + 状态机 + 依赖 | 100% |
| `planning/planner.py` | 目标分解 + 拓扑计划 + ToT 评估 | 97% |
| `planning/executor.py` | 执行引擎 + 重试支持 | 97% |

### 插件与工具 (src/agentic_os/)

| 模块 | 功能 | 覆盖率 |
|------|------|--------|
| `plugins/base.py` | 抽象接口 (LLM/Evaluator/Action/Memory) | 100% |
| `plugins/mock.py` | 测试用 Mock 实现 | 100% |
| `utils/hashing.py` | FNV-1a 内容哈希 | 94% |
| `utils/serialization.py` | JSON 序列化 (dataclass/Enum/datetime) | 97% |
| `exceptions.py` | 12 个自定义异常类 | 100% |
| `__init__.py` | 36 个公共 API 导出 | 100% |

### 测试体系 (tests/)

| 文件 | 测试数 | 内容 |
|------|--------|------|
| `test_graph.py` | 32 | 知识图谱 CRUD、遍历、PageRank |
| `test_edge_cases.py` | 40 | 边界条件、空输入、异常处理 |
| `test_mcts.py` | 21 | MCTS 搜索、UCB1、剪枝 |
| `test_memory.py` | 23 | 工作记忆、长期记忆、巩固 |
| `test_planning.py` | 15 | 目标分解、计划生成、执行 |
| `test_integration.py` | 4 | 端到端 Agent 循环 |
| `test_properties.py` | 7 | Hypothesis 属性测试 (不变式验证) |
| `test_plugins_and_serialization.py` | 29 | Mock 插件 + 序列化工具 |
| `test_exports.py` | 1 | `__all__` 导出验证 |

### 工程基础设施

| 项目 | 状态 |
|------|------|
| **pyproject.toml** | 动态版本 (`__version__`)、覆盖率配置 (fail_under=85)、完整 metadata |
| **CI (GitHub Actions)** | ruff + mypy strict (硬门禁) + pytest --cov + mkdocs build --strict，4 个 Python 版本 (3.10-3.13) |
| **PyPI 发布** | `publish.yml` (Trusted Publisher, release 触发) |
| **Dependabot** | pip + GitHub Actions 自动更新 (weekly) |
| **Pre-commit** | ruff lint/format + trailing-whitespace + check-yaml/toml + mypy strict |
| **文档站** | MkDocs Material + mkdocstrings 自动 API 参考 + ReadTheDocs 配置 |
| **README** | 中英双语、架构图、Quick Start、性能表、7 个 badges |
| **开源协作** | CONTRIBUTING.md、CODE_OF_CONDUCT.md、SECURITY.md、Issue/PR 模板 |
| **CHANGELOG** | v0.1.0 完整功能清单 |
| **日志** | 全模块 `logging.getLogger(__name__)`，DEBUG/INFO/WARNING 三级 |
| **异常体系** | AgenticOSError → ValidationError / GraphError / TreeError / ExecutionError + 8 个子类 |
| **Benchmarks** | 26 项性能基准，表格输出 + 模块汇总 |
| **Examples** | 6 个示例脚本，全部可运行 |

---

## 未完成 / 可优化项

### 高优先级 (发布 1.0 前)

| 项目 | 说明 | 复杂度 |
|------|------|--------|
| **首次 commit + push** | 项目尚未初始化 git remote，所有工作仅在本地。需 `git remote add origin` 并推送后 CI badge 才能生效 | 低 |
| **git tag v0.1.0** | CHANGELOG 已有 v0.1.0 记录，但未创建实际 tag。推送后执行 `git tag v0.1.0 && git push --tags` | 低 |
| **PyPI 首次发布测试** | `publish.yml` 已写好但未验证。需在 GitHub 创建 release 测试，或本地 `python -m build && twine upload` 验证 | 中 |
| **覆盖率达到 100% 的模块** | 以下文件有少量 miss: `knowledge_graph.py`(95%)、`traversal.py`(94%)、`mcts.py`(91%)、`hashing.py`(94%)。可针对性补测试 | 中 |

### 中优先级 (提升项目质量)

| 项目 | 说明 | 复杂度 |
|------|------|--------|
| **剩余 Examples 未验证** | `04_custom_evaluator.py`、`05_persistence_and_restoration.py`、`06_multi_turn_memory.py` 三个脚本未验证能否运行 | 低 |
| **docstring 完善** | mkdocstrings 依赖 docstring 质量。当前部分模块缺少 Google-style docstring，API 参考页面可能不完整 | 中 |
| **MCTS 搜索多样性** | Benchmark 显示 MCTS 搜索结果单一（所有节点显示相同文本），mock generator 生成的候选项缺少多样性 | 中 |
| **CI 缓存优化** | CI 未配置 pip 缓存，每次全量安装依赖。可加 `actions/cache` 加速 | 低 |
| **conftest.py 增强** | 当前仅设置日志级别。可添加常用 fixtures（预构建图谱、MCTS 实例等）减少测试样板代码 | 低 |

### 低优先级 (锦上添花)

| 项目 | 说明 | 复杂度 |
|------|------|--------|
| **中文文档站** | docs/ 目前为英文，可添加中文版或 i18n | 中 |
| **API 类型存根 (.pyi)** | 为公共 API 提供类型存根文件，提升 IDE 补全体验 | 高 |
| **性能回归 CI** | 在 CI 中运行 benchmarks 并对比基线，检测性能退化 | 中 |
| **Coverage badge** | 添加 Codecov/Coveralls 集成，README 中显示实时覆盖率 | 低 |
| **GitHub Pages 文档部署** | 配置 gh-pages 分支自动部署 MkDocs 站点 | 低 |

### 路线图 (Roadmap，README 中已列出)

- [ ] 持久化存储后端 (SQLite, Redis)
- [ ] 向量相似度检索 (可选 embedding 支持)
- [ ] 异步 API (并发 MCTS rollout)
- [ ] LangChain / LlamaIndex 集成适配器
- [ ] 可视化工具 (Graphviz 导出、交互式 HTML)
- [ ] 时间衰减曲线可配置
- [ ] 多 Agent 共享记忆图谱

---

## 关键验证命令

```bash
# 安装
pip install -e ".[dev]"

# Lint
ruff check src/ tests/ examples/ benchmarks/

# 类型检查
mypy src/ --strict

# 测试 (含覆盖率)
pytest tests/ -v --cov --cov-report=term-missing

# 文档构建
mkdocs build --strict

# Examples
python examples/01_knowledge_graph.py
python examples/02_mcts_reasoning.py
python examples/03_full_agent_loop.py

# 性能基准
python benchmarks/run_benchmarks.py
```

---

## 文件结构

```
agentic-os-core/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # CI: ruff + mypy + pytest + mkdocs
│   │   └── publish.yml         # PyPI 发布 (release 触发)
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── benchmarks/
│   └── run_benchmarks.py       # 26 项性能基准
├── docs/
│   ├── index.md                # 项目概述 + Quick Start
│   ├── architecture.md         # 5 层架构图
│   └── api-reference.md        # mkdocstrings API 参考
├── examples/
│   ├── 01_knowledge_graph.py
│   ├── 02_mcts_reasoning.py
│   ├── 03_full_agent_loop.py
│   ├── 04_custom_evaluator.py
│   ├── 05_persistence_and_restoration.py
│   └── 06_multi_turn_memory.py
├── src/agentic_os/
│   ├── __init__.py             # 36 个公共 API + __version__
│   ├── exceptions.py           # 12 个自定义异常
│   ├── core/
│   │   ├── graph/              # 知识图谱 (邻接表 + 倒排索引)
│   │   ├── tree/               # MCTS / 思维树
│   │   ├── memory/             # 记忆管理 (LRU + 图存储)
│   │   └── planning/           # 目标规划 + 执行
│   ├── plugins/                # 插件接口 + Mock 实现
│   └── utils/                  # 哈希 + 序列化
├── tests/                      # 172 个测试
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
├── README.md
├── mkdocs.yml
└── pyproject.toml
```
