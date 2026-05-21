# Architecture

agentic-os-core is organized into five layers, each with a clear responsibility boundary.

## Layer Overview

```
┌─────────────────────────────────────────────┐
│              Plugin Layer                     │
│  LLMBackend · Evaluator · ActionExecutor     │
├─────────────────────────────────────────────┤
│            Planning Layer                     │
│  Goal · Planner · Executor                   │
├─────────────────────────────────────────────┤
│            Memory Layer                       │
│  WorkingMemory · LongTermMemory · Manager    │
│  Consolidation Strategies                    │
├─────────────────────────────────────────────┤
│             Tree Layer                        │
│  ThoughtNode · ThoughtTree · MCTS            │
│  Search Strategies · Pruning                 │
├─────────────────────────────────────────────┤
│             Graph Layer                       │
│  MemoryNode · Edge · KnowledgeGraph          │
│  Traversal · Scoring                         │
└─────────────────────────────────────────────┘
```

## Graph Layer

The foundation. `KnowledgeGraph` uses an adjacency-list representation (`dict[str, dict[str, Edge]]`) with two auxiliary indexes:

- **Keyword index**: `dict[str, set[str]]` mapping tokens to node IDs for O(1) content search
- **Type index**: `dict[NodeType, set[str]]` for fast filtering by node type

Traversal algorithms (BFS, DFS, Dijkstra shortest path, topological sort) and PageRank importance scoring operate directly on this graph.

## Tree Layer

`ThoughtTree` manages a tree of `ThoughtNode` instances with configurable depth and branching limits. `MCTS` implements the classic four-phase loop:

1. **Selection** -- descend via UCB1: `avg_score + c * sqrt(ln(N_parent) / N_child)`
2. **Expansion** -- generate candidate thoughts and attach children
3. **Simulation** -- evaluate the new node with a heuristic
4. **Backpropagation** -- propagate rewards up to the root

Three pruning strategies keep the tree tractable: score threshold, depth limit, and redundancy removal.

## Memory Layer

A two-tier memory system inspired by human cognition:

- **WorkingMemory**: LRU cache via `OrderedDict`, O(1) put/get/evict
- **LongTermMemory**: wraps a `KnowledgeGraph` with four retrieval strategies (keyword, importance, recency, association)
- **MemoryManager**: unified facade that coordinates both layers

Consolidation strategies migrate data from working to long-term memory, optionally filtering by importance or extracting patterns.

## Planning Layer

`Goal` is a dataclass with priority, state lifecycle (pending -> in_progress -> completed/failed), and dependency tracking. `Planner` decomposes goals into subgoals and generates topologically-sorted execution plans. `Executor` runs plans sequentially, stopping on failure with optional retry.

## Plugin Layer

Abstract interfaces (`LLMBackend`, `Evaluator`, `ActionExecutor`, `MemoryStore`) allow integration with any external system. Mock implementations are provided for testing.
