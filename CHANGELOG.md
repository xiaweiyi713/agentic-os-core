# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-21

### Added

- Knowledge graph with adjacency-list storage and inverted index
- Graph traversal: BFS, DFS, Dijkstra shortest path, topological sort, connected components
- PageRank-style importance scoring with dangling-node redistribution
- Tree of Thoughts with depth and branching limits
- MCTS (Monte Carlo Tree Search) with UCB1 selection, expansion, simulation, backpropagation
- Search strategies: best-first search, beam search
- Pruning strategies: score threshold, depth limit, redundancy removal
- Working memory (LRU cache via OrderedDict)
- Long-term memory backed by knowledge graph with 4 retrieval strategies
- Memory consolidation: simple, importance-filtered, pattern-extracting
- Unified memory manager facade
- Goal dataclass with priority, state lifecycle, and dependency tracking
- Planner with goal decomposition and topological plan ordering
- Plan executor with retry support
- Plugin interfaces: LLMBackend, Evaluator, ActionExecutor, MemoryStore
- Mock backends for testing
- Custom exception hierarchy (AgenticOSError, ValidationError, GraphError, etc.)
- Structured logging across all modules
- FNV-1a content hashing
- JSON serialization with dataclass/Enum support
- 142 tests (unit, integration, edge cases, property-based)
- Hypothesis property-based tests for invariant verification
- MkDocs documentation site with auto-generated API reference
- CI pipeline (ruff, mypy strict, pytest with coverage)
- Performance benchmarks (26 benchmarks across all modules)
