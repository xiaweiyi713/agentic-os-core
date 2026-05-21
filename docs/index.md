# agentic-os-core

A memory and planning engine for AI Agents, built on graph data structures and MCTS tree search.

**Zero external dependencies** -- pure Python 3.10+, using only the standard library.

## Features

- **Knowledge Graph** -- adjacency-list storage with inverted index, BFS/DFS/Dijkstra traversal, PageRank scoring
- **MCTS Tree of Thoughts** -- full four-phase Monte Carlo Tree Search with UCB1 selection
- **Hierarchical Memory** -- LRU working memory + graph-backed long-term memory with multi-strategy retrieval
- **Goal Planning** -- hierarchical decomposition, topological plan ordering, execution with retry
- **Plugin Architecture** -- abstract interfaces for LLM backends, evaluators, and action executors

## Quick Start

```python
from agentic_os import KnowledgeGraph, MCTS, MemoryManager, create_episode

# Build a knowledge graph
kg = KnowledgeGraph()
kg.add_node(create_episode("User asked about weather"))
kg.add_node(create_episode("Weather API returned sunny forecast"))

# MCTS reasoning
mcts = MCTS(max_iterations=100)
tree = mcts.search(
    "decision problem",
    {},
    generator=lambda s: ["option A", "option B"],
    evaluator=lambda s, t: 0.8 if "A" in t else 0.3,
)

# Memory management
memory = MemoryManager()
memory.add_experience("Learned new knowledge")
memory.consolidate()
results = memory.recall("knowledge")
```

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## License

MIT
