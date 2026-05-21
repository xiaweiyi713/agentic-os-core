# Contributing to Agentic OS Core

Thank you for your interest in contributing! This guide covers the basics.

## Setup

```bash
git clone https://github.com/xuwenyao/agentic-os-core.git
cd agentic-os-core
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run the full check suite:
   ```bash
   pytest tests/ -v       # All 172+ tests must pass
   ruff check src/ tests/ # No lint errors
   mypy src/ --strict     # Zero type errors
   ```
4. Submit a pull request

## Code Style

- **Python 3.10+** — use `X | Y` union syntax, `match` statements, etc.
- **Zero dependencies** — only Python standard library in `src/`. Test/dev deps are fine.
- **Type hints** — all public APIs must have complete type annotations.
- **Docstrings** — Google style, English, for all public classes and methods.
- **No comments** — code should be self-documenting. Only add comments for non-obvious constraints.
- **Line length** — 100 chars max.

## Project Structure

```
src/agentic_os/
├── core/graph/       # Knowledge Graph (do not add LLM calls here)
├── core/tree/        # MCTS / Tree-of-Thoughts
├── core/memory/      # Memory management (orchestration layer)
├── core/planning/    # Goal decomposition and execution
├── plugins/          # Abstract interfaces + mock implementations
└── utils/            # Hashing, serialization
```

## Design Principles

1. **Data structure first** — every module should be usable without an LLM.
2. **O(1) or O(log n)** — hot paths (memory put/get, graph add/remove) must be fast.
3. **Pluggable** — LLM calls, evaluators, and action executors are abstracted behind interfaces.
4. **Serializable** — all state can be persisted to JSON and restored.

## Adding a New Feature

1. If it touches core data structures (graph, tree, memory), add it under `core/`.
2. If it integrates with external systems, add it under `plugins/`.
3. Add tests in `tests/` — aim for >90% coverage of new code.
4. Update `__init__.py` exports if adding new public symbols.
5. Add an example in `examples/` if the feature is non-trivial.

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run checks automatically:

```bash
pip install pre-commit
pre-commit install
```

Hooks run on every commit: ruff lint + format, trailing whitespace, YAML/TOML validation, mypy strict.

## Reporting Issues

- Use [GitHub Issues](https://github.com/xuwenyao/agentic-os-core/issues).
- Include: Python version, OS, minimal reproducible example.
