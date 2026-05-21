"""LlamaIndex integration adapters for AgenticOS.

Provides a VectorStore wrapper that bridges AgenticOS core components
with LlamaIndex interfaces. This adapter gracefully degrades when
LlamaIndex is not installed, falling back to plain Python base classes
so that import-time errors never occur.

Requires the ``llamaindex`` optional dependency::

    pip install agentic-os-core[llamaindex]
"""

from agentic_os.ext.llamaindex.vector_store import (
    AgenticOSVectorStore as AgenticOSVectorStore,
)

__all__ = [
    "AgenticOSVectorStore",
]
