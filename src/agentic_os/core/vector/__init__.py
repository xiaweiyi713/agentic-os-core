"""Vector similarity search module."""

from agentic_os.core.vector.base import SearchResult as SearchResult
from agentic_os.core.vector.base import VectorStore as VectorStore
from agentic_os.core.vector.numpy_backend import NumpyVectorStore as NumpyVectorStore

__all__ = [
    "NumpyVectorStore",
    "SearchResult",
    "VectorStore",
]
