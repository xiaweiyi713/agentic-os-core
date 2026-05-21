"""LangChain-compatible retriever adapter backed by AgenticOS VectorStore."""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.vector.base import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / optional LangChain import
# ---------------------------------------------------------------------------
try:
    from langchain_core.retrievers import BaseRetriever as _BaseRetriever  # type: ignore[import-not-found, import-untyped]  # noqa: I001

    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.schema import BaseRetriever as _BaseRetriever  # type: ignore[import-not-found, import-untyped, no-redef]  # noqa: I001

        _HAS_LANGCHAIN = True
    except ImportError:
        _HAS_LANGCHAIN = False
        _BaseRetriever = object  # type: ignore[assignment, misc]

# Resolve PrivateAttr for pydantic v2 when langchain is present
_PrivateAttr: Any = None
if _HAS_LANGCHAIN:
    try:
        from pydantic import PrivateAttr as _PrivateAttr  # type: ignore[import-not-found, import-untyped]  # noqa: I001
    except ImportError:
        _PrivateAttr = None


class AgenticOSRetriever(_BaseRetriever):  # type: ignore[misc, valid-type]
    """LangChain-compatible retriever backed by AgenticOS VectorStore.

    Wraps an AgenticOS :class:`~agentic_os.core.vector.base.VectorStore`
    together with a callable embedder so that it can be used wherever
    LangChain expects a retriever.

    When LangChain is **not** installed the class falls back to inheriting
    from ``object`` so that importing this module never raises.

    Args:
        vector_store: An AgenticOS ``VectorStore`` instance.
        embedder: A callable that converts a string into a ``list[float]``.
        top_k: Maximum number of results per query.  Defaults to ``5``.

    Examples::

        from agentic_os import VectorStore
        from agentic_os.ext.langchain import AgenticOSRetriever

        store = VectorStore(...)
        retriever = AgenticOSRetriever(store, embed_fn, top_k=3)
        docs = retriever.invoke("search query")
    """

    if _HAS_LANGCHAIN and _PrivateAttr is not None:
        _store: VectorStore = _PrivateAttr()  # type: ignore[assignment, misc]
        _embedder: Any = _PrivateAttr()  # type: ignore[assignment, misc]
        _top_k: int = _PrivateAttr()  # type: ignore[assignment, misc]

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Any,  # Callable[[str], list[float]] -- Any for langchain compat
        top_k: int = 5,
    ) -> None:
        if _HAS_LANGCHAIN:
            super().__init__()  # type: ignore[no-untyped-call]
        self._store = vector_store  # type: ignore[assignment]
        self._embedder = embedder  # type: ignore[assignment]
        self._top_k = top_k  # type: ignore[assignment]

    def invoke(self, query: str) -> list[dict[str, Any]]:  # type: ignore[override]
        """Retrieve documents matching *query*.

        Args:
            query: Search text.

        Returns:
            A list of dicts with ``page_content`` and ``metadata`` keys,
            mimicking the LangChain ``Document`` interface.
        """
        return self._get_relevant_documents(query)

    def _get_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Internal retrieval used by both ``invoke`` and LangChain.

        Args:
            query: Search text.

        Returns:
            LangChain ``Document``-compatible dicts.
        """
        query_vector = self._embedder(query)
        results = self._store.search(query_vector, top_k=self._top_k)
        return [
            {
                "page_content": result.metadata.get("content", ""),
                "metadata": {
                    "id": result.id,
                    "score": result.score,
                    **{
                        k: v
                        for k, v in result.metadata.items()
                        if k != "content"
                    },
                },
            }
            for result in results
        ]
