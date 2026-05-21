"""LlamaIndex VectorStore adapter for AgenticOS."""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.vector.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Lazy import LlamaIndex
try:
    from llama_index.core.vector_stores.types import (  # type: ignore[import-not-found]
        BasePydanticVectorStore,
        VectorStoreQueryResult,
    )

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False
    # Fallback to simple base
    BasePydanticVectorStore = object


class AgenticOSVectorStore(BasePydanticVectorStore):  # type: ignore[misc]
    """LlamaIndex-compatible VectorStore backed by AgenticOS.

    Delegates storage and search to an internal VectorStore instance.

    Args:
        vector_store: An AgenticOS VectorStore implementation.
        embedder: Callable to generate embeddings from text.
    """

    stores_text: bool = True

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Any = None,
        **kwargs: Any,
    ) -> None:
        if _HAS_LLAMAINDEX:
            super().__init__(**kwargs)
        self._store = vector_store
        self._embedder = embedder

    def add(self, nodes: list[Any], **kwargs: Any) -> list[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of LlamaIndex TextNode objects.

        Returns:
            List of node IDs.
        """
        ids: list[str] = []
        for node in nodes:
            vector = getattr(node, "embedding", None)
            if vector is None and self._embedder:
                text = getattr(node, "text", "") or getattr(
                    node, "get_content", lambda: ""
                )()
                vector = self._embedder(text)
            if vector:
                metadata = getattr(node, "metadata", {}) or {}
                self._store.add(node.node_id, vector, metadata)
                ids.append(node.node_id)
        return ids

    def query(self, query: Any, **kwargs: Any) -> Any:
        """Query the vector store.

        Args:
            query: VectorStoreQuery object with query_embedding.

        Returns:
            VectorStoreQueryResult with matching nodes.
        """
        if not _HAS_LLAMAINDEX:
            raise ImportError("llama-index is required for query()")

        query_vector = getattr(query, "query_embedding", None)
        if query_vector is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        top_k = query.similarity_top_k if hasattr(query, "similarity_top_k") else 10
        results: list[SearchResult] = self._store.search(query_vector, top_k=top_k)

        nodes: list[Any] = []
        similarities: list[float] = []
        ids: list[str] = []
        for r in results:
            try:
                from llama_index.core.schema import TextNode  # type: ignore[import-not-found]

                node = TextNode(text="", id_=r.id, metadata=r.metadata)
            except ImportError:
                node = type(
                    "Node",
                    (),
                    {"id_": r.id, "metadata": r.metadata, "text": ""},
                )()
            nodes.append(node)
            similarities.append(r.score)
            ids.append(r.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a node by its reference document ID."""
        self._store.delete(ref_doc_id)
