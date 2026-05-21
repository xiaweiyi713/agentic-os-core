"""NumPy-backed vector store for cosine similarity search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agentic_os.core.vector.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _require_numpy() -> Any:
    """Lazy-import numpy, raising a helpful error when it is not installed."""
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for NumpyVectorStore.  "
            "Install it with: pip install agentic-os-core[numpy]"
        ) from exc


class NumpyVectorStore(VectorStore):
    """In-memory vector store powered by NumPy cosine similarity.

    Vectors and metadata are held in plain dicts; persistence is provided
    via :meth:`save` / :meth:`load` which write ``.npz`` + ``.json`` files.

    Example::

        store = NumpyVectorStore(dimension=3)
        store.add("v1", [1.0, 0.0, 0.0], {"label": "x"})
        store.add("v2", [0.0, 1.0, 0.0], {"label": "y"})
        hits = store.search([1.0, 0.1, 0.0], top_k=2)
    """

    def __init__(self, dimension: int | None = None) -> None:
        self._np = _require_numpy()
        self._dimension = dimension
        self._vectors: dict[str, Any] = {}   # id -> np.ndarray
        self._metadata: dict[str, dict[str, Any]] = {}  # id -> metadata

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None:
        """Add a single vector to the store.

        Args:
            id: Unique identifier for the vector.
            vector: Embedding vector.
            metadata: Optional key-value metadata.
        """
        np = self._np
        arr = np.asarray(vector, dtype=np.float64)
        if self._dimension is not None and arr.shape[0] != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}"
            )
        self._vectors[id] = arr
        self._metadata[id] = metadata if metadata is not None else {}
        logger.debug("add: id=%s dim=%d", id, arr.shape[0])

    def add_batch(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        """Add multiple vectors at once.

        Args:
            items: List of ``(id, vector, metadata)`` tuples.
        """
        for id_, vector, metadata in items:
            self.add(id_, vector, metadata)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]:
        """Search for the *top_k* most similar vectors using cosine similarity.

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of results.

        Returns:
            List of :class:`SearchResult` sorted by descending similarity.
        """
        np = self._np
        q = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        results: list[SearchResult] = []
        for vid, vec in self._vectors.items():
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            cos_sim = float(np.dot(q, vec) / (q_norm * v_norm))
            results.append(SearchResult(id=vid, score=cos_sim, metadata=dict(self._metadata[vid])))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.debug("search: %d candidates, returning top %d", len(results), top_k)
        return results[:top_k]

    def delete(self, id: str) -> bool:
        """Delete a vector by identifier.

        Args:
            id: The vector identifier to remove.

        Returns:
            ``True`` if found and deleted, ``False`` otherwise.
        """
        if id in self._vectors:
            del self._vectors[id]
            del self._metadata[id]
            logger.debug("delete: id=%s", id)
            return True
        return False

    def count(self) -> int:
        """Return the number of stored vectors."""
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist vectors and metadata to disk.

        Writes two files:
        - ``<path>.npz`` -- compressed NumPy array archive.
        - ``<path>.json`` -- metadata JSON file.

        Args:
            path: Base file path (without extension).
        """
        np = self._np
        path = Path(path)
        if not self._vectors:
            logger.warning("save: store is empty, writing empty files")
        arrays = {f"vec_{vid}": vec for vid, vec in self._vectors.items()}
        id_list = list(self._vectors.keys())
        arrays["_ids"] = np.array(id_list)
        np.savez(str(path.with_suffix(".npz")), **arrays)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"metadata": self._metadata, "ids": id_list}, f, ensure_ascii=False, indent=2)
        logger.debug("save: %d vectors to %s", len(self._vectors), path)

    @classmethod
    def load(cls, path: str | Path) -> NumpyVectorStore:
        """Load vectors and metadata from disk.

        Args:
            path: Base file path (without extension).

        Returns:
            A populated :class:`NumpyVectorStore` instance.
        """
        np = _require_numpy()
        path = Path(path)
        data = np.load(str(path.with_suffix(".npz")), allow_pickle=False)
        with open(path.with_suffix(".json"), encoding="utf-8") as f:
            meta = json.load(f)
        store = cls.__new__(cls)
        store._np = np
        store._dimension = None
        store._vectors = {}
        store._metadata = {}
        ids: list[str] = meta["ids"]
        for vid in ids:
            store._vectors[vid] = data[f"vec_{vid}"]
            store._metadata[vid] = meta["metadata"].get(vid, {})
        logger.debug("load: %d vectors from %s", len(store._vectors), path)
        return store
