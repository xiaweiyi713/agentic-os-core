"""Abstract base class for vector similarity search backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        id: Unique identifier of the matched vector.
        score: Similarity score (higher is more similar).
        metadata: Optional key-value pairs attached to the result.
    """

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search.

    Concrete implementations must support adding, deleting, and searching
    vectors by cosine similarity.
    """

    @abstractmethod
    def add(self, id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None:
        """Add a single vector to the store.

        Args:
            id: Unique identifier for the vector.
            vector: Embedding vector.
            metadata: Optional key-value metadata.
        """
        ...

    @abstractmethod
    def add_batch(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        """Add multiple vectors at once.

        Args:
            items: List of ``(id, vector, metadata)`` tuples.
        """
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]:
        """Search for the *top_k* most similar vectors.

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`SearchResult` sorted by descending similarity.
        """
        ...

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a vector by its identifier.

        Args:
            id: The vector identifier to remove.

        Returns:
            ``True`` if the vector was found and deleted, ``False`` otherwise.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""
        ...
