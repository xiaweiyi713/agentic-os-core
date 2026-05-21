"""Abstract storage backend interface for persistent knowledge graph storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import MemoryNode


class StorageBackend(ABC):
    """Abstract interface for persistent storage backends.

    Implementations must support saving and loading complete knowledge graphs
    as well as individual node CRUD operations.
    """

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> None:
        """Serialize and persist an entire knowledge graph.

        Args:
            graph: The ``KnowledgeGraph`` to persist.
        """
        ...

    @abstractmethod
    def load_graph(self) -> KnowledgeGraph | None:
        """Load a previously saved knowledge graph.

        Returns:
            The reconstructed ``KnowledgeGraph``, or ``None`` if no data exists.
        """
        ...

    @abstractmethod
    def save_node(self, node: MemoryNode) -> None:
        """Persist or update a single node.

        Args:
            node: The ``MemoryNode`` to save.
        """
        ...

    @abstractmethod
    def load_node(self, node_id: str) -> MemoryNode | None:
        """Load a single node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            The ``MemoryNode`` if found, else ``None``.
        """
        ...

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            ``True`` if the node existed and was deleted.
        """
        ...

    @abstractmethod
    def query_nodes(self, filter_fn: Callable[[str, dict[str, Any]], bool]) -> list[MemoryNode]:
        """Query nodes matching a filter function.

        Args:
            filter_fn: Callable ``(key, value_dict) -> bool``.

        Returns:
            List of matching ``MemoryNode`` instances.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources and close connections."""
        ...
