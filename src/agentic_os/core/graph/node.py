"""Knowledge graph node system with typed memory nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentic_os.utils.hashing import content_id


class NodeType(Enum):
    """Semantic types for knowledge graph nodes.

    Each type represents a different cognitive layer:
        EPISODE: Raw experiences and interactions.
        FACT: Extracted factual knowledge.
        REFLECTION: Agent introspections and insights.
        GOAL: Intentions and objectives.
    """

    EPISODE = "episode"        # Raw experience / interaction
    FACT = "fact"              # Extracted fact / knowledge
    REFLECTION = "reflection"  # Agent reflection / insight
    GOAL = "goal"              # Goal / intention


@dataclass
class MemoryNode:
    """A typed memory node in the knowledge graph.

    Nodes carry content, an importance score, and access metadata.
    Time complexity for all field access is O(1).

    Attributes:
        id: Unique identifier derived from content hash.
        type: Semantic node type.
        content: The actual text / payload stored in this node.
        importance: Relevance score in [0, 1]. Defaults to 0.5.
        access_count: Number of times this node has been accessed.
        created_at: ISO-8601 UTC timestamp of creation.
        updated_at: ISO-8601 UTC timestamp of last update.
        metadata: Arbitrary key-value pairs for extensions.
    """

    id: str
    type: NodeType
    content: str
    importance: float = 0.5
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MemoryNode({self.type.value}, {self.content[:40]!r})"

    def touch(self) -> None:
        """Increment access count and refresh ``updated_at`` timestamp.

        Example::

            node = MemoryNode(id="n1", type=NodeType.FACT, content="hello")
            node.touch()
            assert node.access_count == 1
        """
        self.access_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()


def create_episode(content: str, **metadata: Any) -> MemoryNode:
    """Create an episode node from raw experience content.

    Args:
        content: Raw text describing the experience.
        **metadata: Optional key-value pairs attached to the node.

    Returns:
        A ``MemoryNode`` with ``NodeType.EPISODE``.

    Example::

        node = create_episode("User asked about the weather", source="chat")
    """
    node_id = content_id(content, prefix="ep")
    return MemoryNode(id=node_id, type=NodeType.EPISODE, content=content, metadata=metadata)


def create_fact(content: str, importance: float = 0.6, **metadata: Any) -> MemoryNode:
    """Create a fact node with extracted knowledge.

    Args:
        content: The factual statement.
        importance: Initial importance score. Defaults to 0.6.
        **metadata: Optional key-value pairs attached to the node.

    Returns:
        A ``MemoryNode`` with ``NodeType.FACT``.

    Example::

        node = create_fact("Python 3.12 released in Oct 2023", importance=0.8)
    """
    node_id = content_id(content, prefix="fact")
    return MemoryNode(id=node_id, type=NodeType.FACT, content=content, importance=importance, metadata=metadata)


def create_reflection(content: str, importance: float = 0.7, **metadata: Any) -> MemoryNode:
    """Create a reflection node representing agent introspection.

    Args:
        content: The reflection or insight text.
        importance: Initial importance score. Defaults to 0.7.
        **metadata: Optional key-value pairs attached to the node.

    Returns:
        A ``MemoryNode`` with ``NodeType.REFLECTION``.

    Example::

        node = create_reflection("I should prioritise user requests", confidence=0.9)
    """
    node_id = content_id(content, prefix="ref")
    return MemoryNode(id=node_id, type=NodeType.REFLECTION, content=content, importance=importance, metadata=metadata)


def create_goal(content: str, importance: float = 0.8, **metadata: Any) -> MemoryNode:
    """Create a goal node representing an intention or objective.

    Args:
        content: The goal description.
        importance: Initial importance score. Defaults to 0.8.
        **metadata: Optional key-value pairs attached to the node.

    Returns:
        A ``MemoryNode`` with ``NodeType.GOAL``.

    Example::

        node = create_goal("Improve response latency", priority="high")
    """
    node_id = content_id(content, prefix="goal")
    return MemoryNode(id=node_id, type=NodeType.GOAL, content=content, importance=importance, metadata=metadata)
