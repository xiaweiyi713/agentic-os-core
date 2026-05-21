"""Goal data model - represents an agent goal with lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentic_os.utils.hashing import content_id


class GoalPriority(Enum):
    """Priority levels for goals, ordered from lowest to highest.

    Attributes:
        CRITICAL: Must be resolved immediately.
        HIGH: Important but not blocking.
        MEDIUM: Default priority.
        LOW: Nice-to-have, can be deferred.
    """
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class GoalState(Enum):
    """Lifecycle states of a goal.

    Attributes:
        PENDING: Not yet started.
        IN_PROGRESS: Currently being worked on.
        COMPLETED: Successfully finished.
        FAILED: Could not be completed.
        CANCELLED: Explicitly cancelled.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    """Represents a single agent goal with priority, state, and dependency tracking.

    Supports hierarchical decomposition via ``subgoals`` and ``parent_id``.
    """
    id: str
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    state: GoalState = GoalState.PENDING
    dependencies: list[str] = field(default_factory=list)   # depended Goal IDs
    subgoals: list[str] = field(default_factory=list)        # sub-goal ID list
    parent_id: str | None = None
    result: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Goal({self.description[:30]!r}, {self.state.value})"

    @property
    def is_terminal(self) -> bool:
        """bool: True if the goal is in a final state (completed, failed, or cancelled)."""
        return self.state in (GoalState.COMPLETED, GoalState.FAILED, GoalState.CANCELLED)

    def start(self) -> None:
        """Transition the goal to IN_PROGRESS."""
        self.state = GoalState.IN_PROGRESS

    def complete(self, result: str | None = None) -> None:
        """Mark the goal as COMPLETED.

        Args:
            result: Optional outcome description.
        """
        self.state = GoalState.COMPLETED
        self.result = result

    def fail(self, reason: str | None = None) -> None:
        """Mark the goal as FAILED.

        Args:
            reason: Optional failure reason.
        """
        self.state = GoalState.FAILED
        self.result = reason

    def cancel(self) -> None:
        """Transition the goal to CANCELLED."""
        self.state = GoalState.CANCELLED


def create_goal(description: str, priority: GoalPriority = GoalPriority.MEDIUM,
                **metadata: Any) -> Goal:
    """Factory function to create a ``Goal`` with an auto-generated ID.

    Args:
        description: Human-readable goal text.
        priority: Goal priority. Defaults to MEDIUM.
        **metadata: Extra metadata stored on the goal.

    Returns:
        A new ``Goal`` instance with a content-derived ID.

    Examples:
        >>> g = create_goal("Refactor auth module", priority=GoalPriority.HIGH)
    """
    gid = content_id(description, prefix="goal")
    return Goal(id=gid, description=description, priority=priority, metadata=metadata)
