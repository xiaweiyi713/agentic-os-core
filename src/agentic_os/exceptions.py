"""Custom exception hierarchy for agentic-os-core.

All exceptions inherit from :class:`AgenticOSError`, allowing callers to
catch all project-specific errors with a single ``except AgenticOSError``
clause.

These are used only at explicit ``raise`` sites. Public API methods that
return ``None`` for missing or invalid inputs continue to do so (with
WARNING-level logging added separately).
"""

from __future__ import annotations


class AgenticOSError(Exception):
    """Base exception for all agentic-os-core errors."""


class ValidationError(AgenticOSError):
    """Input validation failure (weight range, score range, invalid types)."""


class GraphError(AgenticOSError):
    """Base for graph-related errors."""


class NodeNotFoundError(GraphError):
    """A required graph node was not found."""


class EdgeNotFoundError(GraphError):
    """A required graph edge was not found."""


class CyclicGraphError(GraphError):
    """A cyclic dependency was detected in a DAG operation."""


class TreeError(AgenticOSError):
    """Base for tree-related errors."""


class DepthExceededError(TreeError):
    """Tree depth limit exceeded."""


class MaxChildrenExceededError(TreeError):
    """Maximum children per node exceeded."""


class ExecutionError(AgenticOSError):
    """Base for execution-related errors."""


class GoalExecutionError(ExecutionError):
    """A goal execution step failed."""


class MemoryCapacityError(AgenticOSError):
    """Memory capacity limit exceeded."""
