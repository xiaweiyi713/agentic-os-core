"""Abstract plugin interfaces for LLM backends, evaluators, actions, and persistent memory storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agentic_os.core.planning.goal import Goal


class LLMBackend(ABC):
    """Abstract interface for large-language-model backends.

    Implementations must provide text generation and embedding methods.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            **kwargs: Backend-specific generation options.

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Compute an embedding vector for the given text.

        Args:
            text: Input text.

        Returns:
            List of floats representing the embedding.
        """
        ...


class Evaluator(ABC):
    """Abstract interface for scoring states and thoughts."""

    @abstractmethod
    def evaluate(self, state: dict[str, Any], thought: str) -> float:
        """Score a (state, thought) pair.

        Args:
            state: Current state dict.
            thought: Candidate thought string.

        Returns:
            Quality score in [0, 1].
        """
        ...


class ActionExecutor(ABC):
    """Abstract interface for executing agent actions."""

    @abstractmethod
    def execute(self, goal: Goal) -> tuple[bool, str]:
        """Execute a goal and return the outcome.

        Args:
            goal: The ``Goal`` to execute.

        Returns:
            Tuple of (success, result_message).
        """
        ...


class MemoryStore(ABC):
    """Abstract interface for persistent memory storage backends."""

    @abstractmethod
    def save(self, key: str, value: dict[str, Any]) -> None:
        """Persist a value under the given key.

        Args:
            key: Unique identifier.
            value: Dict payload to store.
        """
        ...

    @abstractmethod
    def load(self, key: str) -> dict[str, Any] | None:
        """Load a value by key.

        Args:
            key: Unique identifier.

        Returns:
            The stored dict, or ``None`` if not found.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry by key.

        Args:
            key: Unique identifier.

        Returns:
            True if the entry existed and was deleted.
        """
        ...

    @abstractmethod
    def query(self, filter_fn: Callable[[str, dict[str, Any]], bool]) -> list[tuple[str, dict[str, Any]]]:
        """Query entries matching a filter function.

        Args:
            filter_fn: Callable ``(key, value) -> bool``.

        Returns:
            List of ``(key, value)`` tuples that pass the filter.
        """
        ...
