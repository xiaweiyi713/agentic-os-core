"""Mock backends for testing and development - deterministic, in-process implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from agentic_os.core.planning.goal import Goal
from agentic_os.plugins.base import ActionExecutor, Evaluator, LLMBackend, MemoryStore
from agentic_os.utils.hashing import fnv1a_hash


class MockLLM(LLMBackend):
    """Deterministic mock LLM using rule-based or pre-configured responses.

    Useful for unit tests where real LLM calls should be avoided.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Initialize with optional response mappings.

        Args:
            responses: Dict mapping substring patterns to canned responses.
                If no pattern matches, a generic mock reply is returned.
        """
        self.responses = responses or {}
        self.call_log: list[tuple[str, str]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a canned response if the prompt contains a known key.

        Args:
            prompt: Input text.
            **kwargs: Ignored.

        Returns:
            Matched response or a generic mock string.
        """
        self.call_log.append(("generate", prompt))
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        return f"[Mock] Analysis result: {prompt[:50]}..."

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from a hash of *text*.

        Args:
            text: Input text.

        Returns:
            Normalised 8-dimensional float vector.
        """
        self.call_log.append(("embed", text))
        # Generate pseudo-vector (deterministic based on hash)
        seed = fnv1a_hash(text)
        vec = []
        for i in range(8):
            val = math.sin(seed * (i + 1) * 0.001)
            vec.append(val)
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec


class MockEvaluator(Evaluator):
    """Deterministic evaluator using content-hash scoring with a configurable bias."""

    def __init__(self, bias: float = 0.5) -> None:
        """Initialize with a bias factor.

        Args:
            bias: Value blended 50/50 with the hash-derived score.
                Defaults to 0.5.
        """
        self.bias = bias
        self.call_log: list[tuple[dict[str, Any], str, float]] = []

    def evaluate(self, state: dict[str, Any], thought: str) -> float:
        """Produce a deterministic score in [0, 1].

        Args:
            state: Current state (used only for logging).
            thought: Thought string to score.

        Returns:
            Score clamped to [0, 1].
        """
        h = fnv1a_hash(thought)
        raw = (h % 1000) / 1000.0
        score = raw * 0.5 + self.bias * 0.5
        score = max(0.0, min(1.0, score))
        self.call_log.append((state, thought, score))
        return score


class MockExecutor(ActionExecutor):
    """Mock action executor that logs calls and returns pre-configured results."""

    def __init__(self, results: dict[str, tuple[bool, str]] | None = None) -> None:
        """Initialize with optional per-goal results.

        Args:
            results: Dict mapping goal ID or description to an
                ``(success, message)`` tuple. Unmatched goals succeed by default.
        """
        self.results = results or {}
        self.call_log: list[tuple[str, tuple[bool, str]]] = []

    def execute(self, goal: Goal) -> tuple[bool, str]:
        """Return a pre-configured result or a default success.

        Args:
            goal: The ``Goal`` to "execute".

        Returns:
            Tuple of (success, result_message).
        """
        if goal.id in self.results:
            result = self.results[goal.id]
        elif goal.description in self.results:
            result = self.results[goal.description]
        else:
            result = (True, f"Completed: {goal.description[:50]}")
        self.call_log.append((goal.id, result))
        return result


class MockMemoryStore(MemoryStore):
    """In-memory dict-backed storage for testing."""

    def __init__(self) -> None:
        """Initialize with an empty internal dict."""
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, key: str, value: dict[str, Any]) -> None:
        """Store a value in the in-memory dict.

        Args:
            key: Unique identifier.
            value: Dict payload to store.
        """
        self._store[key] = value

    def load(self, key: str) -> dict[str, Any] | None:
        """Retrieve a value by key.

        Args:
            key: Unique identifier.

        Returns:
            The stored dict, or ``None``.
        """
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Remove an entry by key.

        Args:
            key: Unique identifier.

        Returns:
            True if the entry existed.
        """
        return self._store.pop(key, None) is not None

    def query(self, filter_fn: Callable[[str, dict[str, Any]], bool]) -> list[tuple[str, dict[str, Any]]]:
        """Return entries matching the filter.

        Args:
            filter_fn: Callable ``(key, value) -> bool``.

        Returns:
            List of matching ``(key, value)`` tuples.
        """
        return [(k, v) for k, v in self._store.items() if filter_fn(k, v)]
