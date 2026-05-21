"""LangChain-compatible memory adapter backed by AgenticOS MemoryManager."""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / optional LangChain import
# ---------------------------------------------------------------------------
try:
    from langchain_core.memory import BaseMemory as _BaseMemory  # type: ignore[import-not-found, import-untyped]  # noqa: I001

    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.memory import BaseMemory as _BaseMemory  # type: ignore[import-not-found, import-untyped]  # noqa: I001

        _HAS_LANGCHAIN = True
    except ImportError:
        _HAS_LANGCHAIN = False
        _BaseMemory = object  # type: ignore[assignment, misc]


class AgenticOSMemory(_BaseMemory):  # type: ignore[misc, valid-type]
    """LangChain-compatible memory backed by AgenticOS MemoryManager.

    Wraps :class:`~agentic_os.core.memory.manager.MemoryManager` as a
    LangChain ``BaseMemory``, storing conversation context as episodes and
    retrieving relevant memories via keyword matching.

    When LangChain is **not** installed the class falls back to inheriting
    from ``object`` so that importing this module never raises.

    Args:
        memory_manager: An AgenticOS ``MemoryManager`` instance.
        memory_key: Key name used for memory variables.  Defaults to
            ``"history"``.

    Examples::

        from agentic_os import MemoryManager
        from agentic_os.ext.langchain import AgenticOSMemory

        mm = MemoryManager()
        memory = AgenticOSMemory(mm, memory_key="chat_history")
        memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
        result = memory.load_memory_variables({"input": "Hello"})
    """

    def __init__(self, memory_manager: MemoryManager, memory_key: str = "history") -> None:
        self._memory = memory_manager
        self._memory_key = memory_key

    # -- BaseMemory interface --------------------------------------------------

    @property  # type: ignore[override]
    def memory_variables(self) -> list[str]:
        """List of variable names this memory class provides."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Retrieve relevant memories formatted as a single string.

        Uses the value associated with ``memory_key`` (or ``"input"`` as a
        fallback) as the recall query.

        Args:
            inputs: The current prompt inputs.

        Returns:
            A dict mapping ``memory_key`` to the joined memory texts.
        """
        query = inputs.get(self._memory_key, inputs.get("input", ""))
        if query:
            memories = self._memory.recall(str(query))
            texts = [m.content for m in memories]
            return {self._memory_key: "\n".join(texts)}
        return {self._memory_key: ""}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store a conversational turn as two separate experiences.

        Args:
            inputs: Must contain ``"input"`` with the user message.
            outputs: Must contain ``"output"`` with the AI response.
        """
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        if user_input:
            self._memory.add_experience(f"User: {user_input}")
        if ai_output:
            self._memory.add_experience(f"AI: {ai_output}")

    def clear(self) -> None:
        """Clear working memory only; long-term memory is preserved."""
        self._memory.working.clear()
