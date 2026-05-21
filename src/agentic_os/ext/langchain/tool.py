"""LangChain-compatible tool adapter for querying AgenticOS knowledge graph."""

from __future__ import annotations

import json
import logging

from agentic_os.core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / optional LangChain import
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import BaseTool as _BaseTool  # type: ignore[import-not-found, import-untyped]  # noqa: I001

    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.tools import BaseTool as _BaseTool  # type: ignore[import-not-found, import-untyped]  # noqa: I001

        _HAS_LANGCHAIN = True
    except ImportError:
        _HAS_LANGCHAIN = False
        _BaseTool = object  # type: ignore[assignment, misc]


class AgenticOSGraphTool(_BaseTool):  # type: ignore[misc, valid-type]
    """LangChain-compatible tool for querying an AgenticOS knowledge graph.

    Wraps :class:`~agentic_os.core.memory.manager.MemoryManager.recall` as a
    callable tool that can be registered with LangChain agents.

    When LangChain is **not** installed the class falls back to inheriting
    from ``object`` so that importing this module never raises.

    Args:
        memory_manager: An AgenticOS ``MemoryManager`` instance.
        name: Tool name exposed to the agent.  Defaults to
            ``"agentic_graph_query"``.
        description: Tool description exposed to the agent.

    Examples::

        from agentic_os import MemoryManager
        from agentic_os.ext.langchain import AgenticOSGraphTool

        mm = MemoryManager()
        tool = AgenticOSGraphTool(mm)
        result = tool._run("search query")
    """

    name: str = "agentic_graph_query"  # type: ignore[assignment]
    description: str = (  # type: ignore[assignment]
        "Query the agent's knowledge graph to recall relevant memories, "
        "facts, and reflections."
    )

    def __init__(
        self,
        memory_manager: MemoryManager,
        name: str = "agentic_graph_query",
        description: str = (
            "Query the agent's knowledge graph to recall relevant memories, "
            "facts, and reflections."
        ),
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._memory = memory_manager
        self.name = name  # type: ignore[assignment]
        self.description = description  # type: ignore[assignment]

    def _run(self, query: str) -> str:
        """Execute a knowledge-graph query.

        Args:
            query: Natural-language search text.

        Returns:
            JSON string containing a list of recalled memory contents.
        """
        results = self._memory.recall(query)
        items = [
            {
                "id": node.id,
                "type": node.type.value,
                "content": node.content,
                "importance": node.importance,
            }
            for node in results
        ]
        return json.dumps(items, ensure_ascii=False, indent=2)
