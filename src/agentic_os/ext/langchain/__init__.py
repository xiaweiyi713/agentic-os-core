"""LangChain integration adapters for AgenticOS.

Provides memory, retriever, and tool wrappers that bridge AgenticOS
core components with LangChain interfaces. These adapters gracefully
degrade when LangChain is not installed, falling back to plain Python
base classes so that import-time errors never occur.

Requires the ``langchain`` optional dependency::

    pip install agentic-os-core[langchain]
"""

from agentic_os.ext.langchain.memory import AgenticOSMemory as AgenticOSMemory
from agentic_os.ext.langchain.retriever import AgenticOSRetriever as AgenticOSRetriever
from agentic_os.ext.langchain.tool import AgenticOSGraphTool as AgenticOSGraphTool

__all__ = [
    "AgenticOSGraphTool",
    "AgenticOSMemory",
    "AgenticOSRetriever",
]
