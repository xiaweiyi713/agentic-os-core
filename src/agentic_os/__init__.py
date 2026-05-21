"""Agentic OS Core - Agent memory and planning engine.

A memory and planning scheduling framework for hard-core agents
based on graph data structures and MCTS.

Quick Start::

    from agentic_os import KnowledgeGraph, MCTS, MemoryManager

    # Build knowledge graph
    kg = KnowledgeGraph()
    kg.add_node(create_episode("User asked about weather"))

    # MCTS reasoning
    mcts = MCTS(max_iterations=100)
    tree = mcts.search("decision problem", {}, generator, evaluator)

    # Memory management
    memory = MemoryManager()
    memory.add_experience("Learned new knowledge")
    memory.recall("knowledge")
"""

__version__ = "0.1.0"

from agentic_os.core.graph.edge import Edge as Edge
from agentic_os.core.graph.edge import EdgeType as EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph as KnowledgeGraph
from agentic_os.core.graph.node import (
    MemoryNode as MemoryNode,
)
from agentic_os.core.graph.node import (
    NodeType as NodeType,
)
from agentic_os.core.graph.node import (
    create_episode as create_episode,
)
from agentic_os.core.graph.node import (
    create_fact as create_fact,
)
from agentic_os.core.graph.node import (
    create_goal as create_goal,
)
from agentic_os.core.graph.node import (
    create_reflection as create_reflection,
)
from agentic_os.core.memory.longterm import LongTermMemory as LongTermMemory
from agentic_os.core.memory.longterm import RetrievalStrategy as RetrievalStrategy
from agentic_os.core.memory.manager import MemoryManager as MemoryManager
from agentic_os.core.memory.shared import AgentMemoryHandle as AgentMemoryHandle
from agentic_os.core.memory.shared import SharedMemoryGraph as SharedMemoryGraph
from agentic_os.core.memory.working import WorkingMemory as WorkingMemory
from agentic_os.core.planning.executor import Executor as Executor
from agentic_os.core.planning.goal import Goal as Goal
from agentic_os.core.planning.goal import GoalPriority as GoalPriority
from agentic_os.core.planning.goal import GoalState as GoalState
from agentic_os.core.planning.goal import create_goal as create_planning_goal
from agentic_os.core.planning.planner import Planner as Planner
from agentic_os.core.tree.async_mcts import AsyncMCTS as AsyncMCTS
from agentic_os.core.tree.mcts import MCTS as MCTS
from agentic_os.core.tree.thought_node import ThoughtNode as ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree as ThoughtTree
from agentic_os.core.vector.base import SearchResult as SearchResult
from agentic_os.core.vector.base import VectorStore as VectorStore
from agentic_os.core.vector.numpy_backend import NumpyVectorStore as NumpyVectorStore
from agentic_os.exceptions import (
    AgenticOSError as AgenticOSError,
)
from agentic_os.exceptions import (
    CyclicGraphError as CyclicGraphError,
)
from agentic_os.exceptions import (
    DepthExceededError as DepthExceededError,
)
from agentic_os.exceptions import (
    EdgeNotFoundError as EdgeNotFoundError,
)
from agentic_os.exceptions import (
    ExecutionError as ExecutionError,
)
from agentic_os.exceptions import (
    GoalExecutionError as GoalExecutionError,
)
from agentic_os.exceptions import (
    GraphError as GraphError,
)
from agentic_os.exceptions import (
    MaxChildrenExceededError as MaxChildrenExceededError,
)
from agentic_os.exceptions import (
    MemoryCapacityError as MemoryCapacityError,
)
from agentic_os.exceptions import (
    NodeNotFoundError as NodeNotFoundError,
)
from agentic_os.exceptions import (
    TreeError as TreeError,
)
from agentic_os.exceptions import (
    ValidationError as ValidationError,
)
from agentic_os.ext.langchain.memory import AgenticOSMemory as AgenticOSMemory
from agentic_os.ext.langchain.retriever import AgenticOSRetriever as AgenticOSRetriever
from agentic_os.ext.langchain.tool import AgenticOSGraphTool as AgenticOSGraphTool
from agentic_os.ext.llamaindex.vector_store import (
    AgenticOSVectorStore as AgenticOSVectorStore,
)
from agentic_os.ext.visualization.graph_visualizer import (
    KnowledgeGraphVisualizer as KnowledgeGraphVisualizer,
)
from agentic_os.ext.visualization.tree_visualizer import (
    ThoughtTreeVisualizer as ThoughtTreeVisualizer,
)
from agentic_os.plugins.base import ActionExecutor as ActionExecutor
from agentic_os.plugins.base import Evaluator as BaseEvaluator
from agentic_os.plugins.base import LLMBackend as LLMBackend
from agentic_os.plugins.base import MemoryStore as MemoryStore
from agentic_os.plugins.mock import MockEvaluator as MockEvaluator
from agentic_os.plugins.mock import MockExecutor as MockExecutor
from agentic_os.plugins.mock import MockLLM as MockLLM
from agentic_os.plugins.mock import MockMemoryStore as MockMemoryStore

__all__ = [
    "MCTS",
    "ActionExecutor",
    "AgentMemoryHandle",
    "AgenticOSError",
    "AgenticOSGraphTool",
    "AgenticOSMemory",
    "AgenticOSRetriever",
    "AgenticOSVectorStore",
    "AsyncMCTS",
    "BaseEvaluator",
    "CyclicGraphError",
    "DepthExceededError",
    "Edge",
    "EdgeNotFoundError",
    "EdgeType",
    "ExecutionError",
    "Executor",
    "Goal",
    "GoalExecutionError",
    "GoalPriority",
    "GoalState",
    "GraphError",
    "KnowledgeGraph",
    "KnowledgeGraphVisualizer",
    "LLMBackend",
    "LongTermMemory",
    "MaxChildrenExceededError",
    "MemoryCapacityError",
    "MemoryManager",
    "MemoryNode",
    "MemoryStore",
    "MockEvaluator",
    "MockExecutor",
    "MockLLM",
    "MockMemoryStore",
    "NodeNotFoundError",
    "NodeType",
    "NumpyVectorStore",
    "Planner",
    "RetrievalStrategy",
    "SearchResult",
    "SharedMemoryGraph",
    "ThoughtNode",
    "ThoughtTree",
    "ThoughtTreeVisualizer",
    "TreeError",
    "ValidationError",
    "VectorStore",
    "WorkingMemory",
    "create_episode",
    "create_fact",
    "create_goal",
    "create_planning_goal",
    "create_reflection",
]
