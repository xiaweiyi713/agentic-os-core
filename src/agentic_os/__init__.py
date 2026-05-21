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
from agentic_os.core.memory.working import WorkingMemory as WorkingMemory
from agentic_os.core.planning.executor import Executor as Executor
from agentic_os.core.planning.goal import Goal as Goal
from agentic_os.core.planning.goal import GoalPriority as GoalPriority
from agentic_os.core.planning.goal import GoalState as GoalState
from agentic_os.core.planning.planner import Planner as Planner
from agentic_os.core.tree.mcts import MCTS as MCTS
from agentic_os.core.tree.thought_node import ThoughtNode as ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree as ThoughtTree
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
from agentic_os.plugins.base import ActionExecutor as ActionExecutor
from agentic_os.plugins.base import Evaluator as BaseEvaluator
from agentic_os.plugins.base import LLMBackend as LLMBackend

__all__ = [
    "MCTS",
    "ActionExecutor",
    "AgenticOSError",
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
    "LLMBackend",
    "LongTermMemory",
    "MaxChildrenExceededError",
    "MemoryCapacityError",
    "MemoryManager",
    "MemoryNode",
    "NodeNotFoundError",
    "NodeType",
    "Planner",
    "RetrievalStrategy",
    "ThoughtNode",
    "ThoughtTree",
    "TreeError",
    "ValidationError",
    "WorkingMemory",
    "create_episode",
    "create_fact",
    "create_goal",
    "create_reflection",
]
