"""Shared memory graph for multi-agent collaboration."""

from __future__ import annotations

import logging
import threading
from typing import Any

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import MemoryNode, create_episode

logger = logging.getLogger(__name__)


class AgentMemoryHandle:
    """Memory handle for a single agent within a shared memory graph.

    Each agent operates within its own namespace but shares the underlying
    graph. Agents can optionally access other agents' memories.

    Args:
        agent_id: Unique identifier for this agent.
        shared_graph: The parent SharedMemoryGraph.
    """

    def __init__(self, agent_id: str, shared_graph: SharedMemoryGraph) -> None:
        self._agent_id = agent_id
        self._graph = shared_graph

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def store(self, node: MemoryNode) -> str:
        """Store a node in the agent's namespace.

        Args:
            node: MemoryNode to store.

        Returns:
            The node ID.
        """
        with self._graph._lock:
            self._graph._graph.add_node(node)
            self._graph._namespaces.setdefault(self._agent_id, set()).add(node.id)
        logger.debug("Agent %s stored node %s", self._agent_id, node.id)
        return node.id

    def store_episode(self, content: str, **metadata: Any) -> str:
        """Create and store an episode node."""
        return self.store(create_episode(content, **metadata))

    def retrieve(self, query: str, top_k: int = 10,
                 include_shared: bool = True) -> list[MemoryNode]:
        """Retrieve memories.

        Args:
            query: Search text.
            top_k: Max results.
            include_shared: If True, also search other agents' shared memories.

        Returns:
            List of matching MemoryNode instances.
        """
        with self._graph._lock:
            own_ids = self._graph._namespaces.get(self._agent_id, set())
            if include_shared:
                # Search entire graph
                results = self._graph._graph.find_nodes(query)
            else:
                # Search only own namespace
                all_results = self._graph._graph.find_nodes(query)
                results = [n for n in all_results if n.id in own_ids]
        return results[:top_k]

    def share_with(self, node_id: str, target_agent_id: str) -> None:
        """Explicitly share a node with another agent.

        Args:
            node_id: ID of the node to share.
            target_agent_id: ID of the target agent.
        """
        with self._graph._lock:
            if target_agent_id not in self._graph._namespaces:
                raise ValueError(f"Agent {target_agent_id} not registered")
            self._graph._namespaces[target_agent_id].add(node_id)
        logger.info("Agent %s shared node %s with %s", self._agent_id, node_id, target_agent_id)

    def get_shared_from(self, source_agent_id: str) -> list[MemoryNode]:
        """Get memories shared by another agent that are also in our namespace.

        Args:
            source_agent_id: ID of the source agent.

        Returns:
            List of MemoryNode instances.
        """
        with self._graph._lock:
            own_ids = self._graph._namespaces.get(self._agent_id, set())
            source_ids = self._graph._namespaces.get(source_agent_id, set())
            shared_ids = own_ids & source_ids
        nodes = []
        for nid in shared_ids:
            node = self._graph._graph.get_node(nid)
            if node:
                nodes.append(node)
        return nodes

    def link(self, source_id: str, target_id: str,
             edge_type: EdgeType = EdgeType.ASSOCIATIVE, weight: float = 1.0) -> None:
        """Create an edge between two nodes."""
        with self._graph._lock:
            self._graph._graph.add_edge(source_id, target_id, edge_type, weight)


class SharedMemoryGraph:
    """Thread-safe shared memory graph for multi-agent collaboration.

    Multiple agents share a single knowledge graph but maintain separate
    namespaces. Agents can query their own memories, shared memories,
    or explicitly share nodes with specific agents.

    Example::

        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")

        alice.store_episode("Discovered new pattern")
        bob.retrieve("pattern", include_shared=True)  # finds alice's episode
    """

    def __init__(self) -> None:
        self._graph = KnowledgeGraph()
        self._namespaces: dict[str, set[str]] = {}  # agent_id -> node_ids
        self._lock = threading.RLock()

    def register_agent(self, agent_id: str) -> AgentMemoryHandle:
        """Register a new agent and return its memory handle.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            An AgentMemoryHandle for the agent.

        Raises:
            ValueError: If agent_id is already registered.
        """
        with self._lock:
            if agent_id in self._namespaces:
                raise ValueError(f"Agent {agent_id} already registered")
            self._namespaces[agent_id] = set()
        logger.info("Registered agent: %s", agent_id)
        return AgentMemoryHandle(agent_id, self)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's namespace (nodes remain in the shared graph).

        Args:
            agent_id: The agent to unregister.
        """
        with self._lock:
            self._namespaces.pop(agent_id, None)
        logger.info("Unregistered agent: %s", agent_id)

    def query_shared(self, query: str, top_k: int = 10,
                     exclude_agent: str | None = None) -> list[MemoryNode]:
        """Query across all agents' memories.

        Args:
            query: Search text.
            top_k: Max results.
            exclude_agent: Optional agent ID to exclude.

        Returns:
            List of matching MemoryNode instances.
        """
        with self._lock:
            results = self._graph.find_nodes(query)
            if exclude_agent:
                excluded_ids = self._namespaces.get(exclude_agent, set())
                results = [n for n in results if n.id not in excluded_ids]
        return results[:top_k]

    def list_agents(self) -> list[str]:
        """Return list of registered agent IDs."""
        with self._lock:
            return list(self._namespaces.keys())

    def stats(self) -> dict[str, Any]:
        """Return statistics about the shared graph and agents."""
        with self._lock:
            graph_stats = self._graph.stats()
            return {
                **graph_stats,
                "agents": len(self._namespaces),
                "namespaces": {aid: len(nids) for aid, nids in self._namespaces.items()},
            }
