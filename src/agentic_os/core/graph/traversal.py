"""Graph traversal algorithms -- BFS, DFS, shortest path, topological sort."""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentic_os.core.graph.edge import EdgeType
    from agentic_os.core.graph.knowledge_graph import KnowledgeGraph


def bfs(graph: KnowledgeGraph, start_id: str, max_depth: int = -1) -> dict[int, list[str]]:
    """Breadth-first traversal returning nodes grouped by depth.

    Args:
        graph: The knowledge graph to traverse.
        start_id: ID of the starting node.
        max_depth: Maximum depth to explore. ``-1`` means unlimited.

    Returns:
        A mapping ``{depth: [node_ids]}`` of discovered nodes per level.

    Example::

        layers = bfs(kg, "node_1", max_depth=3)
        # {0: ["node_1"], 1: ["node_2", "node_3"], 2: ["node_4"]}
    """
    result: dict[int, list[str]] = {}
    if start_id not in graph._nodes:
        return result
    logger.debug("bfs start=%s", start_id)
    visited: set[str] = {start_id}
    queue: deque[tuple[str, int]] = deque([(start_id, 0)])

    while queue:
        node_id, depth = queue.popleft()
        if max_depth >= 0 and depth > max_depth:
            continue
        result.setdefault(depth, []).append(node_id)
        for neighbor in graph.out_neighbors(node_id):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return result


def dfs(graph: KnowledgeGraph, start_id: str, max_depth: int = -1) -> list[str]:
    """Depth-first traversal returning nodes in visitation order.

    Args:
        graph: The knowledge graph to traverse.
        start_id: ID of the starting node.
        max_depth: Maximum depth to explore. ``-1`` means unlimited.

    Returns:
        A list of node IDs in the order they were first visited.

    Example::

        order = dfs(kg, "node_1", max_depth=2)
        # ["node_1", "node_2", "node_4", "node_3"]
    """
    result: list[str] = []
    visited: set[str] = set()
    if start_id not in graph._nodes:
        return result
    logger.debug("dfs start=%s", start_id)

    def _visit(node_id: str, depth: int) -> None:
        if node_id in visited:
            return
        if max_depth >= 0 and depth > max_depth:
            return
        visited.add(node_id)
        result.append(node_id)
        for neighbor in graph.out_neighbors(node_id):
            _visit(neighbor, depth + 1)

    _visit(start_id, 0)
    return result


def shortest_path(graph: KnowledgeGraph, start_id: str, end_id: str) -> list[str] | None:
    """Find the weighted shortest path using Dijkstra's algorithm.

    Edge cost is defined as ``1 - weight``, so higher-weighted edges are
    preferred.

    Args:
        graph: The knowledge graph to search.
        start_id: ID of the source node.
        end_id: ID of the destination node.

    Returns:
        A list of node IDs forming the shortest path, or ``None`` if no
        path exists.

    Example::

        path = shortest_path(kg, "node_1", "node_5")
        # ["node_1", "node_3", "node_5"]
    """
    if start_id == end_id:
        return [start_id]
    dist: dict[str, float] = {start_id: 0.0}
    prev: dict[str, str] = {}
    heap: list[tuple[float, str]] = [(0.0, start_id)]
    visited: set[str] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == end_id:
            path = []
            cur = end_id
            while cur != start_id:
                path.append(cur)
                cur = prev[cur]
            path.append(start_id)
            return path[::-1]
        for edge in graph.out_edges(u):
            if edge.target_id in visited:
                continue
            nd = d + (1.0 - edge.weight)  # Higher weight => shorter distance
            if nd < dist.get(edge.target_id, float("inf")):
                dist[edge.target_id] = nd
                prev[edge.target_id] = u
                heapq.heappush(heap, (nd, edge.target_id))
    logger.warning("shortest_path no path found: %s -> %s", start_id, end_id)
    return None


def traverse_by_type(graph: KnowledgeGraph, start_id: str,
                     edge_type: EdgeType, max_depth: int = -1) -> list[str]:
    """Traverse the graph following only edges of a specific type.

    Args:
        graph: The knowledge graph to traverse.
        start_id: ID of the starting node.
        edge_type: Only edges matching this type are followed.
        max_depth: Maximum depth to explore. ``-1`` means unlimited.

    Returns:
        A list of node IDs in visitation order.

    Example::

        causal_chain = traverse_by_type(kg, "event_1", EdgeType.CAUSAL)
    """
    result: list[str] = []
    visited: set[str] = set()

    def _visit(node_id: str, depth: int) -> None:
        if node_id in visited:
            return
        if max_depth >= 0 and depth > max_depth:
            return
        visited.add(node_id)
        result.append(node_id)
        for edge in graph.out_edges(node_id):
            if edge.type == edge_type and edge.target_id not in visited:
                _visit(edge.target_id, depth + 1)

    _visit(start_id, 0)
    return result


def topological_sort(graph: KnowledgeGraph) -> list[str] | None:
    """Return a topological ordering using Kahn's algorithm.

    Useful for analysing causal chains. If the graph contains a cycle,
    ``None`` is returned.

    Args:
        graph: The knowledge graph to sort.

    Returns:
        A list of node IDs in topological order, or ``None`` if a cycle
        exists.

    Example::

        order = topological_sort(kg)
        if order is None:
            print("Graph has a cycle")
    """
    in_degree: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        in_degree.setdefault(node.id, 0)
    for edge_list in graph._out_edges.values():
        for edge in edge_list.values():
            in_degree[edge.target_id] += 1

    queue: deque[str] = deque(nid for nid, d in in_degree.items() if d == 0)
    result: list[str] = []

    while queue:
        nid = queue.popleft()
        result.append(nid)
        for edge in graph.out_edges(nid):
            in_degree[edge.target_id] -= 1
            if in_degree[edge.target_id] == 0:
                queue.append(edge.target_id)

    if len(result) != graph.node_count:
        logger.warning("topological_sort cycle detected")
        return None
    return result


def connected_components(graph: KnowledgeGraph) -> list[set[str]]:
    """Find all weakly connected components (edge direction ignored).

    Time complexity: O(V + E) where V is the number of nodes and E is the
    number of edges.

    Args:
        graph: The knowledge graph to analyse.

    Returns:
        A list of sets, each set containing the node IDs of one connected
        component.

    Example::

        components = connected_components(kg)
        # [{"n1", "n2", "n3"}, {"n4", "n5"}]
    """
    visited: set[str] = set()
    components: list[set[str]] = []

    def _bfs(start: str) -> set[str]:
        comp: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in comp:
                continue
            comp.add(nid)
            for neighbor in graph.out_neighbors(nid):
                if neighbor not in comp:
                    queue.append(neighbor)
            for neighbor in graph.in_neighbors(nid):
                if neighbor not in comp:
                    queue.append(neighbor)
        return comp

    for node in graph.nodes:
        if node.id not in visited:
            comp = _bfs(node.id)
            visited |= comp
            components.append(comp)
    return components
