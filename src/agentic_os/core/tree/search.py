"""Search strategies for the Tree of Thoughts.

Provides three complementary tree-search algorithms:

- **UCB1** -- upper confidence bound for balancing exploration/exploitation.
- **Best-First Search** -- greedy expansion of the highest-scoring leaf.
- **Beam Search** -- width-limited breadth-first traversal.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable

from agentic_os.core.tree.thought_node import ThoughtNode


def ucb1_score(node: ThoughtNode, c: float = 1.414) -> float:
    """Compute the UCB1 score for a given node.

    The formula is::

        UCB1 = avg_score + c * sqrt(ln(N_parent) / N_child)

    Unvisited nodes return ``inf`` so they are always explored first.

    Args:
        node: The node to score.
        c: Exploration coefficient.  Defaults to ``sqrt(2)`` (~1.414).

    Returns:
        The UCB1 value as a float.

    Example::

        root = ThoughtNode(thought="Q", visits=10, score=5.0)
        child = ThoughtNode(thought="A", visits=3, score=3.0, parent=root)
        root.children.append(child)
        val = ucb1_score(child)
    """
    if node.visits == 0:
        return float("inf")
    if node.parent is None or node.parent.visits == 0:
        return node.avg_score
    return node.avg_score + c * math.sqrt(math.log(node.parent.visits) / node.visits)


def best_first_search(root: ThoughtNode, evaluator: Callable[[ThoughtNode], float],
                      max_nodes: int = 100) -> ThoughtNode:
    """Best-first search: always expand the current best-scoring leaf.

    Uses a max-heap (via negated scores) to efficiently retrieve the
    highest-scoring node at each step.

    Args:
        root: The root of the tree to search.
        evaluator: Function mapping a node to a numeric score.
        max_nodes: Maximum number of nodes to visit.

    Returns:
        The node with the highest evaluator score found during search.

    Example::

        root = ThoughtNode(thought="Root")
        a = ThoughtNode(thought="A", score=0.5, parent=root)
        b = ThoughtNode(thought="B", score=0.9, parent=root)
        root.children = [a, b]
        best = best_first_search(root, evaluator=lambda n: n.score)
        assert best.thought == "B"
    """
    counter = 0
    heap: list[tuple[float, int, ThoughtNode]] = []
    initial_score = -evaluator(root)
    heapq.heappush(heap, (initial_score, counter, root))
    best = root

    while heap and counter < max_nodes:
        _, _, node = heapq.heappop(heap)
        if evaluator(node) > evaluator(best):
            best = node
        for child in node.children:
            counter += 1
            score = -evaluator(child)
            heapq.heappush(heap, (score, counter, child))
    return best


def beam_search(root: ThoughtNode, beam_width: int,
                evaluator: Callable[[ThoughtNode], float]) -> list[ThoughtNode]:
    """Beam search: at each level keep only the top *beam_width* nodes.

    Progressively filters the tree by retaining the highest-scoring nodes
    per layer, pruning the rest.  Leaf nodes that have no siblings to
    compete with pass through automatically.

    Args:
        root: The root of the tree to search.
        beam_width: Number of top nodes to retain per level.
        evaluator: Function mapping a node to a numeric score.

    Returns:
        The final beam -- a list of the best leaf-level nodes.

    Example::

        root = ThoughtNode(thought="Q")
        for s in ["a", "b", "c", "d"]:
            root.children.append(ThoughtNode(thought=s, score=ord(s)))
        results = beam_search(root, beam_width=2, evaluator=lambda n: n.score)
        assert len(results) == 2
    """
    if root.is_leaf:
        return [root]

    current_beam: list[ThoughtNode] = [root]
    all_expanded: list[ThoughtNode] = [root]

    while True:
        candidates: list[ThoughtNode] = []
        has_children = False
        for node in current_beam:
            if node.children:
                has_children = True
                candidates.extend(node.children)
            else:
                candidates.append(node)

        if not has_children:
            break

        # Sort by evaluator, keep top beam_width
        candidates.sort(key=evaluator, reverse=True)
        current_beam = candidates[:beam_width]
        all_expanded.extend(current_beam)

    return current_beam
