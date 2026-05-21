"""Pruning strategies for the Tree of Thoughts.

Provides three complementary pruning approaches to keep the search tree
tractable:

- **Score threshold** -- remove low-scoring nodes.
- **Depth limit** -- truncate branches beyond a maximum depth.
- **Redundancy** -- collapse nodes that are semantically similar to siblings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree

logger = logging.getLogger(__name__)


def score_threshold_prune(tree: ThoughtTree, min_score: float) -> int:
    """Remove nodes whose cumulative score falls below *min_score*.

    Delegates to ``ThoughtTree.prune``; the root is never removed.

    Args:
        tree: The thought tree to prune (mutated in place).
        min_score: Minimum score a node must have to survive.

    Returns:
        The number of nodes removed.

    Example::

        tree = ThoughtTree()
        root = tree.set_root("Q")
        tree.add_thought(root, "Low", score=0.1)
        tree.add_thought(root, "High", score=0.9)
        n = score_threshold_prune(tree, min_score=0.5)
        assert n == 1
    """
    removed = tree.prune(min_score)
    logger.info("score_threshold_prune: removed %d nodes (min_score=%.2f)", removed, min_score)
    return removed


def depth_prune(tree: ThoughtTree, max_depth: int) -> int:
    """Truncate all branches deeper than *max_depth*.

    Nodes at or beyond *max_depth* have their entire subtree removed.

    Args:
        tree: The thought tree to prune (mutated in place).
        max_depth: Maximum allowed depth (root is depth 0).

    Returns:
        The number of nodes removed.

    Example::

        tree = ThoughtTree(max_depth=10)
        root = tree.set_root("Q")
        child = tree.add_thought(root, "A")
        tree.add_thought(child, "B")  # depth 2
        n = depth_prune(tree, max_depth=1)
        assert n == 1  # "B" removed
    """
    removed = 0

    def _prune(node: ThoughtNode, depth: int) -> None:
        nonlocal removed
        if depth >= max_depth:
            for child in node.children:
                removed += _count_subtree(child)
            node.children = []
            return
        for child in node.children:
            _prune(child, depth + 1)

    if tree.root:
        _prune(tree.root, 0)
    logger.info("depth_prune: removed %d nodes (max_depth=%d)", removed, max_depth)
    return removed


def redundancy_prune(tree: ThoughtTree, similarity_fn: Callable[[str, str], float]) -> int:
    """Remove nodes whose thought text is too similar to a retained sibling.

    For each group of siblings, nodes are kept greedily in order: a node
    is discarded if its similarity to **any** already-kept sibling exceeds
    the internal threshold of 0.8.

    Args:
        tree: The thought tree to prune (mutated in place).
        similarity_fn: A function ``(thought_a, thought_b) -> float``
            returning a similarity score in ``[0, 1]``.

    Returns:
        The number of nodes (and their subtrees) removed.

    Example::

        def sim(a, b):
            return 1.0 if a == b else 0.0

        tree = ThoughtTree()
        root = tree.set_root("Q")
        tree.add_thought(root, "idea A")
        tree.add_thought(root, "idea A")  # duplicate
        n = redundancy_prune(tree, similarity_fn=sim)
        assert n == 1
    """
    removed = 0
    threshold = 0.8

    def _prune(node: ThoughtNode) -> None:
        nonlocal removed
        if len(node.children) <= 1:
            for child in node.children:
                _prune(child)
            return

        to_keep: list[ThoughtNode] = []
        for child in node.children:
            is_redundant = False
            for kept in to_keep:
                if similarity_fn(child.thought, kept.thought) > threshold:
                    is_redundant = True
                    break
            if is_redundant:
                removed += _count_subtree(child)
            else:
                to_keep.append(child)

        node.children = to_keep
        for child in to_keep:
            _prune(child)

    if tree.root:
        _prune(tree.root)
    logger.info("redundancy_prune: removed %d nodes", removed)
    return removed


def _count_subtree(node: ThoughtNode) -> int:
    """Return the total number of nodes in the subtree rooted at *node*.

    Args:
        node: The root of the subtree to count.

    Returns:
        Number of nodes (inclusive of *node* itself).
    """
    count = 1
    for child in node.children:
        count += _count_subtree(child)
    return count
