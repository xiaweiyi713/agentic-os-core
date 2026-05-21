"""Tree manager for the Tree of Thoughts reasoning framework.

Provides creation, traversal, pruning, and visualisation of a
``ThoughtNode`` tree with configurable depth and branching limits.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.tree.thought_node import ThoughtNode

logger = logging.getLogger(__name__)


class ThoughtTree:
    """Manages a complete thought-reasoning tree with depth/width bounds.

    Enforces ``max_depth`` and ``max_children`` constraints when adding
    nodes.  Exposes path retrieval, score-based pruning, and a text-based
    tree visualisation.

    Args:
        max_depth: Maximum allowed tree depth (root is depth 0).
        max_children: Maximum number of children per node.

    Example::

        tree = ThoughtTree(max_depth=3, max_children=5)
        root = tree.set_root("Initial problem")
        tree.add_thought(root, "Step 1", score=0.8)
    """

    def __init__(self, max_depth: int = 10, max_children: int = 5) -> None:
        self.root: ThoughtNode | None = None
        self.max_depth = max_depth
        self.max_children = max_children
        self._size = 0

    def __repr__(self) -> str:
        root_text = self.root.thought[:20] if self.root else "<empty>"
        return f"ThoughtTree(size={self._size}, root={root_text!r})"

    def set_root(self, thought: str, state: dict[str, Any] | None = None) -> ThoughtNode:
        """Create and set the root node, resetting the tree.

        Args:
            thought: The root thought text (typically the problem statement).
            state: Optional initial state dictionary.

        Returns:
            The newly created root ``ThoughtNode``.

        Example::

            tree = ThoughtTree()
            root = tree.set_root("Solve x^2 = 4", state={"domain": "math"})
        """
        self.root = ThoughtNode(thought=thought, state=state or {})
        self._size = 1
        logger.debug("set_root: %r", thought[:40])
        return self.root

    def add_thought(self, parent: ThoughtNode, thought: str,
                    state: dict[str, Any] | None = None,
                    score: float = 0.0) -> ThoughtNode | None:
        """Attach a new child thought to *parent* if constraints allow.

        Returns ``None`` silently when the parent is already at max depth or
        has reached the maximum number of children.

        Args:
            parent: The node to attach the new child to.
            thought: The reasoning text for the new step.
            state: Optional state dict (defaults to empty).
            score: Initial score for the new node.

        Returns:
            The new ``ThoughtNode``, or ``None`` if constraints are violated.

        Example::

            tree = ThoughtTree(max_depth=2)
            root = tree.set_root("Problem")
            child = tree.add_thought(root, "Step 1", score=0.7)
            assert child is not None and child.depth == 1
        """
        if parent.depth >= self.max_depth:
            logger.warning("add_thought: depth %d exceeded max_depth %d", parent.depth, self.max_depth)
            return None
        if len(parent.children) >= self.max_children:
            logger.warning("add_thought: max_children %d reached for parent at depth %d", self.max_children, parent.depth)
            return None
        child = ThoughtNode(thought=thought, state=state or {},
                            score=score, parent=parent)
        parent.children.append(child)
        self._size += 1
        logger.debug("add_thought: added %r at depth %d score=%.2f", thought[:30], child.depth, score)
        return child

    def get_best_path(self) -> list[ThoughtNode]:
        """Return the highest-scoring path from root to a leaf.

        At each level the child with the highest ``score`` is selected.

        Returns:
            A list of nodes from root to the best leaf, or an empty list
            if the tree has no root.

        Example::

            tree = ThoughtTree()
            root = tree.set_root("Q")
            tree.add_thought(root, "A", score=0.9)
            tree.add_thought(root, "B", score=0.3)
            best = tree.get_best_path()
            assert best[-1].thought == "A"
        """
        if self.root is None:
            return []
        path = [self.root]
        node = self.root
        while node.children:
            node = node.best_child() or node.children[0]
            path.append(node)
        return path

    def get_all_paths(self) -> list[list[ThoughtNode]]:
        """Return every root-to-leaf path via depth-first traversal.

        Returns:
            A list of paths, where each path is a list of ``ThoughtNode``.

        Example::

            tree = ThoughtTree()
            root = tree.set_root("Q")
            tree.add_thought(root, "A")
            tree.add_thought(root, "B")
            paths = tree.get_all_paths()
            assert len(paths) == 2
        """
        if self.root is None:
            return []
        paths: list[list[ThoughtNode]] = []

        def _dfs(node: ThoughtNode, current: list[ThoughtNode]) -> None:
            current.append(node)
            if node.is_leaf:
                paths.append(current[:])
            else:
                for child in node.children:
                    _dfs(child, current)
            current.pop()

        _dfs(self.root, [])
        return paths

    def prune(self, min_score: float) -> int:
        """Remove all nodes (and their subtrees) whose score < *min_score*.

        The root is never removed regardless of its score.

        Args:
            min_score: Minimum cumulative score to keep a node.

        Returns:
            The number of nodes removed.

        Example::

            tree = ThoughtTree()
            root = tree.set_root("Q")
            tree.add_thought(root, "Low", score=0.1)
            tree.add_thought(root, "High", score=0.9)
            removed = tree.prune(min_score=0.5)
            assert removed == 1
        """
        removed = 0

        def _prune(node: ThoughtNode) -> None:
            nonlocal removed
            to_keep: list[ThoughtNode] = []
            for child in node.children:
                if child.score >= min_score:
                    to_keep.append(child)
                else:
                    removed += _count_subtree(child)
            node.children = to_keep
            for child in to_keep:
                _prune(child)

        if self.root:
            _prune(self.root)
        self._size -= removed
        logger.info("prune: removed %d nodes (min_score=%.2f)", removed, min_score)
        return removed

    @property
    def size(self) -> int:
        """Return the current number of nodes in the tree.

        Returns:
            Total node count (including root).
        """
        return self._size

    def visualize(self) -> str:
        """Return a Unicode tree diagram with scores and visit counts.

        Each node shows its thought text (truncated to 40 chars) and either
        the average score with visits or the raw score.

        Returns:
            A multi-line string representation of the tree.

        Example::

            tree = ThoughtTree()
            root = tree.set_root("Problem")
            tree.add_thought(root, "Idea A", score=0.9)
            print(tree.visualize())
        """
        if self.root is None:
            return "<empty tree>"
        lines: list[str] = []

        def _render(node: ThoughtNode, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            score_str = f"[{node.score:.2f}]" if node.visits == 0 else f"[{node.avg_score:.2f} v{node.visits}]"
            lines.append(f"{prefix}{connector}{node.thought[:40]} {score_str}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _render(child, child_prefix, i == len(node.children) - 1)

        lines.append(f"{self.root.thought[:40]} [root]")
        for i, child in enumerate(self.root.children):
            _render(child, "", i == len(self.root.children) - 1)
        return "\n".join(lines)


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
