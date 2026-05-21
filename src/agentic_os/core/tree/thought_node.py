"""Fundamental building block for the Tree of Thoughts reasoning framework.

Each node represents a single reasoning step with an associated state,
score, and parent-child relationships forming a tree structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtNode:
    """A single step in a reasoning chain within the Tree of Thoughts.

    Stores the thought text, mutable state dict, cumulative score, visit
    count (for MCTS), and parent/child pointers. Designed as a lightweight
    data container; tree operations live in ``ThoughtTree``.

    Attributes:
        thought: The natural-language reasoning text for this step.
        state: Arbitrary key-value state carried through the chain.
        score: Cumulative reward (updated by backpropagation).
        visits: Number of times this node was visited (MCTS).
        parent: Link to the parent node (``None`` for root).
        children: Ordered list of child nodes.
    """

    thought: str
    state: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    visits: int = 0
    parent: ThoughtNode | None = field(default=None, repr=False)
    children: list[ThoughtNode] = field(default_factory=list)
    _depth: int = field(default=0, repr=False)

    @property
    def is_root(self) -> bool:
        """Return ``True`` if this node has no parent.

        Returns:
            Whether the node is the root of its tree.

        Example::

            root = ThoughtNode(thought="Start")
            assert root.is_root is True
        """
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` if this node has no children.

        Returns:
            Whether the node is a leaf (no expansions yet).

        Example::

            node = ThoughtNode(thought="Step 1")
            assert node.is_leaf is True
        """
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Return the depth of this node (root is 0).

        Returns:
            The number of edges from this node to the root.

        Example::

            root = ThoughtNode(thought="Root")
            child = ThoughtNode(thought="Child", parent=root)
            root.children.append(child)
            assert child.depth == 1
        """
        if self.parent is not None and self._depth == 0:
            self._depth = self.parent.depth + 1
        return self._depth

    def best_child(self) -> ThoughtNode | None:
        """Return the child with the highest cumulative score.

        Ties are broken by insertion order (first wins via ``max``).

        Returns:
            The highest-scoring child, or ``None`` if this node is a leaf.

        Example::

            parent = ThoughtNode(thought="Parent")
            a = ThoughtNode(thought="A", score=0.3, parent=parent)
            b = ThoughtNode(thought="B", score=0.9, parent=parent)
            parent.children = [a, b]
            assert parent.best_child() is b
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.score)

    def path_from_root(self) -> list[ThoughtNode]:
        """Return the ordered path from the root to this node (inclusive).

        Returns:
            A list of ``ThoughtNode`` starting at the root and ending at
            ``self``.

        Example::

            root = ThoughtNode(thought="Root")
            child = ThoughtNode(thought="Child", parent=root)
            root.children.append(child)
            assert child.path_from_root() == [root, child]
        """
        path: list[ThoughtNode] = []
        node: ThoughtNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    @property
    def avg_score(self) -> float:
        """Return the average score (cumulative score / visits).

        Returns ``score`` unchanged when ``visits`` is 0.

        Returns:
            The mean reward per visit.

        Example::

            node = ThoughtNode(thought="x", score=4.0, visits=2)
            assert node.avg_score == 2.0
        """
        return self.score / max(self.visits, 1)

    def subtree_size(self) -> int:
        """Return the total number of nodes in this subtree (inclusive).

        Returns:
            Number of nodes in the subtree rooted at this node.

        Example::

            root = ThoughtNode(thought="Root")
            root.children.append(ThoughtNode(thought="A"))
            root.children.append(ThoughtNode(thought="B"))
            assert root.subtree_size() == 3
        """
        count = 1
        for child in self.children:
            count += child.subtree_size()
        return count
