"""Monte Carlo Tree Search (MCTS) for structured reasoning.

Implements the classic four-phase loop:

1. **Selection** -- descend from root using UCB1 until a leaf is reached.
2. **Expansion** -- generate candidate thoughts and add one child.
3. **Simulation** -- evaluate the newly added child with a heuristic.
4. **Backpropagation** -- propagate the reward up to the root.

The UCB1 formula balances exploitation and exploration:

    UCB1 = avg_score + c * sqrt(ln(N_parent) / N_child)

where *c* is the exploration weight (default ``sqrt(2)``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree

logger = logging.getLogger(__name__)

# Evaluator signature: (state, thought) -> float in [0, 1]
Evaluator = Callable[[dict[str, Any], str], float]

# Candidate generator signature: (state) -> list of thought strings
CandidateGenerator = Callable[[dict[str, Any]], list[str]]


class MCTS:
    """Monte Carlo Tree Search engine for the Tree of Thoughts.

    Repeatedly applies the four-phase MCTS loop (Selection, Expansion,
    Simulation, Backpropagation) for a configurable number of iterations,
    using the UCB1 formula to balance exploration and exploitation.

    Args:
        exploration_weight: Exploration coefficient *c* in UCB1.
            Larger values favour visiting unexplored nodes.
        max_iterations: Total number of MCTS iterations to run.
        max_depth: Maximum allowed tree depth during search.

    Example::

        mcts = MCTS(exploration_weight=1.414, max_iterations=500)
        tree = mcts.search(
            root_thought="Solve the puzzle",
            root_state={},
            generator=my_generator,
            evaluator=my_evaluator,
        )
        best = tree.get_best_path()
    """

    def __init__(self, exploration_weight: float = 1.414,
                 max_iterations: int = 1000,
                 max_depth: int = 10) -> None:
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth

    def __repr__(self) -> str:
        return (f"MCTS(c={self.exploration_weight:.2f}, "
                f"iters={self.max_iterations}, depth={self.max_depth})")

    def search(self, root_thought: str, root_state: dict[str, Any],
               generator: CandidateGenerator,
               evaluator: Evaluator) -> ThoughtTree:
        """Execute a full MCTS search and return the built tree.

        Phases per iteration:

        1. **Selection**: walk down from root via UCB1 to find a
           leaf or an unvisited child.
        2. **Expansion**: use *generator* to produce candidates from the
           selected node's state and attach the first viable child.
        3. **Simulation**: score the expanded child with *evaluator*.
        4. **Backpropagation**: add the reward to every ancestor's
           ``score`` and increment ``visits``.

        Args:
            root_thought: The initial problem or prompt text.
            root_state: The starting state dictionary.
            generator: Produces candidate thought strings from a state.
            evaluator: Scores a (state, thought) pair, returning a
                float in ``[0, 1]``.

        Returns:
            A ``ThoughtTree`` populated with the search results.

        Example::

            tree = mcts.search(
                "What is 2+2?", {"turn": 0},
                generator=lambda s: ["3", "4", "5"],
                evaluator=lambda s, t: 1.0 if t == "4" else 0.0,
            )
        """
        tree = ThoughtTree(max_depth=self.max_depth, max_children=20)
        root = tree.set_root(root_thought, root_state)
        root.visits = 1

        for _ in range(self.max_iterations):
            # 1. Selection: follow UCB1-optimal path to an expandable node
            node = self._select(root)
            if node.depth >= self.max_depth:
                continue

            # 2. Expansion: generate candidates and add a child
            candidates = generator(node.state)
            if not candidates:
                continue

            expanded = self._expand(tree, node, candidates, evaluator)
            if expanded is None:
                continue

            # 3. Simulation: evaluate the newly expanded node
            reward = self._simulate(expanded, evaluator)

            # 4. Backpropagation: update scores and visits up to root
            self._backpropagate(expanded, reward)

        logger.info("search: completed %d iterations, tree size=%d", self.max_iterations, tree.size)
        return tree

    def _select(self, node: ThoughtNode) -> ThoughtNode:
        """Selection phase: descend the tree via UCB1 until a leaf.

        If any child of the current node is unvisited (``visits == 0``),
        that child is returned immediately to guarantee exploration.

        Args:
            node: The starting node (typically the root).

        Returns:
            A leaf node or an unvisited child ready for expansion.
        """
        while not node.is_leaf:
            for c in node.children:
                if c.visits == 0:
                    return c
            node = max(node.children, key=lambda c: self._ucb1(c))
        return node

    def _expand(self, tree: ThoughtTree, node: ThoughtNode,
                candidates: list[str], evaluator: Evaluator) -> ThoughtNode | None:
        """Expansion phase: attach the first viable candidate as a child.

        Each candidate is scored and added via ``tree.add_thought``; the
        first successfully added child is returned.

        Args:
            tree: The tree being built.
            node: The node to expand.
            candidates: Thought strings produced by the generator.
            evaluator: Scoring function for (state, thought).

        Returns:
            The first successfully expanded ``ThoughtNode``, or ``None``.
        """
        logger.debug("_expand: %d candidates for node at depth %d", len(candidates), node.depth)
        for thought in candidates:
            new_state = {**node.state, "last_thought": thought}
            score = evaluator(new_state, thought)
            child = tree.add_thought(node, thought, new_state, score=score)
            if child is not None:
                return child
        return None

    def _simulate(self, node: ThoughtNode, evaluator: Evaluator) -> float:
        """Simulation (rollout) phase: estimate the value of *node*.

        In this implementation the evaluator directly scores the node
        rather than performing a stochastic rollout.

        Args:
            node: The newly expanded node to evaluate.
            evaluator: Scoring function returning a float in ``[0, 1]``.

        Returns:
            A reward value, typically in ``[0, 1]``.
        """
        return evaluator(node.state, node.thought)

    def _backpropagate(self, node: ThoughtNode, reward: float) -> None:
        """Backpropagation phase: update ancestors with the reward.

        Walks from *node* to the root, incrementing ``visits`` and
        adding *reward* to ``score`` at each step.

        Args:
            node: The leaf node from which to start propagation.
            reward: The reward value obtained during simulation.
        """
        current: ThoughtNode | None = node
        while current is not None:
            current.visits += 1
            current.score += reward
            current = current.parent

    def _ucb1(self, node: ThoughtNode) -> float:
        """Compute the UCB1 score for a node.

        The formula is::

            UCB1 = avg_score + c * sqrt(ln(N_parent) / N_child)

        Returns ``inf`` for unvisited nodes to guarantee exploration.

        Args:
            node: The node whose UCB1 value is being computed.

        Returns:
            The UCB1 score (``float``).  Unvisited nodes return ``inf``.
        """
        if node.visits == 0:
            return float("inf")
        if node.parent is None or node.parent.visits == 0:
            return node.avg_score
        exploitation = node.avg_score
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )
        return exploitation + exploration
