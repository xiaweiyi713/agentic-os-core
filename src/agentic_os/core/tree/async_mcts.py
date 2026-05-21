"""Async Monte Carlo Tree Search (MCTS) with concurrent rollout evaluation.

Implements the same four-phase MCTS loop as the synchronous version but
evaluates all expansion candidates concurrently via ``asyncio.gather``,
controlled by a semaphore for bounded parallelism.

Phases per iteration:

1. **Selection** -- descend from root using UCB1 until a leaf is reached.
2. **Expansion** -- generate candidate thoughts, evaluate them all in
   parallel, and attach the best-scoring one as a child.
3. **Simulation** -- score the expanded child with the evaluator.
4. **Backpropagation** -- propagate the reward up to the root.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any

from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree

logger = logging.getLogger(__name__)

AsyncEvaluator = Callable[[dict[str, Any], str], Awaitable[float]]
AsyncCandidateGenerator = Callable[[dict[str, Any]], Awaitable[list[str]]]


class AsyncMCTS:
    """Async Monte Carlo Tree Search with concurrent rollouts.

    Uses ``asyncio.gather`` for parallel evaluation of candidate thoughts
    during the expansion phase, controlled by a semaphore.

    Args:
        exploration_weight: UCB1 exploration coefficient.
        max_iterations: Total MCTS iterations.
        max_depth: Maximum tree depth.
        concurrency: Max concurrent async evaluations (default 8).

    Example::

        mcts = AsyncMCTS(concurrency=4, max_iterations=100)
        tree = await mcts.search(
            root_thought="Solve the puzzle",
            root_state={},
            generator=my_async_generator,
            evaluator=my_async_evaluator,
        )
    """

    def __init__(self, exploration_weight: float = 1.414,
                 max_iterations: int = 1000,
                 max_depth: int = 10,
                 concurrency: int = 8) -> None:
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.concurrency = concurrency

    def __repr__(self) -> str:
        return (f"AsyncMCTS(c={self.exploration_weight:.2f}, "
                f"iters={self.max_iterations}, depth={self.max_depth}, "
                f"concurrency={self.concurrency})")

    async def search(self, root_thought: str, root_state: dict[str, Any],
                     generator: AsyncCandidateGenerator,
                     evaluator: AsyncEvaluator) -> ThoughtTree:
        """Execute async MCTS search, returns populated ThoughtTree.

        Phases per iteration:

        1. **Selection**: walk down from root via UCB1 to find a
           leaf or an unvisited child.
        2. **Expansion**: use *generator* to produce candidates, evaluate
           them all concurrently, and attach the best-scoring child.
        3. **Simulation**: score the expanded child with *evaluator*.
        4. **Backpropagation**: add the reward to every ancestor's
           ``score`` and increment ``visits``.

        Args:
            root_thought: The initial problem or prompt text.
            root_state: The starting state dictionary.
            generator: Async function producing candidate thought strings.
            evaluator: Async function scoring a (state, thought) pair,
                returning a float in ``[0, 1]``.

        Returns:
            A ``ThoughtTree`` populated with the search results.
        """
        tree = ThoughtTree(max_depth=self.max_depth, max_children=20)
        root = tree.set_root(root_thought, root_state)
        root.visits = 1

        for _ in range(self.max_iterations):
            # 1. Selection
            node = self._select(root)
            if node.depth >= self.max_depth:
                continue

            # 2. Expansion (async generator + concurrent eval)
            candidates = await generator(node.state)
            if not candidates:
                continue

            expanded = await self._expand_batch(tree, node, candidates, evaluator)
            if expanded is None:
                continue

            # 3. Simulation
            reward = await self._simulate(expanded, evaluator)

            # 4. Backpropagation
            self._backpropagate(expanded, reward)

        logger.info(
            "search: completed %d iterations, tree size=%d",
            self.max_iterations, tree.size,
        )
        return tree

    async def _expand_batch(self, tree: ThoughtTree, node: ThoughtNode,
                            candidates: list[str],
                            evaluator: AsyncEvaluator) -> ThoughtNode | None:
        """Concurrently evaluate all candidates and add the best one.

        Uses ``asyncio.Semaphore`` to limit concurrency and
        ``asyncio.gather`` to evaluate all candidates in parallel.
        The candidate with the highest score is added to the tree.

        Args:
            tree: The tree being built.
            node: The node to expand.
            candidates: Thought strings produced by the generator.
            evaluator: Async scoring function for (state, thought).

        Returns:
            The best-expanded ``ThoughtNode``, or ``None`` if all fail.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _evaluate_one(thought: str) -> tuple[str, float]:
            async with semaphore:
                new_state = {**node.state, "last_thought": thought}
                score = await evaluator(new_state, thought)
                return thought, score

        logger.debug(
            "_expand_batch: %d candidates for node at depth %d",
            len(candidates), node.depth,
        )
        results = await asyncio.gather(*(_evaluate_one(c) for c in candidates))

        # Sort by score descending, pick the best
        results_sorted = sorted(results, key=lambda r: r[1], reverse=True)
        for thought, score in results_sorted:
            new_state = {**node.state, "last_thought": thought}
            child = tree.add_thought(node, thought, new_state, score=score)
            if child is not None:
                return child
        return None

    async def _simulate(self, node: ThoughtNode,
                        evaluator: AsyncEvaluator) -> float:
        """Simulation (rollout) phase: estimate the value of *node*.

        In this implementation the evaluator directly scores the node
        rather than performing a stochastic rollout.

        Args:
            node: The newly expanded node to evaluate.
            evaluator: Async scoring function returning a float in ``[0, 1]``.

        Returns:
            A reward value, typically in ``[0, 1]``.
        """
        return await evaluator(node.state, node.thought)

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
            if any(c.visits == 0 for c in node.children):
                return next(c for c in node.children if c.visits == 0)
            node = max(node.children, key=lambda c: self._ucb1(c))
        return node

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
