"""Planner - decomposes goals into executable sub-goal trees using MCTS / ToT."""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.planning.goal import Goal, GoalPriority, create_goal
from agentic_os.core.tree.mcts import MCTS, Evaluator
from agentic_os.core.tree.thought_tree import ThoughtTree

logger = logging.getLogger(__name__)


class Planner:
    """Breaks complex goals into executable sub-goal trees.

    Internally uses MCTS for plan evaluation and supports topological
    ordering of goals respecting dependency constraints.
    """

    def __init__(self, mcts_iterations: int = 200) -> None:
        """Initialize the planner.

        Args:
            mcts_iterations: Max MCTS iterations when evaluating plans.
        """
        self._mcts = MCTS(max_iterations=mcts_iterations, max_depth=5)
        self._goals: dict[str, Goal] = {}

    def add_goal(self, goal: Goal) -> str:
        """Register a goal with the planner.

        Args:
            goal: The ``Goal`` to register.

        Returns:
            The goal ID.
        """
        self._goals[goal.id] = goal
        return goal.id

    def get_goal(self, goal_id: str) -> Goal | None:
        """Look up a registered goal by ID.

        Args:
            goal_id: The goal identifier.

        Returns:
            The ``Goal`` if found, else ``None``.
        """
        return self._goals.get(goal_id)

    def decompose(self, goal: Goal, sub_descriptions: list[str],
                  priority: GoalPriority | None = None) -> list[Goal]:
        """Decompose a goal into child sub-goals.

        Each sub-goal is registered, linked to the parent via
        ``parent_id`` and ``subgoals``.

        Args:
            goal: The parent goal to decompose.
            sub_descriptions: Description strings for each sub-goal.
            priority: Override priority for sub-goals. Defaults to the
                parent's priority.

        Returns:
            List of newly created sub-goals.
        """
        subgoals: list[Goal] = []
        for i, desc in enumerate(sub_descriptions):
            sub = create_goal(
                desc,
                priority=priority or goal.priority,
                step_index=i,
            )
            sub.parent_id = goal.id
            goal.subgoals.append(sub.id)
            self._goals[sub.id] = sub
            subgoals.append(sub)
        logger.debug("decompose: created %d subgoals for %s", len(subgoals), goal.id)
        return subgoals

    def create_plan(self, root_goal: Goal) -> list[Goal]:
        """Generate a topologically sorted execution plan from a goal tree.

        Goals are ordered by priority (descending) while respecting
        dependency constraints. Cyclic dependencies are broken by
        picking the first pending goal.

        Args:
            root_goal: The root of the goal tree.

        Returns:
            Ordered list of non-terminal ``Goal`` instances ready for
            sequential execution.
        """
        # Collect all incomplete goals
        all_goals = self._collect_goals(root_goal)

        # Sort by priority and dependencies
        pending = [g for g in all_goals if not g.is_terminal]
        pending.sort(key=lambda g: g.priority.value, reverse=True)

        # Topological sort (by dependencies)
        ordered: list[Goal] = []
        completed_ids: set[str] = set()

        while pending:
            ready = [
                g for g in pending
                if all(dep in completed_ids for dep in g.dependencies)
            ]
            if not ready:
                # Cycle detected, force-first
                ready = [pending[0]]
            for g in ready:
                ordered.append(g)
                completed_ids.add(g.id)
                pending.remove(g)

        logger.info("create_plan: %d goals in plan from root %s", len(ordered), root_goal.id)
        return ordered

    def evaluate_plan_with_tot(self, plan: list[Goal], evaluator: Evaluator) -> ThoughtTree:
        """Evaluate plan quality using a Tree-of-Thought search.

        Args:
            plan: Ordered list of goals to evaluate.
            evaluator: An ``Evaluator`` instance that scores each thought.

        Returns:
            A ``ThoughtTree`` capturing the evaluation search space.
        """
        root_state = {
            "plan_steps": [g.description for g in plan],
            "current_step": 0,
        }

        def generator(state: dict[str, Any]) -> list[str]:
            steps = state.get("plan_steps", [])
            current = state.get("current_step", 0)
            if current >= len(steps):
                return []
            # Generate candidate evaluations for the current step
            step = steps[current]
            return [
                f"Step '{step}' is feasible and reasonable",
                f"Step '{step}' needs more resources",
                f"Step '{step}' may encounter obstacles",
            ]

        tree = self._mcts.search(
            root_thought=f"Evaluate plan ({len(plan)} steps)",
            root_state=root_state,
            generator=generator,
            evaluator=evaluator,
        )
        return tree

    def revise_plan(self, failed_goal: Goal,
                    new_sub_descriptions: list[str]) -> list[Goal]:
        """Mark a goal as failed and replace it with new sub-goals.

        The new sub-goals inherit the original goal's dependencies.

        Args:
            failed_goal: The goal that failed.
            new_sub_descriptions: Descriptions for replacement sub-goals.

        Returns:
            List of newly created replacement sub-goals.
        """
        failed_goal.fail("Plan revision")

        new_subs = self.decompose(failed_goal, new_sub_descriptions)
        # New sub-goals inherit the original goal's dependencies
        for sub in new_subs:
            sub.dependencies = list(failed_goal.dependencies)
        return new_subs

    def _collect_goals(self, goal: Goal) -> list[Goal]:
        """Recursively collect all goals in the goal tree.

        Args:
            goal: Root goal to start from.

        Returns:
            Flat list of all goals (root + descendants).
        """
        result = [goal]
        for sub_id in goal.subgoals:
            sub = self._goals.get(sub_id)
            if sub:
                result.extend(self._collect_goals(sub))
        return result

    @property
    def all_goals(self) -> list[Goal]:
        """list[Goal]: All goals registered with the planner."""
        return list(self._goals.values())
