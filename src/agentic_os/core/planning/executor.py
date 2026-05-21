"""Plan executor - runs goals sequentially and records results."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_os.core.planning.goal import Goal
from agentic_os.core.planning.planner import Planner

logger = logging.getLogger(__name__)

ActionResult = tuple[bool, str]  # (success, result_message)
"""Type alias for action results: a (success, message) tuple."""


@dataclass
class ExecutionLog:
    """Record of a single goal execution outcome.

    Attributes:
        goal_id: ID of the executed goal.
        success: Whether execution succeeded.
        result: Human-readable result or error message.
        metadata: Optional extra information (e.g. retry count).
    """
    goal_id: str
    success: bool
    result: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Executor:
    """Executes a plan of goals sequentially, recording results.

    Stops on the first failure. Use ``execute_with_retry`` for
    automatic retries.
    """

    def __init__(self, planner: Planner) -> None:
        """Initialize the executor.

        Args:
            planner: The ``Planner`` instance that created the plan.
        """
        self._planner = planner
        self._logs: list[ExecutionLog] = []

    def execute_plan(self, plan: list[Goal],
                     action_fn: Callable[[Goal], ActionResult]) -> list[ExecutionLog]:
        """Execute goals in order; stop on first failure.

        Args:
            plan: Ordered list of goals to execute.
            action_fn: Callable that takes a ``Goal`` and returns an
                ``ActionResult`` (success, message).

        Returns:
            List of ``ExecutionLog`` entries for each attempted goal.

        Examples:
            >>> logs = executor.execute_plan(plan, lambda g: (True, "ok"))
        """
        self._logs.clear()
        for goal in plan:
            if goal.is_terminal:
                continue
            goal.start()
            success, result = action_fn(goal)
            if success:
                goal.complete(result)
            else:
                goal.fail(result)
            log = ExecutionLog(goal_id=goal.id, success=success, result=result)
            self._logs.append(log)

            if not success:
                # Execution failed, stop remaining steps
                break
        logger.info("execute_plan: %d success, %d failure", self.success_count, self.failure_count)
        return self._logs

    def execute_with_retry(self, plan: list[Goal],
                           action_fn: Callable[[Goal], ActionResult],
                           max_retries: int = 2) -> list[ExecutionLog]:
        """Execute goals with automatic retries on failure.

        Each goal is attempted up to ``max_retries + 1`` times.

        Args:
            plan: Ordered list of goals to execute.
            action_fn: Callable that takes a ``Goal`` and returns an
                ``ActionResult``.
            max_retries: Number of retries per goal. Defaults to 2.

        Returns:
            List of ``ExecutionLog`` entries.
        """
        self._logs.clear()
        for goal in plan:
            if goal.is_terminal:
                continue
            goal.start()

            success = False
            result = ""
            last_attempt = 0
            for attempt in range(max_retries + 1):
                last_attempt = attempt
                success, result = action_fn(goal)
                if success:
                    break

            if success:
                goal.complete(result)
            else:
                goal.fail(f"Failed after {max_retries} retries: {result}")
                logger.warning("execute_with_retry: goal %s failed after %d retries", goal.id, last_attempt + 1)
            log = ExecutionLog(
                goal_id=goal.id, success=success, result=result,
                metadata={"attempts": last_attempt + 1},
            )
            self._logs.append(log)

            if not success:
                break
        return self._logs

    @property
    def logs(self) -> list[ExecutionLog]:
        """list[ExecutionLog]: Snapshot of the current execution logs."""
        return list(self._logs)

    @property
    def success_count(self) -> int:
        """int: Number of goals executed successfully."""
        return sum(1 for log in self._logs if log.success)

    @property
    def failure_count(self) -> int:
        """int: Number of goals that failed."""
        return sum(1 for log in self._logs if not log.success)
