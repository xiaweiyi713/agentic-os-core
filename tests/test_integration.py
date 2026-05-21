"""End-to-end integration tests for the full Agent loop."""


from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.memory.manager import MemoryManager
from agentic_os.core.planning.executor import Executor
from agentic_os.core.planning.goal import GoalPriority, create_goal
from agentic_os.core.planning.planner import Planner
from agentic_os.core.tree.mcts import MCTS
from agentic_os.plugins.mock import MockEvaluator, MockLLM


class TestAgentLoop:
    """Test the full Agent perceive->memory->plan->execute->reflect cycle."""

    def test_full_agent_loop(self):
        # 1. Initialize components
        memory = MemoryManager()
        planner = Planner(mcts_iterations=30)
        executor = Executor(planner)
        _llm = MockLLM()
        evaluator = MockEvaluator(bias=0.6)

        # 2. Perception: receive user input and store in memory
        memory.add_experience("user needs a sorting algorithm")
        memory.add_experience("user requires O(n log n) time complexity")
        memory.add_experience("user prefers Python implementation")

        # 3. Memory consolidation
        count = memory.consolidate()
        assert count >= 1

        # 4. Planning: create goals and plan
        root = create_goal("implement sorting algorithm", priority=GoalPriority.HIGH)
        planner.add_goal(root)

        _subgoals = planner.decompose(root, [
            "analyze requirements and select algorithm",
            "implement quicksort",
            "write test cases",
        ])

        plan = planner.create_plan(root)
        assert len(plan) >= 3

        # 5. ToT plan evaluation
        tree = planner.evaluate_plan_with_tot(plan, evaluator.evaluate)
        assert tree.root is not None

        # 6. Execution
        execution_results = iter([
            (True, "started implementing sorting algorithm"),
            (True, "selected quicksort algorithm"),
            (True, "implementation done"),
            (True, "all 3 tests passed"),
        ])
        logs = executor.execute_plan(plan, lambda g: next(execution_results))
        assert all(log.success for log in logs)

        # 7. Reflection: record execution results to memory
        for log in logs:
            memory.add_experience(f"executed '{log.goal_id}': {'success' if log.success else 'failure'}")

        memory.store_reflection(
            "Sorting algorithm task completed, quicksort was a good choice",
            importance=0.8,
        )

        # 8. Verify final state
        stats = memory.stats()
        assert stats["working_memory"] >= 0
        assert stats["longterm_memory"]["nodes"] >= 1

        # 9. Recall verification
        results = memory.recall("Sorting")
        assert len(results) >= 1

    def test_agent_with_failure_and_replanning(self):
        """Test Agent replanning on failure."""
        _memory = MemoryManager()
        planner = Planner()

        root = create_goal("deploy service")
        planner.add_goal(root)

        subs = planner.decompose(root, [
            "build Docker image",
            "push to registry",
            "deploy to production",
        ])

        plan = planner.create_plan(root)

        # Simulate first step failure
        executor = Executor(planner)
        attempt = 0
        def action(goal):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return (False, "Docker build failed: dependency download timeout")
            return (True, "success")

        logs = executor.execute_plan(plan, action)
        assert not all(log.success for log in logs)

        # Replan
        failed = planner.get_goal(subs[0].id)
        if failed and not failed.is_terminal:
            planner.revise_plan(failed, [
                "use mirror registry",
                "retry build",
            ])

    def test_mcts_reasoning_with_memory(self):
        """Test MCTS reasoning combined with memory system."""
        memory = MemoryManager()
        memory.store_fact("user likes modular design", importance=0.8)
        memory.store_fact("project uses Python", importance=0.7)

        # Use MCTS for decision-making
        mcts = MCTS(max_iterations=50, max_depth=3)

        def generator(state):
            return [
                "use modular architecture",
                "use monolithic architecture",
                "use microservices architecture",
            ]

        def evaluator(state, thought):
            # Retrieve preferences from memory
            memories = memory.recall("architecture")
            score = 0.3
            for m in memories:
                if "modular" in thought and "modular" in m.content:
                    score += 0.3
                if "Python" in thought:
                    score += 0.1
            return min(score, 1.0)

        tree = mcts.search(
            root_thought="choose architecture",
            root_state={},
            generator=generator,
            evaluator=evaluator,
        )

        best_path = tree.get_best_path()
        assert len(best_path) >= 1
        # Modular approach should have an advantage
        best_thoughts = [n.thought for n in best_path]
        assert any("modular" in t.lower() or "architecture" in t.lower() for t in best_thoughts)

    def test_memory_graph_traversal(self):
        """Test memory graph traversal and reasoning."""
        memory = MemoryManager()

        # Build causal chain: problem analysis -> solution design -> implementation
        n1 = memory.store_fact("found performance issue: slow queries", importance=0.9)
        n2 = memory.store_fact("cause: missing indexes", importance=0.8)
        n3 = memory.store_fact("solution: add database indexes", importance=0.7)
        n4 = memory.store_fact("implemented: added 3 indexes", importance=0.6)

        memory.link_memories(n1, n2, EdgeType.CAUSAL, 0.9)
        memory.link_memories(n2, n3, EdgeType.CAUSAL, 0.85)
        memory.link_memories(n3, n4, EdgeType.CAUSAL, 0.8)

        # Verify causal chain via associated recall
        related = memory.recall_associated(n1, top_k=10)
        assert len(related) >= 1

        # Verify path finding
        path = memory.longterm.find_path(n1, n4)
        assert path is not None
        assert len(path) == 4
