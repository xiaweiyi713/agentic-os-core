"""Example: Custom evaluator and generator functions for MCTS.

Shows how to plug in domain-specific reasoning logic without
modifying the core engine.
"""

from agentic_os import MCTS
from agentic_os.plugins.mock import MockEvaluator


def main():
    print("=== Custom Evaluator & Generator for MCTS ===\n")

    # ── Scenario: Code Review Decision ──
    # Evaluate code changes with domain-specific heuristics

    def code_review_generator(state):
        """Generate possible review actions based on current context."""
        depth = state.get("depth", 0)
        if depth == 0:
            return ["check_syntax", "check_logic", "check_style"]
        elif depth == 1:
            last = state.get("last_thought", "")
            if "syntax" in last:
                return ["run_linter", "run_type_checker"]
            elif "logic" in last:
                return ["trace_data_flow", "check_edge_cases"]
            else:
                return ["check_naming", "check_formatting"]
        return []

    def code_review_evaluator(state, thought):
        """Score review actions based on defect-detection value."""
        scores = {
            "run_linter": 0.85,
            "run_type_checker": 0.8,
            "trace_data_flow": 0.75,
            "check_edge_cases": 0.9,
            "check_syntax": 0.4,
            "check_logic": 0.7,
            "check_style": 0.3,
            "check_naming": 0.25,
            "check_formatting": 0.2,
        }
        return scores.get(thought, 0.1)

    mcts = MCTS(exploration_weight=1.0, max_iterations=50, max_depth=2)
    tree = mcts.search(
        root_thought="Code Review Decision",
        root_state={"depth": 0, "file": "auth.py"},
        generator=code_review_generator,
        evaluator=code_review_evaluator,
    )

    print("Decision tree:")
    print(tree.visualize())
    print()

    best_path = tree.get_best_path()
    print("Optimal review path:")
    for i, node in enumerate(best_path):
        print(f"  Step {i}: {node.thought} (score: {node.avg_score:.2f})")

    # ── Scenario: Using MockEvaluator with custom bias ──
    print("\n--- Mock Evaluator with bias=0.8 ---\n")

    evaluator = MockEvaluator(bias=0.8)
    mcts2 = MCTS(max_iterations=30, max_depth=3)

    def simple_generator(state):
        return ["approach_A", "approach_B", "approach_C"]

    tree2 = mcts2.search(
        root_thought="Generic problem",
        root_state={},
        generator=simple_generator,
        evaluator=evaluator.evaluate,
    )

    best = tree2.get_best_path()
    print(f"Best path length: {len(best)}")
    for node in best:
        print(f"  → {node.thought} (visits: {node.visits}, avg_score: {node.avg_score:.2f})")

    print(f"\nEvaluator was called {len(evaluator.call_log)} times")


if __name__ == "__main__":
    main()
