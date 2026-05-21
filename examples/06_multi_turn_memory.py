"""Example: Multi-turn conversation memory with consolidation.

Demonstrates how an agent accumulates experiences across turns,
periodically consolidates and reflects, building a growing knowledge graph.
"""

from agentic_os import EdgeType, MemoryManager


def simulate_turn(memory: MemoryManager, user_input: str, agent_response: str) -> None:
    """Simulate one conversation turn."""
    memory.add_experience(f"User: {user_input}")
    memory.add_experience(f"Agent: {agent_response}")


def main():
    print("=== Multi-turn Memory with Consolidation ===\n")

    memory = MemoryManager(working_capacity=30)

    # ── Turn 1: Python basics ──
    print("--- Turn 1: Python basics ---")
    simulate_turn(memory,
        "What are Python decorators?",
        "Decorators are higher-order functions that modify other functions.")
    memory.store_fact("Python decorators use @ syntax", importance=0.7)

    # ── Turn 2: Follow-up ──
    print("--- Turn 2: Follow-up ---")
    simulate_turn(memory,
        "How do I write a custom decorator?",
        "Define a function that takes another function, wraps it, returns wrapper.")
    memory.store_fact("Custom decorators typically use functools.wraps", importance=0.75)

    # ── Consolidate after 2 turns ──
    count = memory.consolidate()
    print(f"\nConsolidated {count} items to long-term memory")

    # ── Turn 3: New topic ──
    print("\n--- Turn 3: New topic ---")
    simulate_turn(memory,
        "How does MCTS work?",
        "MCTS uses 4 phases: Selection, Expansion, Simulation, Backpropagation.")
    memory.store_fact("UCB1 formula: score/N + c*sqrt(ln(N_parent)/N)", importance=0.8)

    # ── Turn 4: Connection ──
    print("--- Turn 4: Connection ---")
    simulate_turn(memory,
        "Can MCTS be used for code generation?",
        "Yes, treat each partial code as a node, evaluate with tests.")

    # Link facts across topics
    dec_id = memory.store_fact("Tree-of-Thoughts applies MCTS to LLM reasoning", importance=0.85)
    mcts_nodes = memory.recall("MCTS")
    if mcts_nodes:
        memory.link_memories(mcts_nodes[0].id, dec_id, EdgeType.DERIVED_FROM, 0.9)

    # ── Reflect on all experiences ──
    count = memory.consolidate()
    print(f"\nConsolidated {count} more items")
    reflection_ids = memory.reflect()
    print(f"Generated {len(reflection_ids)} reflections")

    # ── Turn 5: Recall test ──
    print("\n--- Turn 5: Recall Test ---")
    simulate_turn(memory,
        "What have we discussed about decorators?",
        "Let me recall...")

    # Recall by different strategies
    print("\nKeyword recall 'decorator':")
    for node in memory.recall("decorator", top_k=5):
        print(f"  [{node.type.value}] {node.content[:60]}")

    print("\nMost important memories:")
    for node in memory.recall_by_importance(top_k=5):
        print(f"  [{node.type.value}] {node.content[:60]} (importance: {node.importance:.2f})")

    print("\nMost recent memories:")
    for node in memory.recall_by_recency(top_k=5):
        print(f"  [{node.type.value}] {node.content[:60]}")

    # ── Final stats ──
    stats = memory.stats()
    print("\nFinal stats:")
    print(f"  Working memory: {stats['working_memory']} items")
    print(f"  Long-term graph: {stats['longterm_memory']['nodes']} nodes, "
          f"{stats['longterm_memory']['edges']} edges")
    print(f"  Memory types: {stats['longterm_memory'].get('types', {})}")

    # ── Forgetting low-value memories ──
    forgotten = memory.forget(min_importance=0.2)
    print(f"\nForgot {forgotten} low-importance memories")


if __name__ == "__main__":
    main()
