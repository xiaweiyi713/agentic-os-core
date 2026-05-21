"""Example: Persistence — save and restore knowledge graph state.

Shows JSON serialization for persistence across sessions.
"""

import json
import tempfile
from pathlib import Path

from agentic_os import (
    EdgeType,
    KnowledgeGraph,
    create_episode,
    create_fact,
    create_reflection,
)


def main():
    print("=== Persistence: Save & Restore Knowledge Graph ===\n")

    # ── Build a knowledge graph ──
    kg = KnowledgeGraph()

    e1 = create_episode("Deployed v1.0 to production")
    e2 = create_episode("Received user feedback: slow query on dashboard")
    f1 = create_fact("Dashboard query joins 5 tables without indexes", importance=0.8)
    f2 = create_fact("Adding composite index on (user_id, created_at) improves query 10x", importance=0.85)
    r1 = create_reflection("Always index columns used in WHERE and JOIN clauses", importance=0.9)

    for n in [e1, e2, f1, f2, r1]:
        kg.add_node(n)

    kg.add_edge(e1.id, e2.id, EdgeType.TEMPORAL, 0.9)
    kg.add_edge(e2.id, f1.id, EdgeType.DERIVED_FROM, 0.8)
    kg.add_edge(f1.id, f2.id, EdgeType.CAUSAL, 0.85)
    kg.add_edge(f2.id, r1.id, EdgeType.DERIVED_FROM, 0.9)

    print(f"Original graph: {kg.node_count} nodes, {kg.edge_count} edges")

    # ── Save to file ──
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
        json.dump(kg.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved to: {path}")
    print(f"File size: {Path(path).stat().st_size} bytes")

    # ── Restore from file ──
    with open(path) as f:
        data = json.load(f)

    kg2 = KnowledgeGraph.from_dict(data)
    print(f"\nRestored graph: {kg2.node_count} nodes, {kg2.edge_count} edges")

    # ── Verify fidelity ──
    print("\nVerification:")
    for node in kg2.nodes:
        orig = kg.get_node(node.id)
        assert orig is not None, f"Missing node: {node.id}"
        assert orig.content == node.content, f"Content mismatch: {node.id}"
        print(f"  ✓ [{node.type.value}] {node.content[:50]}")

    for node in kg2.nodes:
        for edge in kg2.out_edges(node.id):
            orig_edge = kg.get_edge(edge.source_id, edge.target_id)
            assert orig_edge is not None, f"Missing edge: {edge.key}"
            assert orig_edge.weight == edge.weight, f"Weight mismatch: {edge.key}"
    print(f"  ✓ All {kg2.edge_count} edges verified")

    # Cleanup
    Path(path).unlink()
    print(f"\nCleaned up: {path}")


if __name__ == "__main__":
    main()
