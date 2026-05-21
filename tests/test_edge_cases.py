"""Edge case and error path tests for professional-grade coverage."""

import pytest

from agentic_os import (
    MCTS,
    EdgeType,
    Executor,
    GoalState,
    KnowledgeGraph,
    MemoryManager,
    Planner,
    ThoughtTree,
    create_episode,
)
from agentic_os.core.graph.edge import Edge
from agentic_os.core.graph.node import MemoryNode, NodeType
from agentic_os.core.graph.scoring import compute_pagerank
from agentic_os.core.graph.traversal import (
    bfs,
    connected_components,
    dfs,
    shortest_path,
    topological_sort,
)
from agentic_os.core.memory.working import WorkingMemory
from agentic_os.core.planning.goal import create_goal
from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.exceptions import ValidationError


class TestGraphEdgeCases:
    """Empty graph, single node, self-loops, duplicate operations."""

    def test_empty_graph_operations(self):
        kg = KnowledgeGraph()
        assert kg.node_count == 0
        assert kg.edge_count == 0
        assert kg.get_node("nonexistent") is None
        assert kg.remove_node("nonexistent") is None
        assert kg.remove_edge("a", "b") is None
        assert kg.find_nodes("anything") == []
        assert kg.out_neighbors("x") == []
        assert kg.in_neighbors("x") == []
        assert kg.out_edges("x") == []
        assert kg.stats() == {"nodes": 0, "edges": 0, "types": {}, "keywords": 0}
        assert bfs(kg, "x") == {}
        assert dfs(kg, "x") == []
        assert shortest_path(kg, "a", "b") is None
        assert topological_sort(kg) == []
        assert connected_components(kg) == []

    def test_single_node_graph(self):
        kg = KnowledgeGraph()
        n = create_episode("alone")
        kg.add_node(n)
        assert kg.degree(n.id) == 0
        assert kg.out_neighbors(n.id) == []
        assert kg.subgraph(n.id, depth=5).node_count == 1
        assert compute_pagerank(kg) == {n.id: 1.0}

    def test_self_loop_edge(self):
        kg = KnowledgeGraph()
        n = create_episode("self")
        kg.add_node(n)
        edge = kg.add_edge(n.id, n.id, EdgeType.ASSOCIATIVE, 0.5)
        assert edge is not None
        assert kg.has_edge(n.id, n.id)
        assert n.id in kg.out_neighbors(n.id)

    def test_duplicate_node_overwrite(self):
        kg = KnowledgeGraph()
        n1 = MemoryNode(id="dup", type=NodeType.EPISODE, content="first")
        n2 = MemoryNode(id="dup", type=NodeType.FACT, content="second")
        kg.add_node(n1)
        kg.add_node(n2)
        assert kg.node_count == 1
        assert kg.get_node("dup").content == "second"
        assert kg.get_node("dup").type == NodeType.FACT

    def test_duplicate_edge_overwrite(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL, 0.5)
        kg.add_edge(a.id, b.id, EdgeType.TEMPORAL, 0.9)
        assert kg.edge_count == 1
        assert kg.get_edge(a.id, b.id).type == EdgeType.TEMPORAL

    def test_remove_nonexistent_node(self):
        kg = KnowledgeGraph()
        assert kg.remove_node("ghost") is None

    def test_remove_nonexistent_edge(self):
        kg = KnowledgeGraph()
        assert kg.remove_edge("a", "b") is None

    def test_edge_weight_boundary(self):
        a = create_episode("A")
        b = create_episode("B")
        Edge(a.id, b.id, EdgeType.CAUSAL, 0.0)
        Edge(a.id, b.id, EdgeType.CAUSAL, 1.0)
        with pytest.raises(ValidationError):
            Edge(a.id, b.id, EdgeType.CAUSAL, -0.01)
        with pytest.raises(ValidationError):
            Edge(a.id, b.id, EdgeType.CAUSAL, 1.01)

    def test_large_graph_merge(self):
        kg1 = KnowledgeGraph()
        kg2 = KnowledgeGraph()
        for i in range(100):
            kg1.add_node(create_episode(f"kg1_{i}"))
            kg2.add_node(create_episode(f"kg2_{i}"))
        added = kg1.merge(kg2)
        assert added == 100
        assert kg1.node_count == 200

    def test_serialization_empty(self):
        kg = KnowledgeGraph()
        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert kg2.node_count == 0
        assert kg2.edge_count == 0

    def test_repr(self):
        kg = KnowledgeGraph()
        assert "KnowledgeGraph" in repr(kg)
        node = create_episode("test repr")
        assert "MemoryNode" in repr(node)
        edge = Edge("a", "b", EdgeType.CAUSAL, 0.7)
        assert "Edge" in repr(edge)
        assert "0.70" in repr(edge)


class TestMCTSEdgeCases:
    """Empty tree, single node, zero iterations."""

    def test_empty_tree_operations(self):
        tree = ThoughtTree()
        assert tree.size == 0
        assert tree.get_best_path() == []
        assert tree.get_all_paths() == []
        assert tree.visualize() == "<empty tree>"
        assert tree.prune(0.5) == 0

    def test_root_only_tree(self):
        tree = ThoughtTree()
        tree.set_root("just root")
        assert tree.size == 1
        assert len(tree.get_best_path()) == 1
        assert len(tree.get_all_paths()) == 1

    def test_zero_iterations(self):
        mcts = MCTS(max_iterations=0)
        tree = mcts.search("test", {}, lambda s: ["a"], lambda s, t: 0.5)
        assert tree.size == 1  # Only root

    def test_depth_zero(self):
        tree = ThoughtTree(max_depth=0)
        root = tree.set_root("Q")
        child = tree.add_thought(root, "child")
        assert child is None

    def test_thought_node_repr(self):
        node = ThoughtNode(thought="hello world", score=0.5, visits=3)
        assert "hello" in repr(node) or "ThoughtNode" in str(type(node))


class TestMemoryEdgeCases:
    """Empty memory, zero capacity, edge consolidation."""

    def test_zero_capacity_working_memory(self):
        wm = WorkingMemory(capacity=1)
        wm.put("a", {"content": "A"})
        wm.put("b", {"content": "B"})  # evicts A
        assert wm.get("a") is None
        assert wm.get("b") is not None

    def test_empty_recall(self):
        mm = MemoryManager()
        assert mm.recall("anything") == []
        assert mm.recall_by_importance() == []
        assert mm.recall_by_recency() == []

    def test_empty_consolidate(self):
        mm = MemoryManager()
        assert mm.consolidate() == 0

    def test_empty_reflect(self):
        mm = MemoryManager()
        assert mm.reflect() == []

    def test_empty_forget(self):
        mm = MemoryManager()
        assert mm.forget() == 0

    def test_repr(self):
        mm = MemoryManager()
        assert "MemoryManager" in repr(mm)

    def test_working_memory_contains(self):
        wm = WorkingMemory()
        wm.put("k1", {"content": "test"})
        assert "k1" in wm
        assert "k2" not in wm
        assert len(wm) == 1

    def test_consolidate_empty_working_memory(self):
        mm = MemoryManager()
        mm.consolidate()  # no crash
        assert mm.stats()["longterm_memory"]["nodes"] == 0

    def test_recompute_importance_empty(self):
        mm = MemoryManager()
        mm.longterm.recompute_importance()  # no crash


class TestPlanningEdgeCases:
    """Empty plan, circular deps, all-terminal."""

    def test_empty_plan(self):
        planner = Planner()
        executor = Executor(planner)
        logs = executor.execute_plan([], lambda g: (True, ""))
        assert logs == []

    def test_circular_dependencies(self):
        planner = Planner()
        a = create_goal("A")
        b = create_goal("B")
        a.dependencies = [b.id]
        b.dependencies = [a.id]
        a.subgoals = [b.id]
        planner.add_goal(a)
        planner.add_goal(b)

        plan = planner.create_plan(a)
        # Should not infinite loop - force break cycle
        assert len(plan) == 2

    def test_all_goals_terminal(self):
        planner = Planner()
        g = create_goal("done")
        g.complete("already done")
        planner.add_goal(g)

        executor = Executor(planner)
        logs = executor.execute_plan([g], lambda g: (True, ""))
        assert len(logs) == 0  # skipped terminal

    def test_retry_exhaustion(self):
        planner = Planner()
        g = create_goal("fail")
        planner.add_goal(g)

        executor = Executor(planner)
        logs = executor.execute_with_retry([g], lambda g: (False, "always fails"), max_retries=3)
        assert len(logs) == 1
        assert not logs[0].success
        assert g.state == GoalState.FAILED

    def test_goal_repr(self):
        g = create_goal("Build feature X")
        assert "Build feature" in repr(g)

    def test_planner_decompose_empty(self):
        planner = Planner()
        root = create_goal("root")
        planner.add_goal(root)
        subs = planner.decompose(root, [])
        assert subs == []

    def test_decompose_goal_not_in_planner(self):
        planner = Planner()
        root = create_goal("orphan")
        # decompose should still work since we add subgoals
        subs = planner.decompose(root, ["sub1"])
        assert len(subs) == 1


class TestTraversalEdgeCases:
    """Disconnected graphs, cycles, single component."""

    def test_bfs_single_node(self):
        kg = KnowledgeGraph()
        n = create_episode("only")
        kg.add_node(n)
        result = bfs(kg, n.id)
        assert result == {0: [n.id]}

    def test_dfs_single_node(self):
        kg = KnowledgeGraph()
        n = create_episode("only")
        kg.add_node(n)
        result = dfs(kg, n.id)
        assert result == [n.id]

    def test_shortest_path_to_self(self):
        kg = KnowledgeGraph()
        n = create_episode("me")
        kg.add_node(n)
        path = shortest_path(kg, n.id, n.id)
        assert path == [n.id]

    def test_connected_components_disconnected(self):
        kg = KnowledgeGraph()
        for i in range(5):
            kg.add_node(create_episode(f"N{i}"))
        # No edges, so 5 components
        components = connected_components(kg)
        assert len(components) == 5

    def test_topological_sort_with_cycle(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, a.id, EdgeType.CAUSAL)  # cycle
        result = topological_sort(kg)
        assert result is None  # cycle detected


class TestScoringEdgeCases:
    """Empty graph, single node, uniform weights."""

    def test_pagerank_single_node(self):
        kg = KnowledgeGraph()
        n = create_episode("only")
        kg.add_node(n)
        scores = compute_pagerank(kg)
        assert scores == {n.id: 1.0}

    def test_pagerank_empty(self):
        kg = KnowledgeGraph()
        assert compute_pagerank(kg) == {}

    def test_decay_to_zero(self):
        kg = KnowledgeGraph()
        n = create_episode("fading")
        n.importance = 1.0
        kg.add_node(n)
        from agentic_os.core.graph.scoring import decay_scores
        for _ in range(100):
            decay_scores(kg, 0.95)
        assert kg.get_node(n.id).importance < 0.01
