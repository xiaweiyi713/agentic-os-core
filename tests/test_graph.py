"""知识图谱模块测试"""

import pytest

from agentic_os.core.graph.edge import Edge, EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import (
    NodeType,
    create_episode,
    create_fact,
    create_goal,
    create_reflection,
)
from agentic_os.core.graph.scoring import (
    association_strength,
    compute_pagerank,
    decay_scores,
    get_most_important,
    recompute_importance,
)
from agentic_os.core.graph.traversal import (
    bfs,
    connected_components,
    dfs,
    shortest_path,
    topological_sort,
    traverse_by_type,
)
from agentic_os.exceptions import ValidationError


class TestNode:
    def test_create_episode(self):
        node = create_episode("用户询问了天气")
        assert node.type == NodeType.EPISODE
        assert "天气" in node.content
        assert node.id.startswith("ep_")

    def test_create_fact(self):
        node = create_fact("Python 是解释型语言", importance=0.8)
        assert node.type == NodeType.FACT
        assert node.importance == 0.8

    def test_create_reflection(self):
        node = create_reflection("用户偏好简洁回答")
        assert node.type == NodeType.REFLECTION

    def test_create_goal(self):
        node = create_goal("完成报告")
        assert node.type == NodeType.GOAL

    def test_touch_updates_access_count(self):
        node = create_episode("测试")
        assert node.access_count == 0
        node.touch()
        assert node.access_count == 1


class TestEdge:
    def test_edge_creation(self):
        edge = Edge("a", "b", EdgeType.CAUSAL, 0.8)
        assert edge.source_id == "a"
        assert edge.weight == 0.8

    def test_edge_weight_validation(self):
        with pytest.raises(ValidationError):
            Edge("a", "b", EdgeType.CAUSAL, 1.5)

    def test_edge_key(self):
        edge = Edge("a", "b", EdgeType.TEMPORAL)
        assert edge.key == ("a", "b")


class TestKnowledgeGraph:
    def test_add_and_get_node(self):
        kg = KnowledgeGraph()
        node = create_episode("测试内容")
        kg.add_node(node)
        assert kg.has_node(node.id)
        assert kg.get_node(node.id) is not None
        assert kg.node_count == 1

    def test_remove_node(self):
        kg = KnowledgeGraph()
        node = create_episode("要删除的节点")
        kg.add_node(node)
        removed = kg.remove_node(node.id)
        assert removed is not None
        assert not kg.has_node(node.id)
        assert kg.node_count == 0

    def test_add_and_remove_edge(self):
        kg = KnowledgeGraph()
        a = create_episode("节点A")
        b = create_episode("节点B")
        kg.add_node(a)
        kg.add_node(b)
        edge = kg.add_edge(a.id, b.id, EdgeType.CAUSAL, 0.9)
        assert edge is not None
        assert kg.has_edge(a.id, b.id)
        assert kg.edge_count == 1

        removed = kg.remove_edge(a.id, b.id)
        assert removed is not None
        assert not kg.has_edge(a.id, b.id)

    def test_edge_to_nonexistent_node(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        kg.add_node(a)
        result = kg.add_edge(a.id, "nonexistent", EdgeType.CAUSAL)
        assert result is None

    def test_remove_node_cleans_edges(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.remove_node(b.id)
        assert not kg.has_edge(a.id, b.id)

    def test_neighbors(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        c = create_episode("C")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(a.id, c.id, EdgeType.TEMPORAL)
        assert set(kg.out_neighbors(a.id)) == {b.id, c.id}
        assert a.id in kg.in_neighbors(b.id)

    def test_find_nodes_by_keyword(self):
        kg = KnowledgeGraph()
        kg.add_node(create_episode("今天天气晴朗"))
        kg.add_node(create_episode("明天会下雨"))
        kg.add_node(create_episode("Python编程"))
        results = kg.find_nodes("天气")
        assert len(results) == 1
        assert "天气" in results[0].content

    def test_find_nodes_by_type(self):
        kg = KnowledgeGraph()
        kg.add_node(create_episode("经历"))
        kg.add_node(create_fact("事实"))
        episodes = kg.get_nodes_by_type(NodeType.EPISODE)
        assert len(episodes) >= 1

    def test_subgraph(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"节点{i}") for i in range(5)]
        for n in nodes:
            kg.add_node(n)
        for i in range(4):
            kg.add_edge(nodes[i].id, nodes[i + 1].id, EdgeType.TEMPORAL)

        sub = kg.subgraph(nodes[0].id, depth=2)
        assert sub.node_count == 3  # node0, node1, node2

    def test_merge(self):
        kg1 = KnowledgeGraph()
        kg2 = KnowledgeGraph()
        kg1.add_node(create_episode("A"))
        kg2.add_node(create_episode("B"))
        added = kg1.merge(kg2)
        assert added == 1
        assert kg1.node_count == 2

    def test_serialization(self):
        kg = KnowledgeGraph()
        a = create_episode("序列化测试")
        b = create_fact("序列化事实")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.DERIVED_FROM, 0.7)

        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert kg2.node_count == 2
        assert kg2.edge_count == 1

    def test_stats(self):
        kg = KnowledgeGraph()
        kg.add_node(create_episode("E"))
        kg.add_node(create_fact("F"))
        stats = kg.stats()
        assert stats["nodes"] == 2
        assert "episode" in stats["types"]


class TestTraversal:
    def test_bfs(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[0].id, nodes[2].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[3].id, EdgeType.CAUSAL)

        result = bfs(kg, nodes[0].id)
        assert result[0] == [nodes[0].id]
        assert set(result[1]) == {nodes[1].id, nodes[2].id}

    def test_dfs(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(3)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL)

        result = dfs(kg, nodes[0].id)
        assert result[0] == nodes[0].id

    def test_shortest_path(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL, 0.8)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL, 0.9)
        kg.add_edge(nodes[0].id, nodes[2].id, EdgeType.CAUSAL, 0.3)

        path = shortest_path(kg, nodes[0].id, nodes[2].id)
        assert path is not None
        # Intermediate path (0.2+0.1=0.3) < direct (0.7), so path length is 3
        assert len(path) == 3
        assert path[0] == nodes[0].id and path[-1] == nodes[2].id

    def test_shortest_path_no_path(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        assert shortest_path(kg, a.id, b.id) is None

    def test_traverse_by_type(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(3)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.TEMPORAL)

        result = traverse_by_type(kg, nodes[0].id, EdgeType.CAUSAL)
        assert len(result) == 2

    def test_topological_sort(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"Step{i}") for i in range(3)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL)

        result = topological_sort(kg)
        assert result is not None
        assert result.index(nodes[0].id) < result.index(nodes[2].id)

    def test_connected_components(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        c = create_episode("C")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)
        kg.add_edge(a.id, b.id, EdgeType.ASSOCIATIVE)

        components = connected_components(kg)
        assert len(components) == 2


class TestScoring:
    def test_pagerank(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[2].id, nodes[3].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[3].id, nodes[0].id, EdgeType.CAUSAL)

        scores = compute_pagerank(kg)
        assert len(scores) == 4
        assert all(0 <= s <= 1 for s in scores.values())

    def test_decay_scores(self):
        kg = KnowledgeGraph()
        node = create_episode("test")
        node.importance = 1.0
        kg.add_node(node)
        decay_scores(kg, 0.9)
        assert abs(kg.get_node(node.id).importance - 0.9) < 0.01

    def test_get_most_important(self):
        kg = KnowledgeGraph()
        for i in range(5):
            n = create_episode(f"N{i}")
            n.importance = i * 0.2
            kg.add_node(n)

        top = get_most_important(kg, 3)
        assert len(top) == 3
        assert top[0][1] >= top[1][1]

    def test_association_strength(self):
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.ASSOCIATIVE, 0.7)
        strength = association_strength(kg, a.id, b.id)
        assert strength == 0.7

    def test_recompute_importance(self):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(3)]
        for n in nodes:
            n.importance = 0.1
            kg.add_node(n)
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL)
        recompute_importance(kg)
        # 所有节点重要性应被更新
        for n in kg.nodes:
            assert n.importance != 0.1
