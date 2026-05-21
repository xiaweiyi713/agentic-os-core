"""补充测试覆盖率 -- 覆盖 mcts.py、traversal.py、hashing.py 的缺失行。"""

from agentic_os.core.graph.edge import Edge, EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import create_episode
from agentic_os.core.graph.traversal import (
    connected_components,
    dfs,
    shortest_path,
    traverse_by_type,
)
from agentic_os.core.tree.mcts import MCTS
from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree
from agentic_os.utils.hashing import combined_id


# ---------------------------------------------------------------------------
# mcts.py 缺失行: 69, 122, 147, 174, 223, 225
# ---------------------------------------------------------------------------


class TestMCTSRepr:
    """行 69: MCTS.__repr__"""

    def test_repr_format(self):
        mcts = MCTS(exploration_weight=2.0, max_iterations=500, max_depth=8)
        r = repr(mcts)
        assert "MCTS(c=2.00" in r
        assert "iters=500" in r
        assert "depth=8" in r


class TestMCTSExpandAllFail:
    """行 122, 174: _expand 返回 None（所有候选都因约束添加失败）"""

    def test_expand_returns_none_when_all_candidates_rejected(self):
        mcts = MCTS(exploration_weight=1.0, max_iterations=1, max_depth=10)

        # generator 返回候选，但 tree 的 max_children=0 导致所有候选添加失败
        tree = ThoughtTree(max_depth=10, max_children=0)
        root = tree.set_root("root", {"step": 0})
        root.visits = 1

        # 手动调用 _expand，tree 的 max_children=0 使 add_thought 始终返回 None
        result = mcts._expand(tree, root, ["c1", "c2"], lambda s, t: 0.5)
        assert result is None

    def test_search_continues_when_expand_returns_none(self):
        """行 122: search 中 expanded is None 时 continue，树保持只有根节点。"""
        mcts = MCTS(exploration_weight=1.0, max_iterations=5, max_depth=10)

        # 使用 max_children=0 的 tree 内部：但 search 方法会创建自己的 tree
        # 所以我们利用 generator 返回已被添加过的相同 thought 来触发 add_thought 返回 None
        # 实际上更直接的方式：使用 max_children=0 无法实现，因为 search 硬编码 max_children=20
        # 需要用一个巧妙方式：让所有候选都因为 max_depth 限制而失败
        # 用 max_depth=0，root 深度 0 >= max_depth 0 -> add_thought 返回 None
        mcts2 = MCTS(exploration_weight=1.0, max_iterations=3, max_depth=0)
        tree = mcts2.search(
            root_thought="root",
            root_state={},
            generator=lambda s: ["a", "b"],
            evaluator=lambda s, t: 0.5,
        )
        # 由于 max_depth=0，所有 expand 都失败，树只有根节点
        assert tree.size == 1


class TestMCTSSelectUnvisitedChild:
    """行 147: _select 中返回未访问的子节点（visits == 0）"""

    def test_select_returns_unvisited_child(self):
        mcts = MCTS()

        root = ThoughtNode(thought="root", visits=5)
        child_a = ThoughtNode(thought="a", visits=3, parent=root)
        child_b = ThoughtNode(thought="b", visits=0, parent=root)  # 未访问
        root.children = [child_a, child_b]

        selected = mcts._select(root)
        assert selected is child_b


class TestMCTSUCB1EdgeCases:
    """行 223, 225: _ucb1 的边界情况"""

    def test_ucb1_unvisited_returns_inf(self):
        """行 222-223: node.visits == 0 返回 inf"""
        mcts = MCTS()
        node = ThoughtNode(thought="x", visits=0)
        assert mcts._ucb1(node) == float("inf")

    def test_ucb1_parent_none_returns_avg_score(self):
        """行 224-225: node.parent is None 返回 avg_score"""
        mcts = MCTS()
        node = ThoughtNode(thought="x", score=2.5, visits=5)
        # parent is None, visits > 0
        assert mcts._ucb1(node) == node.avg_score

    def test_ucb1_parent_zero_visits_returns_avg_score(self):
        """行 224-225: node.parent.visits == 0 返回 avg_score"""
        mcts = MCTS()
        root = ThoughtNode(thought="root", visits=0)
        child = ThoughtNode(thought="child", score=3.0, visits=2, parent=root)
        root.children.append(child)
        assert mcts._ucb1(child) == child.avg_score


# ---------------------------------------------------------------------------
# traversal.py 缺失行: 76, 78, 118, 130, 162, 164, 245, 252
# ---------------------------------------------------------------------------


class TestDFSVisitedSkipAndMaxDepth:
    """行 76, 78: dfs 中 _visit 的已访问跳过 + max_depth 限制"""

    def test_dfs_skips_already_visited_nodes(self):
        """行 76: 构建一个有环的图（通过双向边），验证已访问节点被跳过。"""
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, a.id, EdgeType.CAUSAL)  # 反向边形成环

        result = dfs(kg, a.id)
        # 不应无限循环，且每个节点只出现一次
        assert len(result) == 2
        assert result[0] == a.id

    def test_dfs_respects_max_depth(self):
        """行 78: max_depth 限制。"""
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        # 0 -> 1 -> 2 -> 3
        for i in range(3):
            kg.add_edge(nodes[i].id, nodes[i + 1].id, EdgeType.CAUSAL)

        result = dfs(kg, nodes[0].id, max_depth=1)
        # 深度 0: nodes[0], 深度 1: nodes[1], 深度 2 > max_depth=1 被跳过
        assert nodes[0].id in result
        assert nodes[1].id in result
        assert nodes[2].id not in result
        assert nodes[3].id not in result


class TestShortestPathVisitedSkips:
    """行 118, 130: shortest_path 中已访问节点的跳过"""

    def test_shortest_path_skips_visited_nodes_in_heap(self):
        """行 118: u in visited 时 continue。

        关键：end_id 不能是 b，否则 Dijkstra 找到 b 时提前返回，不会弹出 b 的第二个堆条目。
        所以 end_id 设为 d，让 b 的第二个堆条目有机会被弹出。

        a -> c (cost=0.1), c -> b (cost=0.1) => b 便宜路径 cost=0.2
        a -> b (cost=0.9) => b 贵路径
        b -> d (cost=0.1)

        堆: pop c(0.1), push b(0.2); pop b(0.2), visit b, push d(0.3)
        找到 d, 返回. 但 b(0.9) 还在堆中。

        为了让 b(0.9) 被弹出，需要不让 d 成为终点，而让 d 之后还有节点。
        改为: end=e, d->e, 且不经过 b 的路径到 e。
        """
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(5)]
        for n in nodes:
            kg.add_node(n)
        a, b, c, d, e = nodes

        kg.add_edge(a.id, c.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL, 0.1)   # cost=0.9 (b 首次入堆)
        kg.add_edge(c.id, b.id, EdgeType.CAUSAL, 0.9)   # cost=0.1 (b 更新到 0.2)
        kg.add_edge(b.id, d.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(d.id, e.id, EdgeType.CAUSAL, 0.9)   # cost=0.1

        path = shortest_path(kg, a.id, e.id)
        assert path is not None
        assert path[0] == a.id
        assert path[-1] == e.id

    def test_shortest_path_skips_visited_edge_targets(self):
        """行 130: edge.target_id in visited 时 continue。

        构造星形图：中心 a 连向 b,c,d。d 也连向 b,c。
        弹出 a(0), push b(0.1), c(0.1), d(0.1)
        弹出 b(0.1), visit b
        弹出 c(0.1), visit c
        弹出 d(0.1), visit d, 此时 d 的出边 b,c 已 visited => 行 130
        然后弹出 d 的出边到不存在的终点继续遍历。
        为触发行 130，终点设为 e（不在图中不可达），使 Dijkstra 遍历所有节点。
        """
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(5)]
        for n in nodes:
            kg.add_node(n)
        a, b, c, d, e = nodes

        kg.add_edge(a.id, b.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(a.id, c.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(a.id, d.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(d.id, b.id, EdgeType.CAUSAL, 0.9)   # b 可能已 visited
        kg.add_edge(d.id, c.id, EdgeType.CAUSAL, 0.9)   # c 可能已 visited
        # e 是孤立节点

        path = shortest_path(kg, a.id, e.id)
        assert path is None  # e 不可达

    def test_shortest_path_all_nodes_visited_duplicate_heap_entries(self):
        """行 118: 通过搜索不可达终点使 Dijkstra 遍历所有节点。

        a -> c (cost=0.1), c -> b (cost=0.1) => b 便宜路径 cost=0.2
        a -> b (cost=0.9) => b 贵路径, b(0.9) 入堆
        b -> d (cost=0.1)

        搜索 a -> e（孤立），迫使遍历完所有节点后弹出 b(0.9)，
        此时 b 已 visited，触发行 118。
        """
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(5)]
        for n in nodes:
            kg.add_node(n)
        a, b, c, d, e = nodes

        kg.add_edge(a.id, c.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL, 0.1)   # cost=0.9 (b 首次入堆)
        kg.add_edge(c.id, b.id, EdgeType.CAUSAL, 0.9)   # cost=0.1 (b 更新到 0.2)
        kg.add_edge(b.id, d.id, EdgeType.CAUSAL, 0.9)   # cost=0.1
        # e 是孤立节点

        path = shortest_path(kg, a.id, e.id)
        assert path is None


class TestTraverseByTypeVisitedAndDepth:
    """行 162, 164: traverse_by_type 中已访问跳过 + max_depth 限制"""

    def test_traverse_by_type_skips_visited_with_cycle(self):
        """行 162: node_id in visited 时 return。

        构造更复杂的三节点 CAUSAL 环：a -> b -> c -> a。
        _visit(c) 时发现 a 已 visited，触发行 162。
        """
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(3)]
        for n in nodes:
            kg.add_node(n)
        a, b, c = nodes

        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, c.id, EdgeType.CAUSAL)
        kg.add_edge(c.id, a.id, EdgeType.CAUSAL)  # 回到 a

        result = traverse_by_type(kg, a.id, EdgeType.CAUSAL)
        assert len(result) == 3
        assert a.id in result
        assert b.id in result
        assert c.id in result

    def test_traverse_by_type_only_follows_matching_edges(self):
        """行 162 补充：确保非匹配边类型的节点不会被访问但不会导致 visited 跳过。

        构造：a -[CAUSAL]-> b -[TEMPORAL]-> c -[CAUSAL]-> d
        从 a 出发只跟随 CAUSAL 边，b 被访问后，b 的 TEMPORAL 边指向 c 不被跟随。
        """
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        a, b, c, d = nodes

        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, c.id, EdgeType.TEMPORAL)   # 不匹配，不跟随
        kg.add_edge(c.id, d.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, d.id, EdgeType.CAUSAL)      # 匹配，跟随到 d

        result = traverse_by_type(kg, a.id, EdgeType.CAUSAL)
        assert a.id in result
        assert b.id in result
        assert d.id in result
        assert c.id not in result   # c 只被 TEMPORAL 边连接，不应出现

    def test_traverse_by_type_max_depth(self):
        """行 164: max_depth 限制。"""
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(4)]
        for n in nodes:
            kg.add_node(n)
        for i in range(3):
            kg.add_edge(nodes[i].id, nodes[i + 1].id, EdgeType.CAUSAL)

        result = traverse_by_type(kg, nodes[0].id, EdgeType.CAUSAL, max_depth=1)
        assert nodes[0].id in result
        assert nodes[1].id in result
        assert nodes[2].id not in result


class TestConnectedComponentsBFSEdgeCases:
    """行 245, 252: connected_components 中 _bfs 的边界"""

    def test_connected_components_skips_nodes_already_in_component(self):
        """行 245: nid in comp 时 continue。

        构建双向边图，使邻居可能被重复加入队列。
        """
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        kg.add_edge(a.id, b.id, EdgeType.CAUSAL)
        kg.add_edge(b.id, a.id, EdgeType.CAUSAL)  # 反向边

        components = connected_components(kg)
        assert len(components) == 1
        assert components[0] == {a.id, b.id}

    def test_connected_components_adds_in_neighbors(self):
        """行 252: neighbor not in comp 时加入队列（入邻居方向）。

        构建只有 b->a 的边（无 a->b），验证 a 仍然通过 in_neighbors 发现 b。
        """
        kg = KnowledgeGraph()
        a = create_episode("A")
        b = create_episode("B")
        kg.add_node(a)
        kg.add_node(b)
        # 只有 b -> a 的边
        kg.add_edge(b.id, a.id, EdgeType.CAUSAL)

        # 从 a 出发，a 的 out_neighbors 为空，但 in_neighbors 包含 b
        components = connected_components(kg)
        assert len(components) == 1
        assert components[0] == {a.id, b.id}

    def test_connected_components_three_nodes_with_back_edges(self):
        """构建含双向边的三节点环，测试 _bfs 中 nid in comp 跳过。"""
        kg = KnowledgeGraph()
        nodes = [create_episode(f"N{i}") for i in range(3)]
        for n in nodes:
            kg.add_node(n)
        # 环: 0 -> 1 -> 2 -> 0
        kg.add_edge(nodes[0].id, nodes[1].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[1].id, nodes[2].id, EdgeType.CAUSAL)
        kg.add_edge(nodes[2].id, nodes[0].id, EdgeType.CAUSAL)

        components = connected_components(kg)
        assert len(components) == 1
        assert components[0] == {nodes[0].id, nodes[1].id, nodes[2].id}


# ---------------------------------------------------------------------------
# hashing.py 缺失行: 64
# ---------------------------------------------------------------------------


class TestHashingCombinedId:
    """行 64: combined_id 函数"""

    def test_combined_id_basic(self):
        result = combined_id("user", "session", "42")
        assert isinstance(result, str)
        assert len(result) == 16  # 16-char hex string

    def test_combined_id_deterministic(self):
        r1 = combined_id("a", "b")
        r2 = combined_id("a", "b")
        assert r1 == r2

    def test_combined_id_different_parts(self):
        r1 = combined_id("a", "b")
        r2 = combined_id("b", "a")
        assert r1 != r2

    def test_combined_id_uses_pipe_separator(self):
        from agentic_os.utils.hashing import content_id

        assert combined_id("x", "y") == content_id("x|y")
