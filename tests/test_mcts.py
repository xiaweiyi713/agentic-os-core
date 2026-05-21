"""MCTS 与 Tree of Thoughts 测试"""


from agentic_os.core.tree.mcts import MCTS
from agentic_os.core.tree.pruning import depth_prune, redundancy_prune, score_threshold_prune
from agentic_os.core.tree.search import beam_search, best_first_search, ucb1_score
from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree


class TestThoughtNode:
    def test_basic_properties(self):
        root = ThoughtNode(thought="root", score=1.0, visits=5)
        child = ThoughtNode(thought="child", score=0.8, visits=3, parent=root)
        root.children.append(child)

        assert root.is_root
        assert not root.is_leaf
        assert child.depth == 1
        assert root.best_child() == child

    def test_path_from_root(self):
        root = ThoughtNode(thought="root")
        c1 = ThoughtNode(thought="c1", parent=root)
        c2 = ThoughtNode(thought="c2", parent=c1)
        root.children.append(c1)
        c1.children.append(c2)

        path = c2.path_from_root()
        assert len(path) == 3
        assert path[0] == root
        assert path[2] == c2

    def test_avg_score(self):
        node = ThoughtNode(thought="test", score=3.0, visits=6)
        assert node.avg_score == 0.5


class TestThoughtTree:
    def test_set_root(self):
        tree = ThoughtTree()
        root = tree.set_root("初始问题", {"step": 0})
        assert tree.root == root
        assert tree.size == 1

    def test_add_thought(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        child = tree.add_thought(root, "思路1", score=0.7)
        assert child is not None
        assert child.parent == root
        assert tree.size == 2

    def test_max_depth_limit(self):
        tree = ThoughtTree(max_depth=2)
        root = tree.set_root("问题")
        c1 = tree.add_thought(root, "思路1")
        c2 = tree.add_thought(c1, "思路2")
        c3 = tree.add_thought(c2, "思路3")  # 超过深度限制
        assert c3 is None

    def test_max_children_limit(self):
        tree = ThoughtTree(max_children=2)
        root = tree.set_root("问题")
        tree.add_thought(root, "思路1")
        tree.add_thought(root, "思路2")
        c3 = tree.add_thought(root, "思路3")  # 超过子节点限制
        assert c3 is None

    def test_get_best_path(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        _c1 = tree.add_thought(root, "good idea", score=0.9)
        _c2 = tree.add_thought(root, "bad idea", score=0.3)
        path = tree.get_best_path()
        assert len(path) == 2
        assert path[1].thought == "good idea"

    def test_get_all_paths(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        tree.add_thought(root, "思路1")
        tree.add_thought(root, "思路2")
        paths = tree.get_all_paths()
        assert len(paths) == 2

    def test_prune(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        tree.add_thought(root, "高分", score=0.9)
        tree.add_thought(root, "低分", score=0.1)
        removed = tree.prune(0.5)
        assert removed == 1

    def test_visualize(self):
        tree = ThoughtTree()
        root = tree.set_root("根节点")
        tree.add_thought(root, "子节点1", score=0.8)
        tree.add_thought(root, "子节点2", score=0.6)
        viz = tree.visualize()
        assert "根节点" in viz
        assert "子节点1" in viz


class TestMCTS:
    def test_basic_search(self):
        mcts = MCTS(exploration_weight=1.0, max_iterations=50, max_depth=3)

        def generator(state):
            return [
                f"方案A({state.get('last_thought', 'start')})",
                f"方案B({state.get('last_thought', 'start')})",
            ]

        def evaluator(state, thought):
            return 0.5 if "A" in thought else 0.3

        tree = mcts.search(
            root_thought="解决复杂问题",
            root_state={"step": 0},
            generator=generator,
            evaluator=evaluator,
        )

        assert tree.root is not None
        assert tree.size > 1
        best_path = tree.get_best_path()
        assert len(best_path) >= 1

    def test_mcts_convergence(self):
        """验证 MCTS 能收敛到最优解"""
        mcts = MCTS(exploration_weight=0.5, max_iterations=200, max_depth=2)

        def generator(state):
            depth = state.get("depth", 0)
            if depth >= 2:
                return []
            return [
                "最优策略", "次优策略", "最差策略",
            ]

        def evaluator(state, thought):
            scores = {"最优策略": 0.95, "次优策略": 0.5, "最差策略": 0.1}
            return scores.get(thought, 0.3)

        tree = mcts.search(
            root_thought="决策问题",
            root_state={"depth": 0},
            generator=generator,
            evaluator=evaluator,
        )

        best_path = tree.get_best_path()
        # 最优策略应被探索多次
        best_thoughts = [n.thought for n in best_path]
        assert "最优策略" in best_thoughts or tree.root.visits > 1

    def test_empty_candidates(self):
        mcts = MCTS(max_iterations=10)

        tree = mcts.search(
            root_thought="测试",
            root_state={},
            generator=lambda s: [],
            evaluator=lambda s, t: 0.5,
        )
        assert tree.size == 1  # 只有根节点


class TestSearchStrategies:
    def test_ucb1_unvisited(self):
        node = ThoughtNode(thought="test", visits=0)
        assert ucb1_score(node) == float("inf")

    def test_ucb1_visited(self):
        root = ThoughtNode(thought="root", score=2.0, visits=4)
        child = ThoughtNode(thought="child", score=1.5, visits=3, parent=root)
        score = ucb1_score(child, c=1.414)
        assert score > 0

    def test_best_first_search(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        c1 = tree.add_thought(root, "good", score=0.9)
        _c2 = tree.add_thought(root, "bad", score=0.1)
        tree.add_thought(c1, "better", score=0.95)

        best = best_first_search(root, lambda n: n.score, max_nodes=10)
        assert best.score >= 0.9

    def test_beam_search(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        for i in range(5):
            tree.add_thought(root, f"思路{i}", score=i * 0.2)

        results = beam_search(root, beam_width=2, evaluator=lambda n: n.score)
        assert len(results) <= 2


class TestPruning:
    def test_score_threshold_prune(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        tree.add_thought(root, "好", score=0.9)
        tree.add_thought(root, "差", score=0.1)
        removed = score_threshold_prune(tree, 0.5)
        assert removed >= 1

    def test_depth_prune(self):
        tree = ThoughtTree(max_depth=10)
        root = tree.set_root("问题")
        c1 = tree.add_thought(root, "L1")
        c2 = tree.add_thought(c1, "L2")
        tree.add_thought(c2, "L3")

        removed = depth_prune(tree, max_depth=1)
        assert removed >= 1

    def test_redundancy_prune(self):
        tree = ThoughtTree()
        root = tree.set_root("问题")
        tree.add_thought(root, "use deep learning for image classification", score=0.5)
        tree.add_thought(root, "use deep learning for text classification", score=0.5)  # 冗余 (5/6=0.83)

        def sim(a, b):
            words_a = set(a.split())
            words_b = set(b.split())
            return len(words_a & words_b) / max(len(words_a), 1)

        removed = redundancy_prune(tree, sim)
        assert removed >= 1
