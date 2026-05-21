"""AsyncMCTS 测试"""

import asyncio
from unittest.mock import AsyncMock

from agentic_os.core.tree.async_mcts import AsyncMCTS
from agentic_os.core.tree.mcts import MCTS


class TestAsyncMCTSBasic:
    def test_basic_search(self):
        """基本搜索 -- async eval 和 generator，验证树构建"""
        mcts = AsyncMCTS(exploration_weight=1.0, max_iterations=50, max_depth=3)

        async def generator(state: dict) -> list[str]:
            return [
                f"方案A({state.get('last_thought', 'start')})",
                f"方案B({state.get('last_thought', 'start')})",
            ]

        async def evaluator(state: dict, thought: str) -> float:
            return 0.5 if "A" in thought else 0.3

        tree = asyncio.run(mcts.search(
            root_thought="解决复杂问题",
            root_state={"step": 0},
            generator=generator,
            evaluator=evaluator,
        ))

        assert tree.root is not None
        assert tree.size > 1
        best_path = tree.get_best_path()
        assert len(best_path) >= 1

    def test_multiple_children_added(self):
        """并发验证 -- 搜索完成后有多个子节点被添加"""
        mcts = AsyncMCTS(
            exploration_weight=2.0, max_iterations=200, max_depth=5,
            concurrency=4,
        )

        async def generator(state: dict) -> list[str]:
            depth = state.get("depth", 0)
            if depth >= 3:
                return []
            return ["思路1", "思路2", "思路3"]

        async def evaluator(state: dict, thought: str) -> float:
            scores = {"思路1": 0.9, "思路2": 0.6, "思路3": 0.3}
            return scores.get(thought, 0.1)

        tree = asyncio.run(mcts.search(
            root_thought="测试",
            root_state={"depth": 0},
            generator=generator,
            evaluator=evaluator,
        ))

        assert tree.root is not None
        # 高探索权重下，多次迭代后树应有多个节点
        assert tree.size > 3

    def test_max_depth_limit(self):
        """max_depth 限制"""
        mcts = AsyncMCTS(max_iterations=50, max_depth=2)

        async def generator(state: dict) -> list[str]:
            return ["更深层"]

        async def evaluator(state: dict, thought: str) -> float:
            return 0.5

        tree = asyncio.run(mcts.search(
            root_thought="根",
            root_state={},
            generator=generator,
            evaluator=evaluator,
        ))

        # 所有节点深度不超过 max_depth
        def _check_depth(node, max_d):
            assert node.depth <= max_d, f"node depth {node.depth} > {max_d}"
            for child in node.children:
                _check_depth(child, max_d)

        _check_depth(tree.root, 2)

    def test_empty_candidates(self):
        """空候选处理"""
        mcts = AsyncMCTS(max_iterations=10)

        tree = asyncio.run(mcts.search(
            root_thought="测试",
            root_state={},
            generator=AsyncMock(return_value=[]),
            evaluator=AsyncMock(return_value=0.5),
        ))

        assert tree.size == 1  # 只有根节点

    def test_same_root_as_sync(self):
        """与同步 MCTS 对比 -- 相同输入，async 和 sync 结果应该有相同的根节点"""
        exploration_weight = 1.0
        max_iterations = 30
        max_depth = 3

        # 同步版本
        sync_mcts = MCTS(
            exploration_weight=exploration_weight,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )

        def sync_generator(state: dict) -> list[str]:
            return ["方案A", "方案B"]

        def sync_evaluator(state: dict, thought: str) -> float:
            return 0.5 if "A" in thought else 0.3

        sync_tree = sync_mcts.search(
            root_thought="决策",
            root_state={},
            generator=sync_generator,
            evaluator=sync_evaluator,
        )

        # 异步版本
        async_mcts = AsyncMCTS(
            exploration_weight=exploration_weight,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )

        async def async_generator(state: dict) -> list[str]:
            return ["方案A", "方案B"]

        async def async_evaluator(state: dict, thought: str) -> float:
            return 0.5 if "A" in thought else 0.3

        async_tree = asyncio.run(async_mcts.search(
            root_thought="决策",
            root_state={},
            generator=async_generator,
            evaluator=async_evaluator,
        ))

        # 根节点应相同
        assert sync_tree.root is not None
        assert async_tree.root is not None
        assert sync_tree.root.thought == async_tree.root.thought
        assert sync_tree.size > 1
        assert async_tree.size > 1

    def test_concurrency_parameter(self):
        """concurrency 参数 -- 验证 Semaphore 生效"""
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0

        mcts = AsyncMCTS(
            max_iterations=10, max_depth=2, concurrency=2,
        )

        async def generator(state: dict) -> list[str]:
            return ["c1", "c2", "c3", "c4"]

        async def evaluator(state: dict, thought: str) -> float:
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            call_count += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return 0.5

        asyncio.run(mcts.search(
            root_thought="测试",
            root_state={},
            generator=generator,
            evaluator=evaluator,
        ))

        # concurrency=2 意味着同时最多 2 个并发评估
        assert max_concurrent <= 2
        assert call_count > 0

    def test_repr(self):
        mcts = AsyncMCTS(concurrency=4, max_iterations=100)
        r = repr(mcts)
        assert "AsyncMCTS" in r
        assert "concurrency=4" in r
