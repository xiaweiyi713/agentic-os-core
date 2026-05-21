"""规划引擎测试"""


from agentic_os.core.planning.executor import Executor
from agentic_os.core.planning.goal import GoalPriority, GoalState, create_goal
from agentic_os.core.planning.planner import Planner
from agentic_os.plugins.mock import MockEvaluator


class TestGoal:
    def test_create_goal(self):
        goal = create_goal("完成项目", priority=GoalPriority.HIGH)
        assert goal.description == "完成项目"
        assert goal.priority == GoalPriority.HIGH
        assert goal.state == GoalState.PENDING

    def test_goal_state_transitions(self):
        goal = create_goal("测试目标")
        assert goal.state == GoalState.PENDING

        goal.start()
        assert goal.state == GoalState.IN_PROGRESS

        goal.complete("成功完成")
        assert goal.state == GoalState.COMPLETED
        assert goal.result == "成功完成"

    def test_goal_fail(self):
        goal = create_goal("失败目标")
        goal.start()
        goal.fail("资源不足")
        assert goal.state == GoalState.FAILED
        assert goal.is_terminal

    def test_goal_cancel(self):
        goal = create_goal("取消目标")
        goal.cancel()
        assert goal.state == GoalState.CANCELLED
        assert goal.is_terminal


class TestPlanner:
    def test_add_and_get_goal(self):
        planner = Planner()
        goal = create_goal("测试目标")
        gid = planner.add_goal(goal)
        assert planner.get_goal(gid) == goal

    def test_decompose(self):
        planner = Planner()
        root = create_goal("完成项目")
        planner.add_goal(root)

        subs = planner.decompose(root, ["需求分析", "编码", "测试"])
        assert len(subs) == 3
        assert len(root.subgoals) == 3
        for sub in subs:
            assert sub.parent_id == root.id

    def test_create_plan(self):
        planner = Planner()
        root = create_goal("项目", priority=GoalPriority.HIGH)
        planner.add_goal(root)

        sub_a = create_goal("步骤A", priority=GoalPriority.HIGH)
        sub_b = create_goal("步骤B", priority=GoalPriority.MEDIUM)
        sub_b.dependencies = [sub_a.id]
        planner.add_goal(sub_a)
        planner.add_goal(sub_b)
        root.subgoals = [sub_a.id, sub_b.id]

        plan = planner.create_plan(root)
        assert len(plan) >= 2
        # A should come before B (B depends on A)
        a_idx = next(i for i, g in enumerate(plan) if g.id == sub_a.id)
        b_idx = next(i for i, g in enumerate(plan) if g.id == sub_b.id)
        assert a_idx < b_idx

    def test_create_plan_respects_priority(self):
        planner = Planner()
        root = create_goal("项目")
        planner.add_goal(root)

        high = create_goal("高优先级", priority=GoalPriority.HIGH)
        low = create_goal("低优先级", priority=GoalPriority.LOW)
        planner.add_goal(high)
        planner.add_goal(low)
        root.subgoals = [high.id, low.id]

        plan = planner.create_plan(root)
        high_idx = next(i for i, g in enumerate(plan) if g.id == high.id)
        low_idx = next(i for i, g in enumerate(plan) if g.id == low.id)
        assert high_idx < low_idx

    def test_evaluate_plan_with_tot(self):
        planner = Planner(mcts_iterations=30)
        root = create_goal("项目")
        planner.add_goal(root)

        sub = create_goal("步骤1")
        planner.add_goal(sub)
        root.subgoals = [sub.id]

        evaluator = MockEvaluator(bias=0.6)
        tree = planner.evaluate_plan_with_tot([root, sub], evaluator.evaluate)
        assert tree.root is not None

    def test_revise_plan(self):
        planner = Planner()
        root = create_goal("项目")
        planner.add_goal(root)

        sub = create_goal("失败步骤")
        planner.add_goal(sub)
        root.subgoals = [sub.id]

        new_subs = planner.revise_plan(sub, ["重试方案A", "备用方案B"])
        assert len(new_subs) == 2
        assert sub.state == GoalState.FAILED


class TestExecutor:
    def test_execute_plan_success(self):
        planner = Planner()
        goal = create_goal("简单任务")
        planner.add_goal(goal)

        executor = Executor(planner)
        logs = executor.execute_plan([goal], lambda g: (True, "完成"))

        assert len(logs) == 1
        assert logs[0].success
        assert goal.state == GoalState.COMPLETED

    def test_execute_plan_failure_stops(self):
        planner = Planner()
        goals = [create_goal(f"任务{i}") for i in range(3)]
        for g in goals:
            planner.add_goal(g)

        executor = Executor(planner)
        results = iter([(False, "失败"), (True, "不该执行")])
        logs = executor.execute_plan(goals, lambda g: next(results))

        assert len(logs) == 1
        assert not logs[0].success
        assert goals[1].state == GoalState.PENDING  # 后续任务未执行

    def test_execute_with_retry(self):
        planner = Planner()
        goal = create_goal("需要重试")
        planner.add_goal(goal)

        executor = Executor(planner)
        attempt_count = 0

        def action(g):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return (False, "重试中")
            return (True, "终于成功")

        logs = executor.execute_with_retry([goal], action, max_retries=3)
        assert len(logs) == 1
        assert logs[0].success

    def test_skip_terminal_goals(self):
        planner = Planner()
        goal = create_goal("已完成")
        goal.complete("已做完")
        planner.add_goal(goal)

        executor = Executor(planner)
        logs = executor.execute_plan([goal], lambda g: (True, ""))
        assert len(logs) == 0

    def test_execution_stats(self):
        planner = Planner()
        goals = [create_goal(f"T{i}") for i in range(3)]
        for g in goals:
            planner.add_goal(g)

        executor = Executor(planner)
        results = iter([(True, "ok"), (False, "fail"), (True, "ok")])
        executor.execute_plan(goals, lambda g: next(results))

        assert executor.success_count == 1
        assert executor.failure_count == 1
