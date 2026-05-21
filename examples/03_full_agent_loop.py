"""示例3: 完整 Agent 循环 - 感知→记忆→规划→执行→反思"""

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.memory.manager import MemoryManager
from agentic_os.core.planning.executor import Executor
from agentic_os.core.planning.goal import GoalPriority, create_goal
from agentic_os.core.planning.planner import Planner
from agentic_os.plugins.mock import MockEvaluator, MockLLM


def main():
    print("=== 完整 Agent 循环示例 ===\n")

    # ── 初始化 ──
    memory = MemoryManager(working_capacity=20)
    planner = Planner(mcts_iterations=30)
    executor = Executor(planner)
    _llm = MockLLM()  # retained for demonstration
    evaluator = MockEvaluator(bias=0.6)

    # ── 阶段 1: 感知 ──
    print("--- 阶段 1: 感知 (接收输入) ---\n")
    user_inputs = [
        "需要开发一个 REST API 服务",
        "要求支持用户认证和授权",
        "使用 Python FastAPI 框架",
        "需要连接 PostgreSQL 数据库",
    ]
    for inp in user_inputs:
        memory.add_experience(inp)
        print(f"  📥 {inp}")

    # ── 阶段 2: 记忆处理 ──
    print("\n--- 阶段 2: 记忆巩固 ---\n")
    count = memory.consolidate()
    print(f"  巩固了 {count} 条短期记忆到长期记忆")

    memory.store_fact("FastAPI 支持自动 OpenAPI 文档生成", importance=0.7)
    memory.store_fact("JWT 是常见的 API 认证方案", importance=0.8)
    print("  存储了 2 条相关知识")

    # 反思
    reflection_ids = memory.reflect()
    print(f"  生成了 {len(reflection_ids)} 条反思")

    # ── 阶段 3: 规划 ──
    print("\n--- 阶段 3: 规划 ---\n")
    root = create_goal("开发 REST API 服务", priority=GoalPriority.HIGH)
    planner.add_goal(root)

    # 分解目标
    phases = planner.decompose(root, [
        "搭建 FastAPI 项目骨架",
        "设计数据库模型",
        "实现用户认证模块",
        "实现 API 端点",
        "编写测试",
        "部署",
    ])
    print(f"  分解为 {len(phases)} 个子目标:")
    for i, sub in enumerate(phases):
        print(f"    {i + 1}. {sub.description}")

    # 生成计划
    plan = planner.create_plan(root)
    print(f"\n  执行计划 ({len(plan)} 步):")
    for i, goal in enumerate(plan):
        print(f"    {i + 1}. [{goal.priority.name}] {goal.description}")

    # ToT 评估
    print("\n  使用 ToT 评估计划质量...")
    tot_tree = planner.evaluate_plan_with_tot(plan, evaluator.evaluate)
    best = tot_tree.get_best_path()
    if best:
        print(f"  ToT 最优路径长度: {len(best)}")

    # ── 阶段 4: 执行 ──
    print("\n--- 阶段 4: 执行 ---\n")
    execution_outputs = [
        (True, "开始开发 REST API 服务"),
        (True, "FastAPI 项目创建成功，目录结构: app/, tests/, docs/"),
        (True, "用户模型定义完成，包含 id/username/email/password_hash"),
        (True, "JWT 认证模块实现，支持注册/登录/刷新令牌"),
        (True, "5 个 API 端点实现: GET/POST/PUT/DELETE /users, POST /auth"),
        (True, "12 个测试用例全部通过"),
        (True, "Docker 镜像构建并部署到 staging 环境"),
    ]
    outputs = iter(execution_outputs)
    logs = executor.execute_plan(plan, lambda g: next(outputs))

    for log in logs:
        goal = planner.get_goal(log.goal_id)
        status = "✅" if log.success else "❌"
        desc = goal.description if goal else log.goal_id
        print(f"  {status} {desc}")
        print(f"     {log.result}")

    print(f"\n  成功: {executor.success_count}, 失败: {executor.failure_count}")

    # ── 阶段 5: 反思与学习 ──
    print("\n--- 阶段 5: 反思与学习 ---\n")

    # 记录执行经验
    for log in logs:
        status = "成功" if log.success else "失败"
        memory.add_experience(f"执行 '{log.goal_id}': {status} - {log.result}")

    # 生成反思
    memory.store_reflection(
        "REST API 项目顺利完成。FastAPI + JWT + PostgreSQL 是一个高效的技术栈组合。",
        importance=0.9,
    )
    memory.store_reflection(
        "数据库模型设计应在编码前完成，可以减少返工。",
        importance=0.85,
    )

    # 建立因果链
    jwt_nodes = memory.recall("JWT")
    auth_nodes = memory.recall("认证")
    if jwt_nodes and auth_nodes and jwt_nodes[0].id != auth_nodes[0].id:
        memory.link_memories(
            jwt_nodes[0].id, auth_nodes[0].id,
            edge_type=EdgeType.CAUSAL,
        )
        print("  已建立 JWT → 认证 因果关联")

    # 记忆统计
    stats = memory.stats()
    print(f"  工作记忆: {stats['working_memory']} 条")
    print(f"  长期记忆: {stats['longterm_memory']['nodes']} 个节点, "
          f"{stats['longterm_memory']['edges']} 条边")
    print(f"  记忆类型分布: {stats['longterm_memory'].get('types', {})}")

    # 回忆测试
    print("\n  🧠 回忆 'API':")
    for node in memory.recall("API", top_k=3):
        print(f"    [{node.type.value}] {node.content[:50]}")

    print("\n=== Agent 循环完成 ===")


if __name__ == "__main__":
    main()
