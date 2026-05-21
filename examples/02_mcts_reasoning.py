"""示例2: MCTS 蒙特卡洛树搜索推理"""

from agentic_os.core.tree.mcts import MCTS
from agentic_os.core.tree.pruning import score_threshold_prune
from agentic_os.core.tree.search import beam_search


def main():
    print("=== MCTS 决策推理示例 ===\n")

    # 问题: 如何提高系统性能?
    mcts = MCTS(exploration_weight=1.414, max_iterations=100, max_depth=4)

    # 候选方案生成器
    solutions_db = {
        0: ["优化数据库查询", "增加缓存层", "升级服务器硬件"],
        1: ["添加数据库索引", "使用查询优化器", "读写分离"],
        2: ["使用 Redis 缓存", "使用本地缓存", "CDN 缓存"],
        3: ["增加 CPU", "增加内存", "使用 SSD"],
    }

    def generator(state):
        depth = state.get("depth", 0)
        return solutions_db.get(depth, ["进一步分析", "收集更多数据"])

    # 评估函数: 基于方案名称的启发式评分
    def evaluator(state, thought):
        scores = {
            "增加缓存层": 0.85, "使用 Redis 缓存": 0.9,
            "优化数据库查询": 0.8, "添加数据库索引": 0.88,
            "读写分离": 0.75, "CDN 缓存": 0.7,
            "升级服务器硬件": 0.5, "增加 CPU": 0.4,
            "使用查询优化器": 0.65, "使用本地缓存": 0.6,
            "增加内存": 0.45, "使用 SSD": 0.55,
        }
        return scores.get(thought, 0.3)

    tree = mcts.search(
        root_thought="如何提高系统性能?",
        root_state={"depth": 0},
        generator=generator,
        evaluator=evaluator,
    )

    print("思维树结构:")
    print(tree.visualize())

    print(f"\n树大小: {tree.size} 个节点")

    # 最优推理路径
    print("\n=== 最优推理路径 ===\n")
    best_path = tree.get_best_path()
    for i, node in enumerate(best_path):
        avg = f"{node.avg_score:.2f}" if node.visits > 0 else f"{node.score:.2f}"
        print(f"  步骤 {i}: {node.thought} (得分: {avg}, 访问: {node.visits})")

    # 所有路径
    print(f"\n=== 所有完整路径 ({len(tree.get_all_paths())} 条) ===\n")
    for i, path in enumerate(tree.get_all_paths()):
        thoughts = " → ".join(n.thought[:15] for n in path)
        print(f"  路径 {i + 1}: {thoughts}")

    # 剪枝
    print("\n=== 剪枝 (得分 < 0.5) ===\n")
    removed = score_threshold_prune(tree, 0.5)
    print(f"  移除了 {removed} 个低分节点")
    print(f"  剪枝后树大小: {tree.size}")

    # Beam Search
    print("\n=== Beam Search (beam_width=2) ===\n")
    if tree.root and tree.root.children:
        results = beam_search(tree.root, beam_width=2, evaluator=lambda n: n.avg_score)
        for node in results:
            print(f"  → {node.thought} (得分: {node.avg_score:.2f})")


if __name__ == "__main__":
    main()
