"""示例1: 知识图谱构建、遍历与评分"""

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.knowledge_graph import KnowledgeGraph
from agentic_os.core.graph.node import create_episode, create_fact, create_reflection
from agentic_os.core.graph.scoring import compute_pagerank
from agentic_os.core.graph.traversal import bfs, shortest_path, topological_sort


def main():
    # 创建知识图谱
    kg = KnowledgeGraph()
    print("=== 构建知识图谱 ===\n")

    # 添加节点
    e1 = create_episode("今天学习了 Python 装饰器")
    e2 = create_episode("装饰器可以用于日志记录和权限检查")
    f1 = create_fact("Python 装饰器是高阶函数的语法糖", importance=0.8)
    f2 = create_fact("装饰器本质上是闭包的应用", importance=0.7)
    r1 = create_reflection("装饰器和闭包有紧密联系，理解闭包是掌握装饰器的关键", importance=0.9)

    for n in [e1, e2, f1, f2, r1]:
        kg.add_node(n)
        print(f"  添加节点: [{n.type.value}] {n.content[:40]}")

    # 添加边建立关联
    kg.add_edge(e1.id, f1.id, EdgeType.DERIVED_FROM, 0.8)
    kg.add_edge(e2.id, f1.id, EdgeType.SUPPORTS, 0.7)
    kg.add_edge(f1.id, f2.id, EdgeType.ASSOCIATIVE, 0.6)
    kg.add_edge(f2.id, r1.id, EdgeType.DERIVED_FROM, 0.9)
    kg.add_edge(e2.id, r1.id, EdgeType.DERIVED_FROM, 0.7)

    print(f"\n  节点数: {kg.node_count}, 边数: {kg.edge_count}")

    # BFS 遍历
    print("\n=== BFS 遍历 (从 e1 出发) ===\n")
    layers = bfs(kg, e1.id, max_depth=3)
    for depth, node_ids in layers.items():
        for nid in node_ids:
            node = kg.get_node(nid)
            if node:
                print(f"  深度 {depth}: [{node.type.value}] {node.content[:40]}")

    # 最短路径
    print("\n=== 最短路径: e1 → r1 ===\n")
    path = shortest_path(kg, e1.id, r1.id)
    if path:
        for nid in path:
            node = kg.get_node(nid)
            if node:
                print(f"  → [{node.type.value}] {node.content[:40]}")

    # 拓扑排序 (causal chain)
    print("\n=== 拓扑排序 (因果链) ===\n")
    order = topological_sort(kg)
    if order:
        for nid in order:
            node = kg.get_node(nid)
            if node:
                print(f"  → [{node.type.value}] {node.content[:40]}")

    # PageRank 重要性
    print("\n=== PageRank 重要性评分 ===\n")
    scores = compute_pagerank(kg)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for nid, score in sorted_scores:
        node = kg.get_node(nid)
        if node:
            print(f"  {score:.4f} - {node.content[:40]}")

    # 关键词搜索
    print("\n=== 关键词搜索: '装饰器' ===\n")
    results = kg.find_nodes("装饰器")
    for node in results:
        print(f"  [{node.type.value}] {node.content[:40]}")

    # 序列化
    print("\n=== 序列化/反序列化 ===\n")
    data = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(data)
    print(f"  原图: {kg.node_count} 节点, {kg.edge_count} 边")
    print(f"  恢复: {kg2.node_count} 节点, {kg2.edge_count} 边")

    print(f"\n图谱统计: {kg.stats()}")


if __name__ == "__main__":
    main()
