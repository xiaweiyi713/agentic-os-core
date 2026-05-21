"""Performance benchmarks for Agentic OS Core.

Usage:
    python benchmarks/run_benchmarks.py
"""

import logging
import platform
import random
import statistics
import time

import agentic_os
from agentic_os import (
    MCTS,
    EdgeType,
    Executor,
    GoalPriority,
    KnowledgeGraph,
    MemoryManager,
    Planner,
    create_episode,
)
from agentic_os.core.graph.scoring import compute_pagerank
from agentic_os.core.graph.traversal import bfs, shortest_path
from agentic_os.core.planning.goal import create_goal

logging.getLogger("agentic_os").setLevel(logging.ERROR)


def bench(label: str, fn, iterations: int = 100) -> dict:
    """Run a benchmark function multiple times and return stats."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "label": label,
        "iterations": iterations,
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "p95_ms": sorted(times)[int(0.95 * len(times))] * 1000,
        "min_ms": min(times) * 1000,
    }


def print_results(results: list[dict]) -> None:
    print(f"{'Benchmark':<45} {'Iters':>6} {'Mean':>9} {'Median':>9} {'P95':>9} {'Min':>9}")
    print("─" * 92)
    for r in results:
        print(f"{r['label']:<45} {r['iterations']:>6} {r['mean_ms']:>8.3f}ms {r['median_ms']:>8.3f}ms "
              f"{r['p95_ms']:>8.3f}ms {r['min_ms']:>8.3f}ms")


def main():
    all_results = []

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Agentic OS Core — Performance Benchmarks            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Version:  {agentic_os.__version__}")
    print(f"  Python:   {platform.python_version()} | {platform.machine()} | {platform.system()}")
    print()

    # ── Knowledge Graph ──
    print("── Knowledge Graph ──\n")

    # Node operations
    kg = KnowledgeGraph()
    all_results.append(bench("KG: add 100 nodes", lambda: _add_nodes(kg, 100)))
    all_results.append(bench("KG: add edge", lambda: _add_random_edge(kg), 500))
    all_results.append(bench("KG: get node", lambda: _get_node(kg), 1000))
    all_results.append(bench("KG: find_nodes (keyword)", lambda: kg.find_nodes("test"), 500))
    all_results.append(bench("KG: out_neighbors", lambda: _get_neighbors(kg), 1000))

    # Large graph
    large_kg = _build_large_graph(1000)
    all_results.append(bench("KG: build 1000-node graph", lambda: _build_large_graph(1000), 10))
    all_results.append(bench("KG[1000]: find_nodes", lambda: large_kg.find_nodes("node"), 200))
    all_results.append(bench("KG[1000]: subgraph(depth=3)", lambda: large_kg.subgraph(next(n.id for n in large_kg.nodes), 3), 100))

    # Traversal
    first_node = next(n.id for n in large_kg.nodes)
    all_results.append(bench("KG[1000]: BFS", lambda: bfs(large_kg, first_node), 50))
    all_results.append(bench("KG[1000]: shortest_path", lambda: _find_path(large_kg), 50))

    # Scoring
    all_results.append(bench("KG[1000]: PageRank (20 iter)", lambda: compute_pagerank(large_kg), 10))

    # Serialization
    all_results.append(bench("KG[1000]: to_dict()", lambda: large_kg.to_dict(), 50))
    data = large_kg.to_dict()
    all_results.append(bench("KG[1000]: from_dict()", lambda: KnowledgeGraph.from_dict(data), 20))

    print_results(all_results)
    print()

    # ── MCTS ──
    print("── MCTS Tree Search ──\n")

    mcts_results = []
    mcts_results.append(bench("MCTS: 100 iterations, depth 3",
        lambda: _run_mcts(100, 3), 20))
    mcts_results.append(bench("MCTS: 500 iterations, depth 5",
        lambda: _run_mcts(500, 5), 10))
    mcts_results.append(bench("MCTS: 1000 iterations, depth 5",
        lambda: _run_mcts(1000, 5), 5))
    print_results(mcts_results)
    print()

    # ── Memory ──
    print("── Memory Management ──\n")

    mem_results = []
    wm_bench = _make_working_memory(100)
    mem_results.append(bench("WM: put (capacity=100)", lambda: _wm_put(wm_bench), 1000))
    mem_results.append(bench("WM: get", lambda: _wm_get(wm_bench), 1000))
    mem_results.append(bench("WM: LRU eviction (over capacity)",
        lambda: _wm_evict(), 500))

    mm = _make_memory_manager(50)
    mem_results.append(bench("MM: add_experience", lambda: mm.add_experience("test data"), 500))
    mem_results.append(bench("MM: recall", lambda: mm.recall("test"), 200))
    mem_results.append(bench("MM: consolidate", lambda: _mm_consolidate(), 100))
    mem_results.append(bench("MM: reflect", lambda: _mm_reflect(), 100))
    print_results(mem_results)
    print()

    # ── Planning ──
    print("── Planning Engine ──\n")

    plan_results = []
    plan_results.append(bench("Planner: decompose (6 subgoals)",
        lambda: _plan_decompose(), 200))
    plan_results.append(bench("Planner: create_plan",
        lambda: _plan_create(), 200))
    plan_results.append(bench("Executor: execute (7 goals)",
        lambda: _plan_execute(), 500))
    print_results(plan_results)
    print()

    # ── Summary ──
    print("═" * 92)
    print("Summary")
    print("═" * 92)
    total = all_results + mcts_results + mem_results + plan_results
    print(f"  Total benchmarks: {len(total)}")
    fastest = min(total, key=lambda r: r["mean_ms"])
    slowest = max(total, key=lambda r: r["mean_ms"])
    avg = statistics.mean(r["mean_ms"] for r in total)
    print(f"  Fastest:  {fastest['label']} ({fastest['mean_ms']:.3f}ms)")
    print(f"  Slowest:  {slowest['label']} ({slowest['mean_ms']:.3f}ms)")
    print(f"  Average:  {avg:.3f}ms")
    print()
    print("  By module:")
    for label, group in [
        ("Knowledge Graph", all_results),
        ("MCTS", mcts_results),
        ("Memory", mem_results),
        ("Planning", plan_results),
    ]:
        mod_avg = statistics.mean(r["mean_ms"] for r in group)
        print(f"    {label:<20} {len(group):>2} benchmarks  avg {mod_avg:>8.3f}ms")


# ── Helpers ──

_counter = 0

def _next_id():
    global _counter
    _counter += 1
    return _counter

def _add_nodes(kg, n):
    for _i in range(n):
        kg.add_node(create_episode(f"node_{_next_id()}"))

def _add_random_edge(kg):
    nodes = [n.id for n in kg.nodes]
    if len(nodes) < 2:
        return
    a, b = random.sample(nodes, 2)
    kg.add_edge(a, b, EdgeType.ASSOCIATIVE, 0.5)

def _get_node(kg):
    nodes = [n.id for n in kg.nodes]
    if nodes:
        kg.get_node(random.choice(nodes))

def _get_neighbors(kg):
    nodes = [n.id for n in kg.nodes]
    if nodes:
        kg.out_neighbors(random.choice(nodes))

def _build_large_graph(n):
    kg = KnowledgeGraph()
    nodes = [create_episode(f"node_{i} test data point") for i in range(n)]
    for node in nodes:
        kg.add_node(node)
    for _i in range(n * 2):
        a, b = random.sample(nodes, 2)
        kg.add_edge(a.id, b.id, EdgeType.ASSOCIATIVE, random.random())
    return kg

def _find_path(kg):
    nodes = [n.id for n in kg.nodes]
    if len(nodes) >= 2:
        shortest_path(kg, nodes[0], nodes[-1])

def _run_mcts(iters, depth):
    mcts = MCTS(max_iterations=iters, max_depth=depth)
    def gen(state):
        return [f"option_{i}" for i in range(3)]
    def ev(state, thought):
        return 0.5
    mcts.search("test", {}, gen, ev)

def _make_working_memory(cap):
    from agentic_os import WorkingMemory
    wm = WorkingMemory(capacity=cap)
    for i in range(cap):
        wm.put_content(f"item_{i}")
    return wm

def _wm_put(wm):
    wm.put_content(f"new_{_next_id()}")

def _wm_get(wm):
    items = wm.peek_all()
    if items:
        wm.get(items[0][0])

def _wm_evict():
    from agentic_os import WorkingMemory
    wm = WorkingMemory(capacity=10)
    for i in range(20):
        wm.put_content(f"evict_{i}")

def _make_memory_manager(n):
    mm = MemoryManager(working_capacity=n)
    for i in range(n):
        mm.add_experience(f"experience_{i} test data")
    return mm

def _mm_consolidate():
    mm = MemoryManager()
    for i in range(20):
        mm.add_experience(f"data_{i}")
    mm.consolidate()

def _mm_reflect():
    mm = MemoryManager()
    for i in range(10):
        mm.add_experience(f"experience about topic_{i % 3}")
    mm.reflect()

def _plan_decompose():
    planner = Planner()
    root = create_goal("project", priority=GoalPriority.HIGH)
    planner.add_goal(root)
    planner.decompose(root, [f"step_{i}" for i in range(6)])

def _plan_create():
    planner = Planner()
    root = create_goal("project")
    planner.add_goal(root)
    planner.decompose(root, ["A", "B", "C", "D", "E", "F"])
    planner.create_plan(root)

def _plan_execute():
    planner = Planner()
    root = create_goal("project")
    planner.add_goal(root)
    planner.decompose(root, ["A", "B", "C", "D", "E", "F"])
    plan = planner.create_plan(root)
    executor = Executor(planner)
    executor.execute_plan(plan, lambda g: (True, "ok"))


if __name__ == "__main__":
    main()
