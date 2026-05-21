"""Importance scoring -- PageRank-style iterative algorithm."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentic_os.core.graph.knowledge_graph import KnowledgeGraph


def compute_pagerank(graph: KnowledgeGraph, damping: float = 0.85,
                     iterations: int = 20, tolerance: float = 1e-6) -> dict[str, float]:
    """Compute PageRank-style importance scores for all nodes.

    Uses iterative power iteration with early stopping when the total
    score delta falls below ``tolerance``.

    Time complexity: O(iterations * E) where E is the number of edges.

    Args:
        graph: The knowledge graph to score.
        damping: Probability of following an outgoing edge (random surfer).
            Defaults to 0.85.
        iterations: Maximum number of power-iteration rounds. Defaults to 20.
        tolerance: Convergence threshold on the L1 delta. Defaults to 1e-6.

    Returns:
        A ``{node_id: importance_score}`` mapping. Scores sum to 1.0.

    Example::

        scores = compute_pagerank(kg)
        top_node = max(scores, key=scores.get)
    """
    n = graph.node_count
    if n == 0:
        return {}

    logger.debug("compute_pagerank nodes=%d", n)
    scores: dict[str, float] = {nid: 1.0 / n for nid in graph._nodes}
    out_degree: dict[str, int] = {nid: len(graph.out_neighbors(nid)) for nid in graph._nodes}

    for i in range(iterations):
        new_scores: dict[str, float] = {}
        dangling_mass = sum(
            scores[nid] for nid in graph._nodes if out_degree[nid] == 0
        )
        for nid in graph._nodes:
            rank = (1 - damping) / n + damping * dangling_mass / n
            for src_id in graph.in_neighbors(nid):
                src_deg = out_degree.get(src_id, 0)
                if src_deg > 0:
                    rank += damping * scores[src_id] / src_deg
            new_scores[nid] = rank

        # Check convergence
        diff = sum(abs(new_scores[nid] - scores[nid]) for nid in scores)
        scores = new_scores
        if diff < tolerance:
            logger.debug("compute_pagerank converged at iter %d", i + 1)
            break

    return scores


def decay_scores(graph: KnowledgeGraph, factor: float = 0.95) -> None:
    """Multiply every node's importance by ``factor`` in-place.

    Useful for implementing time-based importance decay.

    Args:
        graph: The knowledge graph whose nodes to decay.
        factor: Multiplicative decay factor. Defaults to 0.95.

    Example::

        decay_scores(kg, factor=0.9)  # 10% decay
    """
    for node in graph.nodes:
        node.importance *= factor


def get_most_important(graph: KnowledgeGraph, n: int = 10) -> list[tuple[str, float]]:
    """Return the top-N nodes ranked by current importance.

    Time complexity: O(V log V) for sorting.

    Args:
        graph: The knowledge graph to query.
        n: Number of top nodes to return. Defaults to 10.

    Returns:
        A list of ``(node_id, importance)`` tuples sorted descending by
        importance.

    Example::

        top3 = get_most_important(kg, n=3)
        # [("node_1", 0.32), ("node_5", 0.21), ("node_2", 0.18)]
    """
    scored = [(node.id, node.importance) for node in graph.nodes]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def recompute_importance(graph: KnowledgeGraph, damping: float = 0.85) -> None:
    """Run PageRank and write scores back to each node's ``importance`` field.

    Args:
        graph: The knowledge graph to update in-place.
        damping: Damping factor passed to :func:`compute_pagerank`.

    Example::

        recompute_importance(kg)
        top = get_most_important(kg, n=5)
    """
    scores = compute_pagerank(graph, damping=damping)
    for nid, score in scores.items():
        node = graph.get_node(nid)
        if node:
            node.importance = score
    logger.info("recompute_importance updated %d nodes", len(scores))


def association_strength(graph: KnowledgeGraph, source_id: str, target_id: str) -> float:
    """Compute total edge weight between two nodes in both directions.

    Time complexity: O(degree(source_id)).

    Args:
        graph: The knowledge graph to query.
        source_id: ID of the first node.
        target_id: ID of the second node.

    Returns:
        The sum of ``edge.weight`` for all direct edges between the two
        nodes (both directions).

    Example::

        strength = association_strength(kg, "n1", "n2")
        # 1.7 if n1->n2 has weight 0.8 and n2->n1 has weight 0.9
    """
    strength = 0.0
    for edge in graph.out_edges(source_id):
        if edge.target_id == target_id:
            strength += edge.weight
    for edge in graph.in_edges(source_id):
        if edge.source_id == target_id:
            strength += edge.weight
    return strength
