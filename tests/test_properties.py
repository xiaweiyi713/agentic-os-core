"""Property-based tests using Hypothesis for invariant verification."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_os import EdgeType, KnowledgeGraph, WorkingMemory
from agentic_os.core.graph.edge import Edge
from agentic_os.core.graph.node import create_episode
from agentic_os.core.graph.scoring import compute_pagerank
from agentic_os.exceptions import ValidationError

# --- Strategies ---

float_01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
float_outside = st.one_of(
    st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, allow_nan=False, allow_infinity=False),
)
text = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))
node_contents = st.lists(text, min_size=1, max_size=20, unique=True)


# --- Edge weight property ---

class TestEdgeWeightProperty:
    """Edge creation validates weight strictly within [0, 1]."""

    @given(weight=float_01)
    @settings(max_examples=50, deadline=None)
    def test_valid_weight(self, weight):
        Edge("a", "b", EdgeType.CAUSAL, weight)

    @given(weight=float_outside)
    @settings(max_examples=50, deadline=None)
    def test_invalid_weight(self, weight):
        try:
            Edge("a", "b", EdgeType.CAUSAL, weight)
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            pass


# --- KnowledgeGraph add/remove invariant ---

class TestGraphInvariant:
    """Add then remove yields an empty graph."""

    @given(contents=node_contents)
    @settings(max_examples=50, deadline=None)
    def test_add_remove_empty(self, contents):
        kg = KnowledgeGraph()
        nodes = []
        for c in contents:
            n = create_episode(c)
            kg.add_node(n)
            nodes.append(n)
        assert kg.node_count == len(contents)
        for n in nodes:
            kg.remove_node(n.id)
        assert kg.node_count == 0
        assert kg.edge_count == 0


# --- PageRank sum invariant ---

class TestPageRankProperty:
    """PageRank scores sum to approximately 1.0 for connected graphs."""

    @given(n_nodes=st.integers(min_value=2, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_scores_sum_to_one(self, n_nodes):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"n{i}") for i in range(n_nodes)]
        for n in nodes:
            kg.add_node(n)
        # Create a ring to ensure connectivity
        for i in range(n_nodes):
            kg.add_edge(nodes[i].id, nodes[(i + 1) % n_nodes].id, EdgeType.CAUSAL, 0.5)

        scores = compute_pagerank(kg)
        assert abs(sum(scores.values()) - 1.0) < 0.05


# --- WorkingMemory capacity invariant ---

class TestWorkingMemoryProperty:
    """Size never exceeds capacity."""

    @given(
        capacity=st.integers(min_value=1, max_value=10),
        items=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=30),
    )
    @settings(max_examples=50, deadline=None)
    def test_size_never_exceeds_capacity(self, capacity, items):
        wm = WorkingMemory(capacity=capacity)
        for i, item in enumerate(items):
            wm.put(f"k{i}", {"content": item})
        assert wm.size <= capacity

    @given(
        capacity=st.integers(min_value=2, max_value=5),
        extra=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_lru_eviction_retains_latest(self, capacity, extra):
        wm = WorkingMemory(capacity=capacity)
        for i in range(capacity + extra):
            wm.put(f"k{i}", {"content": f"v{i}"})
        assert wm.size == capacity
        # The last `capacity` keys should remain
        for i in range(extra, capacity + extra):
            assert wm.get(f"k{i}") is not None


# --- Serialization roundtrip invariant ---

class TestSerializationProperty:
    """Roundtrip serialization preserves node and edge counts."""

    @given(
        n_nodes=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_roundtrip_preserves_structure(self, n_nodes):
        kg = KnowledgeGraph()
        nodes = [create_episode(f"node_{i}") for i in range(n_nodes)]
        for n in nodes:
            kg.add_node(n)

        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)
        assert kg2.node_count == kg.node_count
        assert kg2.edge_count == kg.edge_count
        for n in nodes:
            assert kg2.get_node(n.id) is not None
