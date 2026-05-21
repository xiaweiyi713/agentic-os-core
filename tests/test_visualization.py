"""Tests for the visualization modules (HTML string validation only)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentic_os import (
    EdgeType,
    KnowledgeGraph,
    KnowledgeGraphVisualizer,
    NodeType,
    ThoughtNode,
    ThoughtTree,
    ThoughtTreeVisualizer,
    create_episode,
    create_fact,
    create_goal,
    create_reflection,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_sample_graph() -> KnowledgeGraph:
    """Return a small multi-type knowledge graph with edges."""
    kg = KnowledgeGraph()
    n1 = create_episode("user asked about weather")
    n2 = create_fact("Python is a programming language", importance=0.8)
    n3 = create_reflection("I should learn more about NLP")
    n4 = create_goal("Improve response quality", importance=0.9)
    for n in (n1, n2, n3, n4):
        kg.add_node(n)
    kg.add_edge(n1.id, n2.id, EdgeType.ASSOCIATIVE, weight=0.7)
    kg.add_edge(n2.id, n3.id, EdgeType.DERIVED_FROM, weight=0.5)
    kg.add_edge(n3.id, n4.id, EdgeType.CAUSAL, weight=0.9)
    return kg


def _build_sample_tree() -> ThoughtTree:
    """Return a small thought tree with branching."""
    tree = ThoughtTree(max_depth=3, max_children=3)
    root = tree.set_root("What is the best approach?")
    a = tree.add_thought(root, "Try brute force", score=0.3)
    b = tree.add_thought(root, "Use dynamic programming", score=0.9)
    c = tree.add_thought(root, "Apply greedy algorithm", score=0.6)
    if a is not None:
        tree.add_thought(a, "Brute force sub-step A", score=0.2)
    if b is not None:
        tree.add_thought(b, "DP sub-step with memoization", score=0.95)
    if c is not None:
        tree.add_thought(c, "Greedy approximation", score=0.55)
    return tree


# ------------------------------------------------------------------
# KnowledgeGraphVisualizer tests
# ------------------------------------------------------------------


class TestKnowledgeGraphVisualizer:
    """Validate generated HTML content for knowledge graph visualisation."""

    def test_html_contains_d3_cdn(self) -> None:
        """Generated HTML must reference the D3.js CDN."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "graph.html")
            html_str = viz.to_html(kg, path)
        assert "cdn.jsdelivr.net/npm/d3@7" in html_str

    def test_html_contains_node_data(self) -> None:
        """Generated HTML must embed node data."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(kg, str(Path(tmp) / "graph.html"))
        assert "user asked about weather" in html_str
        assert "Python is a programming language" in html_str

    def test_html_contains_edge_data(self) -> None:
        """Generated HTML must embed edge data."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(kg, str(Path(tmp) / "graph.html"))
        assert "associative" in html_str
        assert "derived_from" in html_str
        assert "causal" in html_str

    def test_empty_graph_produces_valid_html(self) -> None:
        """An empty graph should still produce a valid HTML document."""
        viz = KnowledgeGraphVisualizer()
        kg = KnowledgeGraph()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "empty.html")
            html_str = viz.to_html(kg, path)
            assert Path(path).exists()
        assert "<!DOCTYPE html>" in html_str
        assert "</html>" in html_str
        assert "cdn.jsdelivr.net/npm/d3@7" in html_str
        # Node/edge arrays should be empty -- JS assigns them
        assert "const nodesData = [];" in html_str or "const nodesData=[];" in html_str.replace(" ", "")
        assert "const edgesData = [];" in html_str or "const edgesData=[];" in html_str.replace(" ", "")

    def test_node_type_coloring(self) -> None:
        """The four NodeType colours should be present in the output."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(kg, str(Path(tmp) / "colors.html"))
        # Check that the _NODE_COLORS mapping is inlined
        assert "#4A90D9" in html_str  # episode blue
        assert "#27AE60" in html_str  # fact green
        assert "#F5A623" in html_str  # reflection orange
        assert "#E74C3C" in html_str  # goal red

    def test_writes_file_to_disk(self) -> None:
        """The output file should be written and contain the HTML."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "output.html")
            result = viz.to_html(kg, path)
            on_disk = Path(path).read_text(encoding="utf-8")
        assert on_disk == result
        assert len(on_disk) > 100

    def test_custom_title(self) -> None:
        """Custom title should appear in the HTML."""
        viz = KnowledgeGraphVisualizer()
        kg = _build_sample_graph()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(kg, str(Path(tmp) / "t.html"), title="My Graph")
        assert "<title>My Graph</title>" in html_str


# ------------------------------------------------------------------
# ThoughtTreeVisualizer tests
# ------------------------------------------------------------------


class TestThoughtTreeVisualizer:
    """Validate generated HTML content for thought tree visualisation."""

    def test_html_contains_d3_cdn(self) -> None:
        """Generated HTML must reference the D3.js CDN."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(tree, str(Path(tmp) / "tree.html"))
        assert "cdn.jsdelivr.net/npm/d3@7" in html_str

    def test_html_contains_thought_data(self) -> None:
        """Generated HTML must embed thought text data."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(tree, str(Path(tmp) / "tree.html"))
        assert "What is the best approach?" in html_str
        assert "Use dynamic programming" in html_str

    def test_empty_tree_produces_valid_html(self) -> None:
        """An empty tree should still produce a valid HTML document."""
        viz = ThoughtTreeVisualizer()
        tree = ThoughtTree()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "empty_tree.html")
            html_str = viz.to_html(tree, path)
            assert Path(path).exists()
        assert "<!DOCTYPE html>" in html_str
        assert "</html>" in html_str
        assert "const thoughtsData = [];" in html_str or "const thoughtsData=[];" in html_str.replace(" ", "")

    def test_best_path_highlighted(self) -> None:
        """The best path IDs must be embedded in the HTML for highlighting."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        best = tree.get_best_path()
        best_ids = [id(n) for n in best]
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(tree, str(Path(tmp) / "best.html"))
        # The best_path_ids JSON should be present
        for bid in best_ids:
            assert str(bid) in html_str

    def test_score_colour_gradient_in_output(self) -> None:
        """The score colour function should generate red-to-green values."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(tree, str(Path(tmp) / "colors.html"))
        # The inline JS contains scoreColor function references
        assert "scoreColor" in html_str

    def test_writes_file_to_disk(self) -> None:
        """The output file should be written and match the returned string."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "disk.html")
            result = viz.to_html(tree, path)
            on_disk = Path(path).read_text(encoding="utf-8")
        assert on_disk == result

    def test_custom_title(self) -> None:
        """Custom title should appear in the HTML."""
        viz = ThoughtTreeVisualizer()
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmp:
            html_str = viz.to_html(tree, str(Path(tmp) / "t.html"), title="My Tree")
        assert "<title>My Tree</title>" in html_str
