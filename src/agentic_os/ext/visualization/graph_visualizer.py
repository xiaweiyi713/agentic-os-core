"""Interactive HTML visualization for knowledge graphs using D3.js."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, ClassVar

from agentic_os.core.graph.knowledge_graph import KnowledgeGraph


class KnowledgeGraphVisualizer:
    """Generate interactive HTML visualization of knowledge graphs using D3.js.

    Produces a self-contained HTML file with a force-directed graph layout.
    Nodes are colored by ``NodeType`` and sized by importance. Edges are
    colored by ``EdgeType`` and their width reflects the edge weight.

    Interactions include: click-to-inspect detail panel, zoom/pan, drag,
    and a keyword search box.
    """

    _NODE_COLORS: ClassVar[dict[str, str]] = {
        "episode": "#4A90D9",
        "fact": "#27AE60",
        "reflection": "#F5A623",
        "goal": "#E74C3C",
    }

    _EDGE_COLORS: ClassVar[dict[str, str]] = {
        "causal": "#E74C3C",
        "temporal": "#3498DB",
        "associative": "#95A5A6",
        "derived_from": "#9B59B6",
        "supports": "#2ECC71",
        "contradicts": "#E67E22",
    }

    def to_html(
        self,
        graph: KnowledgeGraph,
        output_path: str,
        title: str = "Knowledge Graph",
    ) -> str:
        """Generate an interactive HTML file and write it to disk.

        Args:
            graph: The ``KnowledgeGraph`` to visualise.
            output_path: Destination file path for the HTML output.
            title: Page title shown in the browser tab.

        Returns:
            The generated HTML string.
        """
        data = graph.to_dict()
        rendered = self._render(title, data)
        Path(output_path).write_text(rendered, encoding="utf-8")
        return rendered

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self, title: str, data: dict[str, Any]) -> str:
        nodes_json = json.dumps(data.get("nodes", []), ensure_ascii=False)
        edges_json = json.dumps(data.get("edges", []), ensure_ascii=False)
        colors_json = json.dumps(self._NODE_COLORS)
        edge_colors_json = json.dumps(self._EDGE_COLORS)

        escaped_title = html.escape(title)

        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escaped_title}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; overflow: hidden; }}
  #search-box {{ position: absolute; top: 16px; left: 16px; z-index: 10; }}
  #search-box input {{ width: 240px; padding: 8px 12px; border-radius: 6px;
    border: 1px solid #444; background: #2a2a3e; color: #e0e0e0; font-size: 14px; }}
  #search-box input::placeholder {{ color: #888; }}
  svg {{ width: 100vw; height: 100vh; cursor: grab; }}
  svg:active {{ cursor: grabbing; }}
  .link {{ stroke-opacity: 0.6; fill: none; }}
  .node {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node:hover {{ stroke-width: 3px; }}
  .node.highlight {{ stroke: #fff; stroke-width: 3px; filter: drop-shadow(0 0 6px rgba(255,255,255,0.5)); }}
  .node.dim {{ opacity: 0.15; }}
  .link.dim {{ opacity: 0.05; }}
  .node-label {{ font-size: 11px; fill: #ccc; pointer-events: none; text-anchor: middle; }}
  #detail-panel {{ position: absolute; top: 16px; right: 16px; width: 320px;
    max-height: calc(100vh - 32px); overflow-y: auto; background: #2a2a3e;
    border-radius: 10px; padding: 20px; display: none; z-index: 10;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
  #detail-panel h3 {{ margin-bottom: 10px; color: #fff; font-size: 16px; }}
  #detail-panel .field {{ margin-bottom: 8px; font-size: 13px; }}
  #detail-panel .field span {{ color: #aaa; }}
  #detail-panel .close-btn {{ position: absolute; top: 10px; right: 14px;
    background: none; border: none; color: #aaa; font-size: 20px; cursor: pointer; }}
  #detail-panel .close-btn:hover {{ color: #fff; }}
  #legend {{ position: absolute; bottom: 16px; left: 16px; z-index: 10;
    background: #2a2a3e; border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}
  #legend h4 {{ margin-bottom: 6px; font-size: 12px; color: #aaa; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px;
    font-size: 12px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  #tooltip {{ position: absolute; display: none; background: rgba(0,0,0,0.85);
    color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 12px;
    pointer-events: none; z-index: 20; max-width: 300px; }}
</style>
</head>
<body>

<div id="search-box">
  <input type="text" id="search-input" placeholder="Search nodes..." />
</div>

<svg id="graph-svg"></svg>

<div id="tooltip"></div>

<div id="detail-panel">
  <button class="close-btn" id="close-detail">&times;</button>
  <h3 id="detail-title"></h3>
  <div id="detail-body"></div>
</div>

<div id="legend">
  <h4>Node Types</h4>
  <div id="legend-items"></div>
</div>

<script>
(function() {{
  const nodesData = {nodes_json};
  const edgesData = {edges_json};
  const nodeColors = {colors_json};
  const edgeColors = {edge_colors_json};

  const svg = d3.select("#graph-svg");
  const width = window.innerWidth;
  const height = window.innerHeight;
  svg.attr("viewBox", [0, 0, width, height]);

  // Zoom
  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => {{
    g.attr("transform", event.transform);
  }}));

  // Build D3 data
  const nodeMap = new Map();
  nodesData.forEach(n => nodeMap.set(n.id, n));
  const d3Nodes = nodesData.map(n => ({{...n}}));
  const d3Links = edgesData
    .filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
    .map(e => ({{source: e.source_id, target: e.target_id,
                 type: e.type, weight: e.weight, metadata: e.metadata}}));

  // Legend
  const legendItems = d3.select("#legend-items");
  Object.entries(nodeColors).forEach(([type, color]) => {{
    legendItems.append("div").attr("class", "legend-item")
      .html('<div class="legend-dot" style="background:' + color + '"></div>' + type);
  }});

  // Links
  const link = g.append("g").selectAll("line")
    .data(d3Links).join("line")
    .attr("class", "link")
    .attr("stroke", d => edgeColors[d.type] || "#666")
    .attr("stroke-width", d => Math.max(1, d.weight * 5));

  // Nodes
  const node = g.append("g").selectAll("circle")
    .data(d3Nodes).join("circle")
    .attr("class", "node")
    .attr("r", d => 6 + d.importance * 14)
    .attr("fill", d => nodeColors[d.type] || "#888")
    .call(d3.drag()
      .on("start", dragStart)
      .on("drag", dragging)
      .on("end", dragEnd));

  // Labels
  const label = g.append("g").selectAll("text")
    .data(d3Nodes).join("text")
    .attr("class", "node-label")
    .text(d => d.content.length > 18 ? d.content.substring(0, 18) + "..." : d.content);

  // Tooltip
  const tooltip = d3.select("#tooltip");
  node.on("mouseover", function(event, d) {{
    tooltip.style("display", "block")
      .html("<strong>" + d.type + "</strong><br/>" + d.content);
  }})
  .on("mousemove", function(event) {{
    tooltip.style("left", (event.pageX + 12) + "px")
           .style("top", (event.pageY - 20) + "px");
  }})
  .on("mouseout", function() {{
    tooltip.style("display", "none");
  }});

  // Detail panel
  node.on("click", function(event, d) {{
    event.stopPropagation();
    d3.select("#detail-title").text(d.type + ": " + (d.content.substring(0, 50)));
    let body = '<div class="field"><span>ID:</span> ' + d.id + '</div>';
    body += '<div class="field"><span>Importance:</span> ' + d.importance.toFixed(2) + '</div>';
    body += '<div class="field"><span>Access Count:</span> ' + d.access_count + '</div>';
    body += '<div class="field"><span>Content:</span> ' + d.content + '</div>';
    body += '<div class="field"><span>Created:</span> ' + d.created_at + '</div>';
    body += '<div class="field"><span>Updated:</span> ' + d.updated_at + '</div>';
    if (d.metadata && Object.keys(d.metadata).length > 0) {{
      body += '<div class="field"><span>Metadata:</span> ' + JSON.stringify(d.metadata) + '</div>';
    }}
    d3.select("#detail-body").html(body);
    d3.select("#detail-panel").style("display", "block");
  }});

  d3.select("#close-detail").on("click", () => {{
    d3.select("#detail-panel").style("display", "none");
  }});

  svg.on("click", () => {{
    d3.select("#detail-panel").style("display", "none");
  }});

  // Search
  d3.select("#search-input").on("input", function() {{
    const query = this.value.toLowerCase();
    if (!query) {{
      node.classed("highlight", false).classed("dim", false);
      link.classed("dim", false);
      return;
    }}
    const matchIds = new Set();
    d3Nodes.forEach(n => {{
      if (n.content.toLowerCase().includes(query) || n.type.toLowerCase().includes(query))
        matchIds.add(n.id);
    }});
    node.classed("highlight", d => matchIds.has(d.id))
        .classed("dim", d => !matchIds.has(d.id));
    link.classed("dim", d => !matchIds.has(d.source.id) || !matchIds.has(d.target.id));
  }});

  // Simulation
  const simulation = d3.forceSimulation(d3Nodes)
    .force("link", d3.forceLink(d3Links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => 6 + d.importance * 14 + 4));

  simulation.on("tick", () => {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y - (6 + d.importance * 14) - 4);
  }});

  function dragStart(event, d) {{
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }}
  function dragging(event, d) {{ d.fx = event.x; d.fy = event.y; }}
  function dragEnd(event, d) {{
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }}
}})();
</script>
</body>
</html>"""


def _escape_js(s: str) -> str:
    """Escape a string for safe embedding in a JavaScript string literal."""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
