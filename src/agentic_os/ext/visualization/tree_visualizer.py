"""Interactive HTML visualization for thought trees using D3.js."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree


class ThoughtTreeVisualizer:
    """Generate interactive HTML visualization of thought trees using D3.js.

    Produces a self-contained HTML file with a top-down tree layout.
    Node size is proportional to ``log(visits + 1)`` and node colour
    transitions from red (score 0) to green (score 1). The best path
    from root to the best leaf is highlighted.
    """

    def to_html(
        self,
        tree: ThoughtTree,
        output_path: str,
        title: str = "Thought Tree",
    ) -> str:
        """Generate an interactive HTML file and write it to disk.

        Args:
            tree: The ``ThoughtTree`` to visualise.
            output_path: Destination file path for the HTML output.
            title: Page title shown in the browser tab.

        Returns:
            The generated HTML string.
        """
        data = self._tree_to_dict(tree)
        rendered = self._render(title, data)
        Path(output_path).write_text(rendered, encoding="utf-8")
        return rendered

    # ------------------------------------------------------------------
    # Data conversion
    # ------------------------------------------------------------------

    def _tree_to_dict(self, tree: ThoughtTree) -> dict[str, Any]:
        """Convert a ThoughtTree into a JSON-serialisable dict."""
        if tree.root is None:
            return {"thoughts": [], "best_path_ids": []}

        thoughts: list[dict[str, Any]] = []
        self._flatten(tree.root, thoughts)

        best_path = tree.get_best_path()
        best_path_ids = [id(n) for n in best_path]

        return {
            "thoughts": thoughts,
            "best_path_ids": best_path_ids,
        }

    def _flatten(
        self,
        node: ThoughtNode,
        acc: list[dict[str, Any]],
    ) -> None:
        """Recursively flatten the tree into a list of dicts."""
        node_id = id(node)
        parent_id = id(node.parent) if node.parent is not None else None
        acc.append({
            "id": node_id,
            "parent_id": parent_id,
            "thought": node.thought,
            "score": node.score,
            "visits": node.visits,
            "avg_score": node.avg_score,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
        })
        for child in node.children:
            self._flatten(child, acc)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self, title: str, data: dict[str, Any]) -> str:
        thoughts_json = json.dumps(data["thoughts"], ensure_ascii=False)
        best_path_json = json.dumps(data["best_path_ids"])

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
         background: #1a1a2e; color: #e0e0e0; overflow: auto; }}
  svg {{ display: block; margin: 0 auto; }}
  .link {{ fill: none; stroke: #555; stroke-width: 1.5px; stroke-opacity: 0.5; }}
  .link.best {{ stroke: #F1C40F; stroke-width: 3px; stroke-opacity: 1;
    filter: drop-shadow(0 0 4px rgba(241,196,15,0.4)); }}
  .node {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node:hover {{ stroke-width: 3px; }}
  .node-label {{ font-size: 11px; fill: #ccc; pointer-events: none; }}
  #tooltip {{ position: absolute; display: none; background: rgba(0,0,0,0.88);
    color: #fff; padding: 10px 14px; border-radius: 6px; font-size: 13px;
    pointer-events: none; z-index: 20; max-width: 350px; line-height: 1.5; }}
  #legend {{ position: absolute; bottom: 16px; left: 16px; z-index: 10;
    background: #2a2a3e; border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}
  #legend h4 {{ margin-bottom: 8px; font-size: 12px; color: #aaa; }}
  .legend-gradient {{ width: 160px; height: 12px; border-radius: 3px;
    background: linear-gradient(to right, #E74C3C, #F39C12, #2ECC71); }}
  .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: #aaa; margin-top: 2px; }}
</style>
</head>
<body>

<svg id="tree-svg"></svg>
<div id="tooltip"></div>

<div id="legend">
  <h4>Score Gradient</h4>
  <div class="legend-gradient"></div>
  <div class="legend-labels"><span>0.0</span><span>0.5</span><span>1.0</span></div>
  <div style="margin-top:8px; font-size:11px; color:#F1C40F;">&#9733; Best path highlighted</div>
</div>

<script>
(function() {{
  const thoughtsData = {thoughts_json};
  const bestPathIds = new Set({best_path_json});

  if (thoughtsData.length === 0) {{
    d3.select("body").append("div")
      .style("text-align", "center").style("margin-top", "100px")
      .style("font-size", "18px").style("color", "#888")
      .text("Empty tree -- nothing to visualise.");
    return;
  }}

  // Build hierarchical data for d3.tree
  const nodeMap = new Map();
  thoughtsData.forEach(t => nodeMap.set(t.id, {{...t, children: []}}));
  let root = null;
  thoughtsData.forEach(t => {{
    const cur = nodeMap.get(t.id);
    if (t.parent_id === null) {{
      root = cur;
    }} else {{
      const parent = nodeMap.get(t.parent_id);
      if (parent) parent.children.push(cur);
    }}
  }});
  if (!root) return;

  const treeLayout = d3.tree().nodeSize([30, 180]);
  const d3Root = d3.hierarchy(root);
  treeLayout(d3Root);

  const nodes = d3Root.descendants();
  const links = d3Root.links();

  // Bounding box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  nodes.forEach(n => {{
    minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
    minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
  }});
  const pad = 80;
  const svgW = (maxY - minY) + pad * 2;
  const svgH = (maxX - minX) + pad * 2;

  const svg = d3.select("#tree-svg")
    .attr("width", svgW)
    .attr("height", svgH);

  const g = svg.append("g")
    .attr("transform", "translate(" + (-minY + pad) + "," + (-minX + pad) + ")");

  // Score colour: red 0 -> green 1
  function scoreColor(s) {{
    s = Math.max(0, Math.min(1, s));
    if (s < 0.5) {{
      const t = s * 2;
      const r = Math.round(231 + (243 - 231) * t);
      const g2 = Math.round(76 + (156 - 76) * t);
      const b = Math.round(60 + (18 - 60) * t);
      return "rgb(" + r + "," + g2 + "," + b + ")";
    }} else {{
      const t = (s - 0.5) * 2;
      const r = Math.round(243 + (46 - 243) * t);
      const g2 = Math.round(156 + (204 - 156) * t);
      const b = Math.round(18 + (113 - 18) * t);
      return "rgb(" + r + "," + g2 + "," + b + ")";
    }}
  }}

  // Node radius: proportional to log(visits + 1)
  function nodeRadius(d) {{
    const v = d.data.visits || 0;
    return 6 + Math.log(v + 1) * 5;
  }}

  // Best path links
  function isBestLink(d) {{
    return bestPathIds.has(d.source.data.id) && bestPathIds.has(d.target.data.id);
  }}

  // Links
  g.selectAll(".link")
    .data(links).join("path")
    .attr("class", d => "link" + (isBestLink(d) ? " best" : ""))
    .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

  // Nodes
  const node = g.selectAll(".node")
    .data(nodes).join("circle")
    .attr("class", "node")
    .attr("cx", d => d.y)
    .attr("cy", d => d.x)
    .attr("r", nodeRadius)
    .attr("fill", d => scoreColor(d.data.avg_score))
    .attr("stroke", d => bestPathIds.has(d.data.id) ? "#F1C40F" : "#fff")
    .attr("stroke-width", d => bestPathIds.has(d.data.id) ? 3 : 1.5);

  // Labels
  g.selectAll(".node-label")
    .data(nodes).join("text")
    .attr("class", "node-label")
    .attr("x", d => d.y)
    .attr("y", d => d.x - nodeRadius(d) - 4)
    .attr("text-anchor", "middle")
    .text(d => {{
      const t = d.data.thought;
      return t.length > 22 ? t.substring(0, 22) + "..." : t;
    }});

  // Tooltip
  const tooltip = d3.select("#tooltip");
  node.on("mouseover", function(event, d) {{
    let html = "<strong>" + d.data.thought + "</strong><br/>";
    html += "Score: " + d.data.score.toFixed(3) + "<br/>";
    html += "Avg Score: " + d.data.avg_score.toFixed(3) + "<br/>";
    html += "Visits: " + d.data.visits + "<br/>";
    html += "Depth: " + d.data.depth;
    if (d.data.is_leaf) html += "<br/><em>leaf</em>";
    if (bestPathIds.has(d.data.id)) html += "<br/><em style='color:#F1C40F'>best path</em>";
    tooltip.style("display", "block").html(html);
  }})
  .on("mousemove", function(event) {{
    tooltip.style("left", (event.pageX + 14) + "px")
           .style("top", (event.pageY - 20) + "px");
  }})
  .on("mouseout", function() {{
    tooltip.style("display", "none");
  }});
}})();
</script>
</body>
</html>"""
