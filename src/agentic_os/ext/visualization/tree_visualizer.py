"""Interactive HTML visualization for thought trees using D3.js."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from agentic_os.core.tree.thought_node import ThoughtNode
from agentic_os.core.tree.thought_tree import ThoughtTree


class ThoughtTreeVisualizer:
    """Generate interactive HTML visualization of thought trees using D3.js."""

    def to_html(
        self,
        tree: ThoughtTree,
        output_path: str,
        title: str = "Thought Tree",
    ) -> str:
        """Generate an interactive HTML file and write it to disk."""
        data = self._tree_to_dict(tree)
        rendered = self._render(title, data)
        Path(output_path).write_text(rendered, encoding="utf-8")
        return rendered

    def _tree_to_dict(self, tree: ThoughtTree) -> dict[str, Any]:
        if tree.root is None:
            return {"thoughts": [], "best_path_ids": []}

        thoughts: list[dict[str, Any]] = []
        self._flatten(tree.root, thoughts)

        best_path = tree.get_best_path()
        best_path_ids = [id(n) for n in best_path]

        return {"thoughts": thoughts, "best_path_ids": best_path_ids}

    def _flatten(self, node: ThoughtNode, acc: list[dict[str, Any]]) -> None:
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

    def _render(self, title: str, data: dict[str, Any]) -> str:
        thoughts_json = json.dumps(data["thoughts"], ensure_ascii=False)
        best_path_json = json.dumps(data["best_path_ids"])
        stats = {
            "nodes": len(data["thoughts"]),
            "depth": max((t["depth"] for t in data["thoughts"]), default=0),
            "best_path": len(data["best_path_ids"]),
        }
        stats_json = json.dumps(stats)
        escaped_title = html.escape(title)

        return f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escaped_title}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif;
    background: #0f0f23; color: #e2e8f0; overflow: hidden; height: 100vh; }}

  #header {{ position: absolute; top: 0; left: 0; right: 0; height: 56px;
    background: rgba(15,15,35,0.85); backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06); z-index: 20;
    display: flex; align-items: center; padding: 0 24px; gap: 16px; }}
  #header h1 {{ font-size: 16px; font-weight: 600; color: #f1f5f9; }}
  #header .stats {{ font-size: 12px; color: #64748b; }}
  #header .stats span {{ color: #94a3b8; font-weight: 500; }}

  #canvas {{ width: 100vw; height: calc(100vh - 56px); margin-top: 56px; overflow: hidden;
    cursor: grab; }}
  #canvas:active {{ cursor: grabbing; }}
  #canvas svg {{ width: 100%; height: 100%; }}

  .link {{ fill: none; stroke: #334155; stroke-width: 1.5; }}
  .link.best {{ stroke: #fbbf24; stroke-width: 3; filter: drop-shadow(0 0 6px rgba(251,191,36,0.3)); }}
  .link-glow {{ fill: none; stroke-width: 6; opacity: 0; }}
  .link.best-glow {{ stroke: #fbbf24; opacity: 0.15; }}

  .node-group {{ cursor: pointer; }}
  .node-group:hover .outer-ring {{ opacity: 0.5; }}
  .node-outer {{ opacity: 0.2; }}
  .node-inner {{ stroke-width: 0; }}
  .outer-ring {{ fill: none; stroke-width: 2; opacity: 0; transition: opacity 0.3s; }}
  .node-group:hover .outer-ring {{ opacity: 0.6; }}
  .node-label {{ font-size: 11px; fill: #94a3b8; pointer-events: none;
    font-weight: 400; }}
  .node-label-bg {{ fill: rgba(15,15,35,0.75); rx: 3; ry: 3; }}
  .node-score {{ font-size: 8px; fill: rgba(255,255,255,0.5); pointer-events: none;
    text-anchor: middle; dominant-baseline: central; font-weight: 600; }}

  .node-group.selected .outer-ring {{ opacity: 1; stroke-width: 3; }}
  .node-group.best-path .node-outer {{ opacity: 0.35; }}
  .node-group.best-path .outer-ring {{ opacity: 0.6; stroke: #fbbf24; }}

  /* Detail Panel */
  #detail {{ position: absolute; top: 72px; right: 20px; width: 340px;
    max-height: calc(100vh - 96px); overflow-y: auto;
    background: rgba(30,30,50,0.92); backdrop-filter: blur(16px);
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
    padding: 24px; display: none; z-index: 15;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5); }}
  #detail .thought-text {{ font-size: 15px; line-height: 1.6; color: #f1f5f9;
    margin-bottom: 16px; font-weight: 500; }}
  #detail .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  #detail .meta-item {{ background: rgba(255,255,255,0.04); border-radius: 8px; padding: 10px 12px; }}
  #detail .meta-item .label {{ font-size: 10px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px; }}
  #detail .meta-item .value {{ font-size: 14px; color: #e2e8f0; font-weight: 500; }}
  #detail .score-bar {{ height: 4px; border-radius: 2px; background: rgba(255,255,255,0.1);
    margin-top: 6px; overflow: hidden; }}
  #detail .score-fill {{ height: 100%; border-radius: 2px; }}
  #detail .close {{ position: absolute; top: 14px; right: 16px; background: none;
    border: none; color: #64748b; font-size: 18px; cursor: pointer; width: 28px;
    height: 28px; display: flex; align-items: center; justify-content: center;
    border-radius: 6px; transition: all 0.2s; }}
  #detail .close:hover {{ color: #f1f5f9; background: rgba(255,255,255,0.1); }}

  /* Legend */
  #legend {{ position: absolute; bottom: 20px; left: 20px; z-index: 15;
    background: rgba(30,30,50,0.9); backdrop-filter: blur(12px);
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);
    padding: 14px 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }}
  #legend h4 {{ font-size: 10px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 8px; }}
  .leg-gradient {{ width: 140px; height: 10px; border-radius: 3px;
    background: linear-gradient(to right, #ef4444, #f59e0b, #22c55e); }}
  .leg-labels {{ display: flex; justify-content: space-between; font-size: 10px;
    color: #64748b; margin-top: 3px; }}
  .leg-best {{ display: flex; align-items: center; gap: 6px; margin-top: 10px;
    padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 11px; color: #fbbf24; }}
  .leg-line {{ width: 20px; height: 3px; background: #fbbf24; border-radius: 2px; }}

  @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(8px); }}
    to {{ opacity:1; transform: translateY(0); }} }}
  #detail {{ animation: fadeIn 0.2s ease; }}
</style>
</head>
<body>

<div id="header">
  <h1>{escaped_title}</h1>
  <div class="stats" id="stats-text"></div>
</div>

<div id="canvas"><svg id="tree-svg"></svg></div>

<div id="detail">
  <button class="close" id="close-detail">&times;</button>
  <div id="detail-body"></div>
</div>

<div id="legend">
  <h4>Score</h4>
  <div class="leg-gradient"></div>
  <div class="leg-labels"><span>0.0</span><span>0.5</span><span>1.0</span></div>
  <div class="leg-best"><div class="leg-line"></div> Best Path</div>
</div>

<script>
(function() {{
  const thoughts = {thoughts_json};
  const bestIds = new Set({best_path_json});
  const stats = {stats_json};

  d3.select("#stats-text").html(
    '<span>'+stats.nodes+'</span> nodes &middot; depth <span>'+stats.depth+
    '</span> &middot; best path <span>'+stats.best_path+'</span>');

  if (!thoughts.length) {{
    d3.select("#canvas").append("div").style("text-align","center")
      .style("margin-top","100px").style("color","#475569")
      .text("Empty tree");
    return;
  }}

  const nodeMap = new Map();
  thoughts.forEach(t => nodeMap.set(t.id, {{...t, children:[]}}));
  let root = null;
  thoughts.forEach(t => {{
    const cur = nodeMap.get(t.id);
    if (t.parent_id === null) root = cur;
    else {{ const p = nodeMap.get(t.parent_id); if (p) p.children.push(cur); }}
  }});
  if (!root) return;

  const treeLayout = d3.tree().nodeSize([48, 220]);
  const d3Root = d3.hierarchy(root);
  treeLayout(d3Root);

  const nodes = d3Root.descendants();
  const links = d3Root.links();

  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  nodes.forEach(n => {{
    minX=Math.min(minX,n.x); maxX=Math.max(maxX,n.x);
    minY=Math.min(minY,n.y); maxY=Math.max(maxY,n.y);
  }});
  const pad = 100;
  const svgW = (maxY-minY)+pad*2;
  const svgH = (maxX-minX)+pad*2;

  const svg = d3.select("#tree-svg").attr("width",svgW).attr("height",svgH);
  const g = svg.append("g").attr("transform",
    "translate("+(-minY+pad)+","+(-minX+pad)+")");

  // Zoom/pan
  const canvas = d3.select("#canvas");
  canvas.call(d3.zoom().scaleExtent([0.2,4]).on("zoom", e =>
    g.attr("transform", e.transform)));

  function scoreColor(s) {{
    s = Math.max(0, Math.min(1, s));
    if (s < 0.5) {{
      const t = s*2;
      return "rgb("+Math.round(239+(245-239)*t)+","+Math.round(68+(158-68)*t)+","+Math.round(68+(11-68)*t)+")";
    }}
    const t = (s-0.5)*2;
    return "rgb("+Math.round(245+(34-245)*t)+","+Math.round(158+(197-158)*t)+","+Math.round(11+(94-11)*t)+")";
  }}

  function nRadius(d) {{ return 8 + Math.log((d.data.visits||0)+1)*4; }}
  function isBest(d) {{ return bestIds.has(d.data.id); }}
  function isBestLink(d) {{ return bestIds.has(d.source.data.id) && bestIds.has(d.target.data.id); }}

  // Link glow for best path
  g.selectAll(".link-glow").data(links).join("path")
    .attr("class", d => "link-glow" + (isBestLink(d) ? " best-glow" : ""))
    .attr("stroke", d => isBestLink(d) ? "#fbbf24" : "transparent")
    .attr("d", d3.linkHorizontal().x(d=>d.y).y(d=>d.x));

  // Links
  g.selectAll(".link").data(links).join("path")
    .attr("class", d => "link" + (isBestLink(d) ? " best" : ""))
    .attr("d", d3.linkHorizontal().x(d=>d.y).y(d=>d.x));

  // Node groups
  const ng = g.selectAll(".node-group").data(nodes).join("g")
    .attr("class", d => "node-group" + (isBest(d) ? " best-path" : ""))
    .attr("transform", d => "translate("+d.y+","+d.x+")");

  ng.append("circle").attr("class","node-outer")
    .attr("r", d => nRadius(d)+6)
    .attr("fill", d => scoreColor(d.data.avg_score));
  ng.append("circle").attr("class","outer-ring")
    .attr("r", d => nRadius(d)+3)
    .attr("stroke", d => isBest(d) ? "#fbbf24" : scoreColor(d.data.avg_score));
  ng.append("circle").attr("class","node-inner")
    .attr("r", nRadius)
    .attr("fill", d => scoreColor(d.data.avg_score));
  ng.append("text").attr("class","node-score")
    .attr("dy","0.1em")
    .text(d => d.data.avg_score > 0 ? (d.data.avg_score*100).toFixed(0) : "");

  // Labels
  const labels = g.append("g");
  nodes.forEach(d => {{
    const txt = d.data.thought;
    const short = txt.length > 24 ? txt.substring(0,24)+"..." : txt;
    const tw = short.length*6.2 + 12;
    labels.append("rect").attr("class","node-label-bg")
      .attr("width",tw).attr("height",16).attr("rx",3).attr("ry",3)
      .attr("x",d.y-tw/2).attr("y",d.x+nRadius(d)+6);
    labels.append("text").attr("class","node-label")
      .attr("x",d.y).attr("y",d.x+nRadius(d)+18)
      .attr("text-anchor","middle").text(short);
  }});

  // Detail panel
  ng.on("click", function(e, d) {{
    e.stopPropagation();
    ng.classed("selected", false);
    d3.select(this).classed("selected", true);

    const sc = d.data.avg_score;
    const c = scoreColor(sc);
    let body = '<div class="thought-text">'+d.data.thought+'</div>';
    body += '<div class="meta-grid">';
    body += '<div class="meta-item"><div class="label">Score</div><div class="value">'+
      d.data.score.toFixed(3)+'</div><div class="score-bar"><div class="score-fill" style="width:'+
      (sc*100)+'%;background:'+c+'"></div></div></div>';
    body += '<div class="meta-item"><div class="label">Avg Score</div><div class="value">'+
      sc.toFixed(3)+'</div><div class="score-bar"><div class="score-fill" style="width:'+
      (sc*100)+'%;background:'+c+'"></div></div></div>';
    body += '<div class="meta-item"><div class="label">Visits</div><div class="value">'+
      d.data.visits+'</div></div>';
    body += '<div class="meta-item"><div class="label">Depth</div><div class="value">'+
      d.data.depth+'</div></div>';
    body += '</div>';
    if (d.data.is_leaf) body += '<div style="margin-top:12px;font-size:11px;color:#64748b">&#9671; Leaf node</div>';
    if (isBest(d)) body += '<div style="margin-top:8px;font-size:11px;color:#fbbf24">&#9733; Best path node</div>';

    d3.select("#detail-body").html(body);
    d3.select("#detail").style("display","block");
  }});

  d3.select("#close-detail").on("click", () => {{
    d3.select("#detail").style("display","none");
    ng.classed("selected", false);
  }});

  // Center view
  canvas.call(d3.zoom().scaleExtent([0.2,4]).on("zoom", e =>
    g.attr("transform", e.transform))
    .transform, d3.zoomIdentity
    .translate(canvas.node().clientWidth/2, canvas.node().clientHeight/2)
    .translate(-(-minY+pad)-svgW/2, -(-minX+pad)-svgH/2));
}})();
</script>
</body>
</html>"""
