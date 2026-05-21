"""Interactive HTML visualization for knowledge graphs using D3.js."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, ClassVar

from agentic_os.core.graph.knowledge_graph import KnowledgeGraph


class KnowledgeGraphVisualizer:
    """Generate interactive HTML visualization of knowledge graphs using D3.js."""

    _NODE_COLORS: ClassVar[dict[str, str]] = {
        "episode": "#60A5FA",
        "fact": "#34D399",
        "reflection": "#FBBF24",
        "goal": "#F87171",
    }

    _NODE_ICONS: ClassVar[dict[str, str]] = {
        "episode": "E",
        "fact": "F",
        "reflection": "R",
        "goal": "G",
    }

    _EDGE_COLORS: ClassVar[dict[str, str]] = {
        "causal": "#FB923C",
        "temporal": "#38BDF8",
        "associative": "#A78BFA",
        "derived_from": "#E879F9",
        "supports": "#4ADE80",
        "contradicts": "#FB7185",
    }

    _EDGE_LABELS: ClassVar[dict[str, str]] = {
        "causal": "因果",
        "temporal": "时序",
        "associative": "关联",
        "derived_from": "衍生",
        "supports": "支持",
        "contradicts": "矛盾",
    }

    def to_html(
        self,
        graph: KnowledgeGraph,
        output_path: str,
        title: str = "Knowledge Graph",
    ) -> str:
        """Generate an interactive HTML file and write it to disk."""
        data = graph.to_dict()
        rendered = self._render(title, data)
        Path(output_path).write_text(rendered, encoding="utf-8")
        return rendered

    def _render(self, title: str, data: dict[str, Any]) -> str:
        nodes_json = json.dumps(data.get("nodes", []), ensure_ascii=False)
        edges_json = json.dumps(data.get("edges", []), ensure_ascii=False)
        colors_json = json.dumps(self._NODE_COLORS)
        icons_json = json.dumps(self._NODE_ICONS)
        edge_colors_json = json.dumps(self._EDGE_COLORS)
        edge_labels_json = json.dumps(self._EDGE_LABELS)
        stats_json = json.dumps({
            "nodes": len(data.get("nodes", [])),
            "edges": len(data.get("edges", [])),
        })

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
  body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
         background: #0f0f23; color: #e2e8f0; overflow: hidden; height: 100vh; }}

  /* Header */
  #header {{ position: absolute; top: 0; left: 0; right: 0; height: 56px;
    background: rgba(15,15,35,0.85); backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06); z-index: 20;
    display: flex; align-items: center; padding: 0 24px; gap: 16px; }}
  #header h1 {{ font-size: 16px; font-weight: 600; color: #f1f5f9; white-space: nowrap; }}
  #header .stats {{ font-size: 12px; color: #64748b; margin-left: 8px; }}
  #header .stats span {{ color: #94a3b8; font-weight: 500; }}

  /* Search */
  #search-wrap {{ position: relative; margin-left: auto; }}
  #search-wrap svg {{ position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
    color: #64748b; }}
  #search-input {{ width: 220px; padding: 7px 12px 7px 32px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05);
    color: #e2e8f0; font-size: 13px; outline: none; transition: all 0.2s; }}
  #search-input:focus {{ border-color: rgba(96,165,250,0.5); background: rgba(255,255,255,0.08);
    box-shadow: 0 0 0 3px rgba(96,165,250,0.1); }}
  #search-input::placeholder {{ color: #475569; }}

  /* SVG */
  svg#graph {{ width: 100vw; height: calc(100vh - 56px); margin-top: 56px; cursor: grab; }}
  svg#graph:active {{ cursor: grabbing; }}

  /* Defs */
  .link {{ fill: none; stroke-linecap: round; }}
  .link-glow {{ fill: none; stroke-linecap: round; opacity: 0.15; }}

  /* Node groups */
  .node-group {{ cursor: pointer; }}
  .node-group:hover .node-ring {{ opacity: 1; }}
  .node-glow {{ opacity: 0; transition: opacity 0.3s; }}
  .node-group:hover .node-glow {{ opacity: 0.3; }}
  .node-ring {{ fill: none; stroke-width: 2; opacity: 0.6; transition: opacity 0.3s; }}
  .node-core {{ stroke-width: 0; transition: r 0.2s; }}
  .node-group:hover .node-core {{ filter: brightness(1.2); }}
  .node-icon {{ font-size: 9px; font-weight: 700; fill: rgba(0,0,0,0.6);
    text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
  .node-label {{ font-size: 11px; fill: #94a3b8; pointer-events: none;
    text-anchor: middle; font-weight: 400; }}
  .node-label-bg {{ fill: rgba(15,15,35,0.7); rx: 3; ry: 3; }}
  .link-label {{ font-size: 9px; fill: #64748b; pointer-events: none;
    text-anchor: middle; font-weight: 500; }}
  .link-label-bg {{ fill: rgba(15,15,35,0.8); rx: 3; ry: 3; }}

  .node-group.highlight .node-ring {{ opacity: 1; stroke-width: 3; }}
  .node-group.dim {{ opacity: 0.08; }}
  .link.dim {{ opacity: 0.03 !important; }}
  .link-glow.dim {{ opacity: 0 !important; }}

  /* Detail Panel */
  #detail {{ position: absolute; top: 72px; right: 20px; width: 360px;
    max-height: calc(100vh - 100px);
    overflow-y: auto; overflow-x: hidden;
    background: rgba(30,30,50,0.95); backdrop-filter: blur(16px);
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
    padding: 24px; display: none; z-index: 15;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05); }}
  #detail::-webkit-scrollbar {{ width: 4px; }}
  #detail::-webkit-scrollbar-track {{ background: transparent; }}
  #detail::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 2px; }}
  #detail::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.25); }}
  #detail .type-badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; }}
  #detail .content {{ font-size: 15px; line-height: 1.6; color: #f1f5f9; margin-bottom: 16px;
    font-weight: 500; word-break: break-all; }}
  #detail .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  #detail .meta-item {{ background: rgba(255,255,255,0.04); border-radius: 8px;
    padding: 10px 12px; }}
  #detail .meta-item .label {{ font-size: 10px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px; }}
  #detail .meta-item .value {{ font-size: 13px; color: #e2e8f0; font-weight: 500;
    word-break: break-all; }}
  #detail .close {{ position: sticky; top: 0; float: right; background: rgba(30,30,50,0.9);
    border: none; color: #64748b; font-size: 18px; cursor: pointer; width: 28px;
    height: 28px; display: flex; align-items: center; justify-content: center;
    border-radius: 6px; transition: all 0.2s; z-index: 1; }}
  #detail .close:hover {{ color: #f1f5f9; background: rgba(255,255,255,0.15); }}
  #detail .connections {{ margin-top: 16px; clear: both; }}
  #detail .connections h4 {{ font-size: 11px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px; }}
  #detail .conn-item {{ display: flex; align-items: flex-start; gap: 8px; padding: 6px 0;
    font-size: 12px; color: #94a3b8; line-height: 1.4; }}
  #detail .conn-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }}
  #detail .conn-text {{ flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }}

  /* Legend */
  #legend {{ position: absolute; bottom: 20px; left: 20px; z-index: 15;
    background: rgba(30,30,50,0.9); backdrop-filter: blur(12px);
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);
    padding: 14px 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }}
  #legend h4 {{ font-size: 10px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 8px; }}
  .leg-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px;
    color: #94a3b8; font-weight: 500; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .leg-sep {{ width: 1px; height: 14px; background: rgba(255,255,255,0.1); margin: 0 2px; }}
  #legend-edges {{ margin-top: 10px; padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.06); }}

  /* Tooltip */
  #tooltip {{ position: absolute; display: none; background: rgba(20,20,40,0.95);
    backdrop-filter: blur(8px); color: #e2e8f0; padding: 8px 12px;
    border-radius: 8px; font-size: 12px; pointer-events: none; z-index: 30;
    max-width: 300px; border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}

  /* Animations */
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }} }}
  #detail {{ animation: fadeIn 0.2s ease; }}
</style>
</head>
<body>

<div id="header">
  <h1>{escaped_title}</h1>
  <div class="stats" id="stats-text"></div>
  <div id="search-wrap">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
    <input type="text" id="search-input" placeholder="搜索节点..." />
  </div>
</div>

<svg id="graph"><defs>
  <filter id="glow"><feGaussianBlur stdDeviation="4" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs></svg>
<div id="tooltip"></div>

<div id="detail">
  <button class="close" id="close-detail">&times;</button>
  <div id="detail-body"></div>
</div>

<div id="legend">
  <h4>节点类型</h4>
  <div class="leg-row" id="leg-nodes"></div>
  <div id="legend-edges">
    <h4>边类型</h4>
    <div class="leg-row" id="leg-edges"></div>
  </div>
</div>

<script>
(function() {{
  const nodesData = {nodes_json};
  const edgesData = {edges_json};
  const nodeColors = {colors_json};
  const nodeIcons = {icons_json};
  const edgeColors = {edge_colors_json};
  const edgeLabels = {edge_labels_json};
  const stats = {stats_json};

  d3.select("#stats-text").html(
    '<span>' + stats.nodes + '</span> nodes &middot; <span>' + stats.edges + '</span> edges');

  // Legend
  const legNodes = d3.select("#leg-nodes");
  Object.entries(nodeColors).forEach(([t, c]) => {{
    legNodes.append("div").attr("class","leg-item")
      .html('<div class="leg-dot" style="background:'+c+'"></div>'+t);
  }});
  const legEdges = d3.select("#leg-edges");
  Object.entries(edgeColors).forEach(([t, c]) => {{
    legEdges.append("div").attr("class","leg-item")
      .html('<div class="leg-dot" style="background:'+c+'"></div>'+(edgeLabels[t]||t));
  }});

  const svg = d3.select("#graph");
  const width = window.innerWidth;
  const height = window.innerHeight - 56;
  svg.attr("viewBox", [0, 0, width, height]);

  const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", e => g.attr("transform", e.transform)));

  const nodeMap = new Map();
  nodesData.forEach(n => nodeMap.set(n.id, n));
  const d3Nodes = nodesData.map(n => ({{...n}}));
  const d3Links = edgesData
    .filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
    .map(e => ({{source: e.source_id, target: e.target_id, type: e.type,
                 weight: e.weight, metadata: e.metadata}}));

  // Build adjacency for detail panel
  const outEdges = new Map(), inEdges = new Map();
  d3Links.forEach(e => {{
    (outEdges.get(e.source) || outEdges.set(e.source,[]).get(e.source)).push(e);
    (inEdges.get(e.target) || inEdges.set(e.target,[]).get(e.target)).push(e);
  }});

  // Edge glow layer
  const glowLayer = g.append("g");
  const linkGlow = glowLayer.selectAll("path").data(d3Links).join("path")
    .attr("class","link-glow")
    .attr("stroke", d => edgeColors[d.type] || "#666")
    .attr("stroke-width", d => Math.max(6, d.weight * 12));

  // Edge layer
  const linkLayer = g.append("g");
  const link = linkLayer.selectAll("path").data(d3Links).join("path")
    .attr("class","link")
    .attr("stroke", d => edgeColors[d.type] || "#666")
    .attr("stroke-width", d => Math.max(1.5, d.weight * 4))
    .attr("stroke-opacity", 0.5);

  // Edge labels (only show for small graphs)
  const showEdgeLabels = d3Links.length <= 20;
  if (showEdgeLabels) {{
    const linkLabelGroup = g.append("g");
    d3Links.forEach(d => {{
      const lbl = edgeLabels[d.type] || d.type;
      linkLabelGroup.append("rect").attr("class","link-label-bg")
        .attr("width", lbl.length * 8 + 8).attr("height", 14)
        .attr("rx", 3).datum(d);
      linkLabelGroup.append("text").attr("class","link-label")
        .text(lbl).datum(d);
    }});
  }}

  // Node layer
  const nodeLayer = g.append("g");
  const nodeGroups = nodeLayer.selectAll(".node-group").data(d3Nodes).join("g")
    .attr("class","node-group")
    .call(d3.drag().on("start",dragStart).on("drag",dragging).on("end",dragEnd));

  const nodeRadius = d => 8 + d.importance * 16;
  const color = d => nodeColors[d.type] || "#888";

  nodeGroups.append("circle").attr("class","node-glow")
    .attr("r", d => nodeRadius(d) + 10).attr("fill", d => color(d));
  nodeGroups.append("circle").attr("class","node-ring")
    .attr("r", d => nodeRadius(d) + 3).attr("stroke", d => color(d));
  nodeGroups.append("circle").attr("class","node-core")
    .attr("r", nodeRadius).attr("fill", d => color(d));
  nodeGroups.append("text").attr("class","node-icon")
    .text(d => nodeIcons[d.type] || "?");

  // Labels with background
  const labelGroup = g.append("g");
  d3Nodes.forEach(d => {{
    const txt = d.content.length > 20 ? d.content.substring(0,20)+"..." : d.content;
    const tw = txt.length * 6.5 + 12;
    labelGroup.append("rect").attr("class","node-label-bg")
      .attr("width", tw).attr("height", 16)
      .attr("rx",3).attr("ry",3).datum(d);
    labelGroup.append("text").attr("class","node-label").text(txt).datum(d);
  }});

  // Tooltip
  const tooltip = d3.select("#tooltip");
  nodeGroups.on("mouseover", function(e, d) {{
    tooltip.style("display","block")
      .html("<strong style='color:" + color(d) + "'>" + d.type +
            "</strong><br/><span style='color:#cbd5e1'>" + d.content + "</span>");
  }}).on("mousemove", function(e) {{
    tooltip.style("left", (e.pageX+14)+"px").style("top", (e.pageY-24)+"px");
  }}).on("mouseout", function() {{ tooltip.style("display","none"); }});

  // Detail panel
  nodeGroups.on("click", function(e, d) {{
    e.stopPropagation();
    let body = '<div class="type-badge" style="background:'+color(d)+'22;color:'+color(d)+
      ';border:1px solid '+color(d)+'44">'+d.type+'</div>';
    body += '<div class="content">' + d.content + '</div>';
    body += '<div class="meta-grid">';
    body += '<div class="meta-item"><div class="label">Importance</div><div class="value">' +
      d.importance.toFixed(2) + '</div></div>';
    body += '<div class="meta-item"><div class="label">Access Count</div><div class="value">' +
      d.access_count + '</div></div>';
    body += '<div class="meta-item"><div class="label">Created</div><div class="value">' +
      (d.created_at || '-') + '</div></div>';
    body += '<div class="meta-item"><div class="label">Updated</div><div class="value">' +
      (d.updated_at || '-') + '</div></div>';
    body += '</div>';

    const outE = (outEdges.get(d.id) || []);
    const inE = (inEdges.get(d.id) || []);
    if (outE.length + inE.length > 0) {{
      body += '<div class="connections"><h4>Connections</h4>';
      outE.forEach(e => {{
        const tn = nodeMap.get(e.target);
        body += '<div class="conn-item"><div class="conn-dot" style="background:'+edgeColors[e.type]+'"></div>'+
          '<div class="conn-text">&rarr; '+(tn ? tn.content.substring(0,40) : e.target)+
          ' <span style="color:#475569">('+(edgeLabels[e.type]||e.type)+')</span></div></div>';
      }});
      inE.forEach(e => {{
        const sn = nodeMap.get(e.source);
        body += '<div class="conn-item"><div class="conn-dot" style="background:'+edgeColors[e.type]+'"></div>'+
          '<div class="conn-text">&larr; '+(sn ? sn.content.substring(0,40) : e.source)+
          ' <span style="color:#475569">('+(edgeLabels[e.type]||e.type)+')</span></div></div>';
      }});
      body += '</div>';
    }}

    d3.select("#detail-body").html(body);
    d3.select("#detail").style("display","block");
  }});

  d3.select("#close-detail").on("click", () => d3.select("#detail").style("display","none"));
  svg.on("click", () => d3.select("#detail").style("display","none"));

  // Search
  d3.select("#search-input").on("input", function() {{
    const q = this.value.toLowerCase();
    if (!q) {{
      nodeGroups.classed("highlight",false).classed("dim",false);
      link.classed("dim",false); linkGlow.classed("dim",false);
      return;
    }}
    const matchIds = new Set();
    d3Nodes.forEach(n => {{
      if (n.content.toLowerCase().includes(q) || n.type.toLowerCase().includes(q))
        matchIds.add(n.id);
    }});
    nodeGroups.classed("highlight", d => matchIds.has(d.id))
               .classed("dim", d => !matchIds.has(d.id));
    link.classed("dim", d => !matchIds.has(d.source.id) || !matchIds.has(d.target.id));
    linkGlow.classed("dim", d => !matchIds.has(d.source.id) || !matchIds.has(d.target.id));
  }});

  // Simulation
  const sim = d3.forceSimulation(d3Nodes)
    .force("link", d3.forceLink(d3Links).id(d => d.id).distance(140).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-500))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(d => nodeRadius(d)+8))
    .force("x", d3.forceX(width/2).strength(0.03))
    .force("y", d3.forceY(height/2).strength(0.03));

  sim.on("tick", () => {{
    // Curved edges
    link.attr("d", d => {{
      const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
      const dr = Math.sqrt(dx*dx+dy*dy) * 0.8;
      return "M"+d.source.x+","+d.source.y+"A"+dr+","+dr+" 0 0,1 "+d.target.x+","+d.target.y;
    }});
    linkGlow.attr("d", link.attr("d"));

    if (showEdgeLabels) {{
      labelGroup.selectAll(".link-label-bg").each(function(d) {{
        const mx = (d.source.x+d.target.x)/2, my = (d.source.y+d.target.y)/2;
        d3.select(this).attr("x", mx - d3.select(this).attr("width")/2).attr("y", my-12);
      }});
      labelGroup.selectAll(".link-label").each(function(d) {{
        const mx = (d.source.x+d.target.x)/2, my = (d.source.y+d.target.y)/2;
        d3.select(this).attr("x", mx).attr("y", my);
      }});
    }}

    nodeGroups.attr("transform", d => "translate("+d.x+","+d.y+")");
    labelGroup.selectAll(".node-label-bg").each(function(d) {{
      const r = nodeRadius(d), tw = d3.select(this).attr("width");
      d3.select(this).attr("x", d.x-tw/2).attr("y", d.y-r-20);
    }});
    labelGroup.selectAll(".node-label").each(function(d) {{
      const r = nodeRadius(d);
      d3.select(this).attr("x", d.x).attr("y", d.y-r-10);
    }});
  }});

  function dragStart(e, d) {{ if (!e.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y; }}
  function dragging(e, d) {{ d.fx = e.x; d.fy = e.y; }}
  function dragEnd(e, d) {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}
}})();
</script>
</body>
</html>"""
