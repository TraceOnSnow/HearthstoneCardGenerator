from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network


def load_graph_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        graph = json.load(f)
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError("Graph JSON must contain top-level keys: nodes, edges")
    return graph


def build_nx_graph(graph_data: dict[str, Any]) -> nx.DiGraph:
    g = nx.DiGraph()

    for node in graph_data.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        node_type = str(node.get("type", "other"))
        node_name = str(node.get("name", node_id))
        g.add_node(
            node_id,
            label=node_name,
            group=node_type,
            title=json.dumps(node, ensure_ascii=False, indent=2),
        )

    for edge in graph_data.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        predicate = str(edge.get("predicate", ""))
        g.add_edge(source, target, label=predicate, title=predicate)

    return g


def save_graph_html(
    graph_data: dict[str, Any],
    output_html: Path,
    *,
    height: str = "900px",
    width: str = "100%",
    directed: bool = True,
    physics_controls: bool = True,
) -> None:
    g = build_nx_graph(graph_data)
    net = Network(
        height=height,
        width=width,
        directed=directed,
        bgcolor="#ffffff",
        font_color="#1f2937",
    )
    net.from_nx(g)
    if physics_controls:
        net.show_buttons(filter_=["physics"])
    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_html))

