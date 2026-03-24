import argparse
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
        attrs = node.get("attributes", {})

        tooltip = {
            "id": node_id,
            "type": node_type,
            "name": node_name,
            "attributes": attrs,
        }

        g.add_node(
            node_id,
            label=node_name,
            group=node_type,
            title=json.dumps(tooltip, ensure_ascii=False, indent=2),
        )

    for edge in graph_data.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue

        predicate = str(edge.get("predicate", ""))
        g.add_edge(source, target, label=predicate, title=predicate)

    return g


def save_html(
    g: nx.DiGraph,
    output_html: Path,
    *,
    height: str,
    width: str,
    directed: bool,
    enable_physics_controls: bool,
) -> None:
    net = Network(
        height=height,
        width=width,
        directed=directed,
        bgcolor="#ffffff",
        font_color="#1f2937",
    )
    net.from_nx(g)

    if enable_physics_controls:
        net.show_buttons(filter_=["physics"])

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_html))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize graph JSON to interactive HTML.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mvp_kg_demo/graph.json"),
        help="Path to graph JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mvp_kg_demo/graph_vis.html"),
        help="Path to output HTML file.",
    )
    parser.add_argument("--height", type=str, default="900px", help="Canvas height, e.g. 900px.")
    parser.add_argument("--width", type=str, default="100%", help="Canvas width, e.g. 100%.")
    parser.add_argument("--undirected", action="store_true", help="Render as undirected graph.")
    parser.add_argument(
        "--no-physics-controls",
        action="store_true",
        help="Disable physics controls panel in HTML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph_data = load_graph_json(args.input)
    g = build_nx_graph(graph_data)

    save_html(
        g,
        args.output,
        height=args.height,
        width=args.width,
        directed=not args.undirected,
        enable_physics_controls=not args.no_physics_controls,
    )

    print(
        f"Saved visualization: {args.output} | nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
