import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.visualize import build_nx_graph, load_graph_json, save_graph_html


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
    save_graph_html(
        graph_data,
        args.output,
        height=args.height,
        width=args.width,
        directed=not args.undirected,
        physics_controls=not args.no_physics_controls,
    )

    g = build_nx_graph(graph_data)
    print(
        f"Saved visualization: {args.output} | nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
