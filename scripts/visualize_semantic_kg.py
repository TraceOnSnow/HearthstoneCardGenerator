#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyvis.network import Network


DEFAULT_NODE_TYPES = {
    "card",
    "class",
    "card_type",
    "keyword",
    "action",
    "target",
    "resource",
    "spell_school",
    "minion_type",
    "mechanic",
    "constraint",
    "action_group",
    "generated_role",
    "generated_card_name",
    "related_card_name",
}

NODE_COLORS = {
    "card": "#f8d16c",
    "class": "#72a7ff",
    "card_type": "#96d38c",
    "keyword": "#f49ac2",
    "action": "#ff8f70",
    "target": "#d6a8ff",
    "resource": "#80d8d0",
    "spell_school": "#b7a6ff",
    "minion_type": "#c6df7c",
    "mechanic": "#ffcf91",
    "constraint": "#fca5a5",
    "action_group": "#fcd34d",
    "generated_role": "#86efac",
    "generated_card_name": "#86efac",
    "related_card_name": "#a7f3d0",
    "artist": "#d4d4d4",
    "set": "#d4d4d4",
    "rarity": "#d4d4d4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render small, human-readable semantic KG subgraphs.")
    parser.add_argument("--graph", type=Path, default=Path("data/semantic_kg/graph.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/semantic_kg/sample_vis"))
    parser.add_argument("--card-id", type=int, action="append", help="Card id to visualize. Can be repeated.")
    parser.add_argument("--random-cards", type=int, default=3, help="Number of random cards to visualize when --card-id is omitted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=45)
    parser.add_argument("--include-derived-links", action="store_true", help="Include card-to-card derived links.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = _load_graph(args.graph)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    adjacency = _build_adjacency(graph["edges"])
    card_ids = args.card_id or _random_card_ids(graph["card_index"], count=args.random_cards, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for card_id in card_ids:
        center = f"card:{card_id}"
        if center not in node_by_id:
            print(f"skip missing card: {card_id}")
            continue
        subgraph = _extract_subgraph(
            center,
            node_by_id=node_by_id,
            adjacency=adjacency,
            depth=args.depth,
            max_nodes=args.max_nodes,
            include_derived_links=args.include_derived_links,
        )
        out = args.out_dir / f"card_{card_id}.html"
        _render_html(subgraph, out)
        outputs.append(out)

    index = args.out_dir / "index.html"
    _write_index(index, outputs)
    print("Done.")
    print(f"cards={len(outputs)}")
    print(f"index={index}")
    for out in outputs:
        print(f"html={out}")


def _load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_adjacency(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        adjacency[source].append(edge)
        adjacency[target].append({"source": target, "predicate": f"INVERSE_{edge.get('predicate')}", "target": source, "attributes": edge.get("attributes", {})})
    return adjacency


def _random_card_ids(card_index: list[dict[str, Any]], *, count: int, seed: int) -> list[int]:
    candidates = [
        row["card_id"]
        for row in card_index
        if row.get("image") and row.get("collectible") and isinstance(row.get("card_id"), int)
    ]
    if count >= len(candidates):
        return candidates
    return random.Random(seed).sample(candidates, count)


def _extract_subgraph(
    center: str,
    *,
    node_by_id: dict[str, dict[str, Any]],
    adjacency: dict[str, list[dict[str, Any]]],
    depth: int,
    max_nodes: int,
    include_derived_links: bool,
) -> dict[str, list[dict[str, Any]]]:
    selected_nodes: set[str] = {center}
    selected_edges: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque([(center, 0)])

    while queue and len(selected_nodes) < max_nodes:
        current, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in adjacency.get(current, []):
            target = edge["target"]
            target_node = node_by_id.get(target)
            if not target_node:
                continue
            if not _include_node(target_node, include_derived_links=include_derived_links):
                continue
            selected_edges.append(edge)
            if target not in selected_nodes:
                selected_nodes.add(target)
                queue.append((target, current_depth + 1))
            if len(selected_nodes) >= max_nodes:
                break

    nodes = [node_by_id[node_id] for node_id in selected_nodes if node_id in node_by_id]
    clean_edges = [
        edge
        for edge in selected_edges
        if edge.get("source") in selected_nodes and edge.get("target") in selected_nodes
    ]
    return {"nodes": nodes, "edges": clean_edges}


def _include_node(node: dict[str, Any], *, include_derived_links: bool) -> bool:
    node_type = node.get("type")
    if node_type == "card":
        return include_derived_links
    return node_type in DEFAULT_NODE_TYPES


def _render_html(subgraph: dict[str, list[dict[str, Any]]], out: Path) -> None:
    net = Network(height="850px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111827")
    net.barnes_hut(gravity=-6000, central_gravity=0.25, spring_length=140, spring_strength=0.03)

    for node in subgraph["nodes"]:
        node_id = node["id"]
        node_type = node.get("type", "other")
        attrs = node.get("attributes", {})
        label = str(node.get("name") or node_id)
        title = json.dumps(node, ensure_ascii=False, indent=2)
        size = 28 if node_type == "card" else 16
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=NODE_COLORS.get(node_type, "#e5e7eb"),
            shape="box" if node_type == "card" else "dot",
            size=size,
        )

    for edge in subgraph["edges"]:
        label = str(edge.get("predicate", ""))
        net.add_edge(edge["source"], edge["target"], label=label, title=json.dumps(edge, ensure_ascii=False, indent=2))

    net.show_buttons(filter_=["physics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))


def _write_index(index: Path, outputs: list[Path]) -> None:
    links = "\n".join(f'<li><a href="{out.name}">{out.name}</a></li>' for out in outputs)
    index.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Semantic KG Samples</title></head>
<body>
<h1>Semantic KG Samples</h1>
<ul>
{links}
</ul>
</body>
</html>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
