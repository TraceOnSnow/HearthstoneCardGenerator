from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from app.kg.io import read_jsonl, write_jsonl


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def build_semantic_kg(*, semantics_path: Path, out_dir: Path) -> dict[str, Any]:
    records = read_jsonl(semantics_path)
    graph = graph_from_semantics(records)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = graph["nodes"]
    edges = graph["edges"]
    write_jsonl(out_dir / "nodes.jsonl", nodes)
    write_jsonl(out_dir / "edges.jsonl", edges)
    write_jsonl(out_dir / "card_index.jsonl", graph["card_index"])
    (out_dir / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(graph["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(out_dir, graph["stats"])
    return graph["stats"] | {"out_dir": str(out_dir)}


def graph_from_semantics(records: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str, str]] = set()
    card_index: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, name: str, **attributes: Any) -> None:
        if node_id in nodes:
            return
        node = {"id": node_id, "type": node_type, "name": name}
        clean_attrs = {key: value for key, value in attributes.items() if value not in (None, "", [])}
        if clean_attrs:
            node["attributes"] = clean_attrs
        nodes[node_id] = node

    def add_edge(source: str, predicate: str, target: str, **attributes: Any) -> None:
        clean_attrs = {key: value for key, value in attributes.items() if value not in (None, "", [])}
        attr_key = json.dumps(clean_attrs, ensure_ascii=False, sort_keys=True)
        key = (source, predicate, target, attr_key)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge = {"source": source, "predicate": predicate, "target": target}
        if clean_attrs:
            edge["attributes"] = clean_attrs
        edges.append(edge)

    known_cards = {record["card_id"] for record in records if isinstance(record.get("card_id"), int)}

    for record in records:
        card_id = record.get("card_id")
        if not isinstance(card_id, int):
            continue
        card_node = f"card:{card_id}"
        stats = record.get("stats", {})
        source = record.get("source", {})
        add_node(
            card_node,
            "card",
            str(record.get("name") or card_id),
            card_id=card_id,
            slug=record.get("slug"),
            collectible=record.get("collectible"),
            is_derived=record.get("is_derived"),
            derivation_depth=record.get("derivation_depth"),
            mana_cost=stats.get("mana_cost"),
            attack=stats.get("attack"),
            health=stats.get("health"),
            image=source.get("art_image"),
        )

        node_ids: list[str] = []
        node_ids.extend(_add_identity(record, card_node, add_node, add_edge))
        node_ids.extend(_add_keywords(record, card_node, add_node, add_edge))
        node_ids.extend(_add_actions(record, card_node, add_node, add_edge))
        node_ids.extend(_add_tags(record, card_node, add_node, add_edge))
        node_ids.extend(_add_constraints(record, card_node, add_node, add_edge))
        node_ids.extend(_add_action_groups(record, card_node, add_node, add_edge))
        node_ids.extend(_add_card_links(record, card_node, known_cards, add_node, add_edge))
        node_ids.extend(_add_generated_refs(record, card_node, known_cards, add_node, add_edge))

        card_index.append(
            {
                "card_id": card_id,
                "name": record.get("name", ""),
                "collectible": record.get("collectible", False),
                "is_derived": record.get("is_derived", False),
                "image": source.get("art_image", ""),
                "text": record.get("text", {}).get("clean", ""),
                "semantic_summary": record.get("semantic_summary", ""),
                "node_ids": sorted(set(node_ids)),
            }
        )

    stats = {
        "cards": len(card_index),
        "nodes": len(nodes),
        "edges": len(edges),
        "card_index": len(card_index),
    }
    return {"nodes": list(nodes.values()), "edges": edges, "card_index": card_index, "stats": stats}


def _add_identity(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    identity = record.get("identity", {})
    node_ids: list[str] = []

    for class_name in identity.get("card_class") or []:
        node_ids.append(_add_value_node(card_node, "HAS_CLASS", "class", class_name, add_node, add_edge))
    for field, predicate, node_type in [
        ("card_type", "HAS_CARD_TYPE", "card_type"),
        ("set", "IN_SET", "set"),
        ("rarity", "HAS_RARITY", "rarity"),
        ("spell_school", "HAS_SPELL_SCHOOL", "spell_school"),
        ("minion_type", "HAS_MINION_TYPE", "minion_type"),
    ]:
        value = identity.get(field)
        if value:
            node_ids.append(_add_value_node(card_node, predicate, node_type, value, add_node, add_edge))

    artist = identity.get("artist")
    if artist:
        node_ids.append(_add_value_node(card_node, "HAS_ARTIST", "artist", artist, add_node, add_edge))
    return node_ids


def _add_keywords(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    node_ids = []
    for keyword in record.get("keywords") or []:
        node_ids.append(_add_value_node(card_node, "HAS_KEYWORD", "keyword", keyword, add_node, add_edge))
    return node_ids


def _add_actions(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    node_ids = []
    for action in record.get("actions") or []:
        if not isinstance(action, dict):
            continue
        action_type = action.get("type") or "other"
        action_node = _add_value_node(
            card_node,
            "PERFORMS_ACTION",
            "action",
            action_type,
            add_node,
            add_edge,
            amount=action.get("amount"),
            target_label=action.get("target"),
            target_scope=action.get("target_scope"),
            resource=action.get("resource"),
            condition=action.get("condition"),
            trigger=action.get("trigger"),
            duration=action.get("duration"),
            raw_phrase=action.get("raw_phrase"),
        )
        node_ids.append(action_node)
        if action.get("target"):
            node_ids.append(_add_value_node(card_node, "TARGETS", "target", action["target"], add_node, add_edge))
        if action.get("resource"):
            node_ids.append(_add_value_node(card_node, "AFFECTS_RESOURCE", "resource", action["resource"], add_node, add_edge))
        if action.get("condition"):
            node_ids.append(_add_value_node(card_node, "HAS_CONDITION", "condition", action["condition"], add_node, add_edge))
        if action.get("trigger"):
            node_ids.append(_add_value_node(card_node, "HAS_TRIGGER", "trigger", action["trigger"], add_node, add_edge))
    return node_ids


def _add_tags(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    node_ids = []
    for tag in record.get("mechanic_tags") or []:
        node_ids.append(_add_value_node(card_node, "HAS_MECHANIC_TAG", "mechanic", tag, add_node, add_edge))
    return node_ids


def _add_constraints(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    node_ids = []
    for constraint in record.get("constraints") or []:
        node_ids.append(_add_value_node(card_node, "HAS_CONSTRAINT", "constraint", constraint, add_node, add_edge))
    return node_ids


def _add_action_groups(record: dict[str, Any], card_node: str, add_node, add_edge) -> list[str]:
    node_ids = []
    for idx, group in enumerate(record.get("action_groups") or []):
        if not isinstance(group, dict):
            continue
        group_type = group.get("type") or "action_group"
        group_node = f"action_group:{record.get('card_id')}:{idx}"
        add_node(
            group_node,
            "action_group",
            str(group_type),
            group_type=group_type,
            raw_phrase=group.get("raw_phrase"),
            options=group.get("options"),
            action_indices=group.get("action_indices"),
        )
        add_edge(card_node, "HAS_ACTION_GROUP", group_node)
        node_ids.append(group_node)
        if group_type:
            node_ids.append(_add_value_node(card_node, "HAS_MECHANIC_TAG", "mechanic", group_type, add_node, add_edge))
    return node_ids


def _add_card_links(record: dict[str, Any], card_node: str, known_cards: set[int], add_node, add_edge) -> list[str]:
    node_ids = []
    child_roles = {
        item.get("card_id"): item
        for item in record.get("derived_cards") or []
        if isinstance(item, dict) and isinstance(item.get("card_id"), int)
    }
    for child_id in record.get("child_card_ids") or []:
        if child_id in known_cards:
            child_node = f"card:{child_id}"
            role_row = child_roles.get(child_id, {})
            add_edge(card_node, "HAS_CHILD_CARD", child_node, role=role_row.get("role"), evidence=role_row.get("evidence"))
            node_ids.append(child_node)
    for parent_id in record.get("parent_card_ids") or []:
        if parent_id in known_cards:
            parent_node = f"card:{parent_id}"
            add_edge(card_node, "DERIVED_FROM", parent_node)
            node_ids.append(parent_node)
    for root_id in record.get("root_collectible_ids") or []:
        if root_id in known_cards and root_id != record.get("card_id"):
            root_node = f"card:{root_id}"
            add_edge(card_node, "HAS_ROOT_COLLECTIBLE", root_node)
            node_ids.append(root_node)
    return node_ids


def _add_generated_refs(record: dict[str, Any], card_node: str, known_cards: set[int], add_node, add_edge) -> list[str]:
    node_ids = []
    for ref in record.get("generated_card_refs") or []:
        if not isinstance(ref, dict):
            continue
        role = ref.get("role")
        if role:
            node_ids.append(_add_value_node(card_node, "GENERATES_CARD_ROLE", "generated_role", role, add_node, add_edge))

        ref_card_id = ref.get("card_id")
        if isinstance(ref_card_id, int) and ref_card_id in known_cards:
            ref_node = f"card:{ref_card_id}"
            add_edge(card_node, "GENERATES_CARD", ref_node, role=role, evidence=ref.get("evidence"))
            node_ids.append(ref_node)
            continue

        name = ref.get("name")
        if name:
            name_node = _add_value_node(card_node, "MENTIONS_GENERATED_CARD", "generated_card_name", name, add_node, add_edge, role=role, evidence=ref.get("evidence"))
            node_ids.append(name_node)

    for ref in record.get("related_card_refs") or []:
        if not isinstance(ref, dict):
            continue
        relation = _normalize_predicate(ref.get("relation") or "RELATED_TO_CARD")
        ref_card_id = ref.get("card_id")
        if isinstance(ref_card_id, int) and ref_card_id in known_cards:
            ref_node = f"card:{ref_card_id}"
            add_edge(card_node, relation, ref_node, evidence=ref.get("evidence"))
            node_ids.append(ref_node)
        elif ref.get("name"):
            node_ids.append(
                _add_value_node(
                    card_node,
                    relation,
                    "related_card_name",
                    ref["name"],
                    add_node,
                    add_edge,
                    evidence=ref.get("evidence"),
                )
            )
    return node_ids


def _add_value_node(card_node: str, predicate: str, node_type: str, value: Any, add_node, add_edge, **edge_attrs: Any) -> str:
    label = str(value).strip()
    node_id = f"{node_type}:{normalize_label(label)}"
    add_node(node_id, node_type, label)
    add_edge(card_node, predicate, node_id, **edge_attrs)
    return node_id


def _normalize_predicate(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "RELATED_TO_CARD"


def _write_readme(out_dir: Path, stats: dict[str, Any]) -> None:
    readme = f"""# Semantic Hearthstone KG

This KG is derived from structured gameplay semantics. Visual tags and LoRA captions
are intentionally excluded from graph facts and retrieval scoring.

## Files

- `graph.json`: complete graph with nodes, edges, card index, and stats.
- `nodes.jsonl`: one node per line.
- `edges.jsonl`: one edge per line.
- `card_index.jsonl`: card-level retrieval index.
- `summary.json`: graph statistics.

## Stats

- Cards: {stats["cards"]}
- Nodes: {stats["nodes"]}
- Edges: {stats["edges"]}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
