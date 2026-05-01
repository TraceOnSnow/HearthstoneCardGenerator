from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.kg.io import read_jsonl, write_jsonl
from app.semantic_kg.build import normalize_label


FIELD_TO_NODE_TYPE = {
    "classes": "class",
    "card_types": "card_type",
    "keywords": "keyword",
    "actions": "action",
    "targets": "target",
    "resources": "resource",
    "spell_schools": "spell_school",
    "minion_types": "minion_type",
    "mechanic_tags": "mechanic",
    "constraints": "constraint",
    "generated_roles": "generated_role",
    "generated_card_names": "generated_card_name",
    "related_card_names": "related_card_name",
    "triggers": "trigger",
    "conditions": "condition",
}

FIELD_WEIGHTS = {
    "classes": 2.0,
    "card_types": 1.5,
    "keywords": 2.0,
    "actions": 3.0,
    "targets": 1.5,
    "resources": 1.0,
    "spell_schools": 1.5,
    "minion_types": 1.5,
    "mechanic_tags": 2.0,
    "constraints": 1.5,
    "generated_roles": 2.0,
    "generated_card_names": 2.5,
    "related_card_names": 2.0,
    "triggers": 1.0,
    "conditions": 1.0,
}

TEXT_MATCH_WEIGHT = 0.2


def load_structured_queries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        return data["queries"]
    if isinstance(data, list):
        return data
    raise ValueError("Query file must be a JSON list or an object with a 'queries' list.")


def retrieve_many(
    *,
    card_index_path: Path,
    queries: list[dict[str, Any]],
    top_k: int,
    out_path: Path | None = None,
    require_image: bool = True,
) -> list[dict[str, Any]]:
    cards = read_jsonl(card_index_path)
    results: list[dict[str, Any]] = []
    for query in queries:
        results.extend(retrieve_one(cards, query=query, top_k=top_k, require_image=require_image))
    if out_path:
        write_jsonl(out_path, results)
    return results


def retrieve_one(
    cards: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    top_k: int,
    require_image: bool = True,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    query_nodes = query_to_nodes(query)
    query_tokens = _tokens(_query_text(query))

    for card in cards:
        if require_image and not card.get("image"):
            continue
        card_nodes = set(card.get("node_ids") or [])
        reasons: list[str] = []
        score = 0.0

        for item in query_nodes:
            if item["node_id"] not in card_nodes:
                continue
            score += item["weight"]
            reasons.append(f"{item['field']}={item['label']}")

        text_overlap = len(query_tokens & _tokens(_card_text(card)))
        if text_overlap:
            score += TEXT_MATCH_WEIGHT * text_overlap
            reasons.append(f"text_overlap={text_overlap}")

        if score <= 0:
            continue
        scored.append(
            {
                "query_id": query.get("query_id", ""),
                "query_text": query.get("text", ""),
                "method": "semantic_kg",
                "card_id": card.get("card_id"),
                "card_name": card.get("name", ""),
                "image": card.get("image", ""),
                "score": round(score, 4),
                "reasons": reasons,
            }
        )

    scored.sort(key=lambda row: (-row["score"], str(row["card_name"]), int(row["card_id"] or 0)))
    ranked = scored[:top_k]
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def query_to_nodes(query: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for field, node_type in FIELD_TO_NODE_TYPE.items():
        values = query.get(field) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if value in (None, ""):
                continue
            label = str(value).strip()
            nodes.append(
                {
                    "field": field,
                    "label": label,
                    "node_id": f"{node_type}:{normalize_label(label)}",
                    "weight": FIELD_WEIGHTS.get(field, 1.0),
                }
            )
    return nodes


def _query_text(query: dict[str, Any]) -> str:
    parts = [str(query.get("text", ""))]
    for field in FIELD_TO_NODE_TYPE:
        values = query.get(field) or []
        if isinstance(values, str):
            values = [values]
        parts.extend(str(value) for value in values)
    return " ".join(parts)


def _card_text(card: dict[str, Any]) -> str:
    return " ".join([str(card.get("name", "")), str(card.get("text", "")), str(card.get("semantic_summary", ""))])


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}
