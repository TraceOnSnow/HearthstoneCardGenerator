from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.kg.io import read_jsonl


def load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        return data["queries"]
    if isinstance(data, list):
        return data
    raise ValueError("Query file must be a JSON list or an object with a 'queries' list.")


def load_caption_corpus(path: Path, *, require_image: bool = True) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        image = row.get("image") or row.get("file_name") or ""
        caption = row.get("caption") or row.get("text") or ""
        if require_image and not image:
            continue
        rows.append(
            {
                "card_id": row.get("card_id"),
                "card_name": row.get("name", ""),
                "image": image,
                "caption": caption,
                "collectible": row.get("collectible", False),
                "root_collectible_ids": row.get("root_collectible_ids", []),
            }
        )
    return rows


def query_to_text(query: dict[str, Any]) -> str:
    parts = [str(query.get("text", ""))]
    for key, value in query.items():
        if key in {"query_id", "text", "generation_hints"}:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def result_row(
    *,
    query: dict[str, Any],
    method: str,
    rank: int,
    card: dict[str, Any],
    score: float,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "query_id": query.get("query_id", ""),
        "query_text": query.get("text", ""),
        "method": method,
        "rank": rank,
        "card_id": card.get("card_id"),
        "card_name": card.get("card_name", ""),
        "image": card.get("image", ""),
        "caption": card.get("caption", ""),
        "score": round(float(score), 6),
        "reasons": reasons or [],
    }
