from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kg.io import chunked
from app.kg.models import CardRecord


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, cards: list[CardRecord]) -> str:
    cards_json = json.dumps(
        [card.to_prompt_dict() for card in cards],
        ensure_ascii=False,
        indent=2,
    )
    return template.replace("{{CARDS_JSON}}", cards_json)


def build_prompt_rows(
    cards: list[CardRecord],
    *,
    template: str,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_idx, chunk in enumerate(chunked(cards, chunk_size), start=1):
        rows.append(
            {
                "batch_id": batch_idx,
                "card_count": len(chunk),
                "card_ids": [card.id for card in chunk],
                "cards": [card.to_prompt_dict() for card in chunk],
                "prompt": build_prompt(template, chunk),
            }
        )
    return rows

