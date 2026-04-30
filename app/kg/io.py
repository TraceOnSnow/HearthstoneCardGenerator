from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from app.kg.models import CardRecord


def load_cards(path: Path, *, collectible_only: bool = True) -> list[CardRecord]:
    cards: list[CardRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            card = CardRecord.from_raw(raw)
            if card is None:
                continue
            if collectible_only and card.collectible != 1:
                continue
            cards.append(card)
    return cards


def select_cards(
    cards: list[CardRecord],
    *,
    limit: int | None,
    sample_size: int | None,
    seed: int,
) -> list[CardRecord]:
    if sample_size is not None:
        if sample_size >= len(cards):
            selected = list(cards)
        else:
            selected = random.Random(seed).sample(cards, sample_size)
    else:
        selected = list(cards)

    if limit is not None:
        selected = selected[:limit]

    return selected


def chunked(items: list[CardRecord], chunk_size: int) -> Iterable[list[CardRecord]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

