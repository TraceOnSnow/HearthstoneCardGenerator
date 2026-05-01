from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from app.kg.io import write_jsonl
from app.kg.metadata import load_metadata, metadata_name
from app.semantics.caption import build_lora_caption
from app.semantics.rule_extractors import extract_actions, infer_mechanic_tags, infer_visual_tags
from app.semantics.text import clean_card_text


SPECIAL_MODE_CARD_SET_IDS = {
    2,  # Missions/debug-only cards in the Blizzard API payload.
    17,  # Hero skins.
    1453,  # Battlegrounds.
    1586,  # Mercenaries.
}

SPECIAL_MODE_CARD_TYPE_IDS = {
    42,  # Battlegrounds Tavern spell / upgrade-style cards.
    43,  # Battlegrounds anomaly / quest reward-style cards.
    44,  # Battlegrounds trinkets.
    999,  # Timewarp system card type.
}


def build_semantics(
    *,
    cards_path: Path,
    metadata_path: Path | None,
    out_dir: Path,
    art_metadata_path: Path | None = None,
    exclude_special_modes: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    raw_cards = _read_cards(cards_path, limit=limit)
    cards = _filter_cards(raw_cards, exclude_special_modes=exclude_special_modes)
    metadata = load_metadata(metadata_path)
    art_index = _load_art_index(art_metadata_path)
    by_id = {card["id"]: card for card in cards if isinstance(card.get("id"), int)}
    parent_index, child_index = _build_edges(cards)
    root_index, depth_index = _derive_roots_and_depths(by_id, parent_index, child_index)

    records = [
        _build_record(card, metadata, art_index, parent_index, child_index, root_index, depth_index)
        for card in cards
        if isinstance(card.get("id"), int)
    ]
    record_by_id = {record["card_id"]: record for record in records}
    for record in records:
        record["expanded_semantics"] = _expanded_semantics(record, record_by_id)
        record["lora_caption"] = build_lora_caption(record)

    edges = _edge_rows(parent_index)
    captions = _caption_rows(records, require_image=bool(art_index))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cards_semantics_base.jsonl", records)
    write_jsonl(out_dir / "derived_edges.jsonl", edges)
    write_jsonl(out_dir / "lora_captions.jsonl", captions)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "cards": len(records),
                "source_cards": len(raw_cards),
                "excluded_special_mode_cards": len(raw_cards) - len(cards),
                "exclude_special_modes": exclude_special_modes,
                "collectible_cards": sum(1 for row in records if row["collectible"]),
                "derived_cards": sum(1 for row in records if row["is_derived"]),
                "derived_edges": len(edges),
                "captions": len(captions),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "source_cards": len(raw_cards),
        "excluded_special_mode_cards": len(raw_cards) - len(cards),
        "cards": len(records),
        "edges": len(edges),
        "captions": len(captions),
        "out_dir": str(out_dir),
    }


def _read_cards(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("id"), int):
                cards.append(row)
            if limit is not None and len(cards) >= limit:
                break
    return cards


def _filter_cards(cards: list[dict[str, Any]], *, exclude_special_modes: bool) -> list[dict[str, Any]]:
    if not exclude_special_modes:
        return cards
    return [card for card in cards if not _is_special_mode_card(card)]


def _is_special_mode_card(card: dict[str, Any]) -> bool:
    return (
        card.get("cardSetId") in SPECIAL_MODE_CARD_SET_IDS
        or card.get("cardTypeId") in SPECIAL_MODE_CARD_TYPE_IDS
    )


def _build_edges(cards: list[dict[str, Any]]) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    parent_index: dict[int, set[int]] = defaultdict(set)
    child_index: dict[int, set[int]] = defaultdict(set)
    card_ids = {card["id"] for card in cards if isinstance(card.get("id"), int)}
    for card in cards:
        card_id = card.get("id")
        if not isinstance(card_id, int):
            continue
        parent_id = card.get("parentId")
        if isinstance(parent_id, int) and parent_id in card_ids:
            parent_index[card_id].add(parent_id)
            child_index[parent_id].add(card_id)
        for child_id in card.get("childIds") or []:
            if isinstance(child_id, int) and child_id in card_ids:
                parent_index[child_id].add(card_id)
                child_index[card_id].add(child_id)
    return parent_index, child_index


def _derive_roots_and_depths(
    by_id: dict[int, dict[str, Any]],
    parent_index: dict[int, set[int]],
    child_index: dict[int, set[int]],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    roots: dict[int, set[int]] = defaultdict(set)
    depths: dict[int, int] = {}
    queue: deque[tuple[int, int, int]] = deque()

    for card_id, card in by_id.items():
        if card.get("collectible") == 1:
            roots[card_id].add(card_id)
            depths[card_id] = 0
            queue.append((card_id, card_id, 0))

    while queue:
        root_id, current_id, depth = queue.popleft()
        for child_id in child_index.get(current_id, set()):
            if child_id not in by_id:
                continue
            previous_roots = len(roots[child_id])
            roots[child_id].add(root_id)
            if child_id not in depths or depth + 1 < depths[child_id]:
                depths[child_id] = depth + 1
            if len(roots[child_id]) != previous_roots:
                queue.append((root_id, child_id, depth + 1))

    for card_id in by_id:
        if card_id not in roots:
            ancestors = _ancestor_collectible_roots(card_id, by_id, parent_index)
            roots[card_id].update(ancestors)
        depths.setdefault(card_id, 0 if by_id[card_id].get("collectible") == 1 else -1)
    return roots, depths


def _ancestor_collectible_roots(card_id: int, by_id: dict[int, dict[str, Any]], parent_index: dict[int, set[int]]) -> set[int]:
    roots: set[int] = set()
    queue: deque[int] = deque(parent_index.get(card_id, set()))
    seen: set[int] = set()
    while queue:
        parent_id = queue.popleft()
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent and parent.get("collectible") == 1:
            roots.add(parent_id)
        queue.extend(parent_index.get(parent_id, set()))
    return roots


def _build_record(
    card: dict[str, Any],
    metadata: dict[str, Any],
    art_index: dict[int, dict[str, Any]],
    parent_index: dict[int, set[int]],
    child_index: dict[int, set[int]],
    root_index: dict[int, set[int]],
    depth_index: dict[int, int],
) -> dict[str, Any]:
    card_id = card["id"]
    clean_text = clean_card_text(card.get("text"))
    keywords = _keywords(card, metadata, clean_text)
    identity = _identity(card, metadata)
    art = _match_art(card, art_index)
    actions = extract_actions(clean_text)
    mechanic_tags = infer_mechanic_tags(card, actions, keywords)
    visual_tags = infer_visual_tags(identity, actions, keywords)

    return {
        "card_id": card_id,
        "dbf_id": card.get("dbfId"),
        "slug": card.get("slug", ""),
        "name": card.get("name", ""),
        "collectible": card.get("collectible") == 1,
        "is_derived": bool(parent_index.get(card_id)) and card.get("collectible") != 1,
        "root_collectible_ids": sorted(root_index.get(card_id, set())),
        "parent_card_ids": sorted(parent_index.get(card_id, set())),
        "child_card_ids": sorted(child_index.get(card_id, set())),
        "derivation_depth": depth_index.get(card_id, -1),
        "identity": identity,
        "stats": {
            "mana_cost": card.get("manaCost"),
            "attack": card.get("attack"),
            "health": card.get("health"),
            "durability": card.get("durability"),
        },
        "text": {
            "raw": card.get("text", ""),
            "clean": clean_text,
        },
        "keywords": keywords,
        "actions": actions,
        "mechanic_tags": mechanic_tags,
        "visual_tags": visual_tags,
        "derived_cards": [
            {"card_id": child_id, "relation": "HAS_CHILD_CARD", "role": None}
            for child_id in sorted(child_index.get(card_id, set()))
        ],
        "semantic_summary": "",
        "source": {
            "image": card.get("image", ""),
            "crop_image": card.get("cropImage", ""),
            "art_image": art.get("file_name", ""),
            "art_card_id": art.get("card_id", ""),
            "art_dbf_id": art.get("dbf_id"),
        },
    }


def _identity(card: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    class_ids = []
    if isinstance(card.get("classId"), int):
        class_ids.append(card["classId"])
    class_ids.extend(card.get("multiClassIds") or [])
    class_names = [metadata_name(metadata, "classId", class_id) for class_id in class_ids]
    return {
        "card_type": metadata_name(metadata, "cardTypeId", card.get("cardTypeId")) or None,
        "card_class": [name for name in class_names if name],
        "set": metadata_name(metadata, "cardSetId", card.get("cardSetId")) or None,
        "rarity": metadata_name(metadata, "rarityId", card.get("rarityId")) or None,
        "spell_school": metadata_name(metadata, "spellSchoolId", card.get("spellSchoolId")) or None,
        "minion_type": metadata_name(metadata, "minionTypeId", card.get("minionTypeId")) or None,
        "artist": card.get("artistName") or None,
    }


def _keywords(card: dict[str, Any], metadata: dict[str, Any], clean_text: str) -> list[str]:
    values = [metadata_name(metadata, "keywordIds", keyword_id) for keyword_id in card.get("keywordIds") or []]
    known_text_keywords = [
        "Battlecry",
        "Deathrattle",
        "Taunt",
        "Lifesteal",
        "Rush",
        "Charge",
        "Divine Shield",
        "Windfury",
        "Reborn",
        "Stealth",
        "Tradeable",
        "Forge",
    ]
    lowered = clean_text.lower()
    values.extend(keyword for keyword in known_text_keywords if keyword.lower() in lowered)
    return _unique([value for value in values if value])


def _expanded_semantics(record: dict[str, Any], record_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    child_ids = record.get("child_card_ids", [])[:20]
    child_records = [record_by_id[child_id] for child_id in child_ids if child_id in record_by_id]
    actions = _unique([action["type"] for action in record.get("actions", [])])
    keywords = list(record.get("keywords", []))
    mechanic_tags = list(record.get("mechanic_tags", []))
    visual_tags = list(record.get("visual_tags", []))
    for child in child_records:
        keywords.extend(child.get("keywords", []))
        actions.extend(action["type"] for action in child.get("actions", []))
        mechanic_tags.extend(child.get("mechanic_tags", []))
        visual_tags.extend(child.get("visual_tags", []))
    return {
        "self_card_id": record["card_id"],
        "included_child_card_ids": [child["card_id"] for child in child_records],
        "keywords": _unique(keywords),
        "actions": _unique(actions),
        "mechanic_tags": _unique(mechanic_tags),
        "visual_tags": _unique(visual_tags),
        "summary": _summary(record),
    }


def _summary(record: dict[str, Any]) -> str:
    identity = record.get("identity", {})
    card_class = "/".join(identity.get("card_class") or [])
    action_text = ", ".join(action["type"].replace("_", " ") for action in record.get("actions", [])[:3])
    pieces = [record.get("name", "This card")]
    if card_class:
        pieces.append(f"is a {card_class} card")
    if action_text:
        pieces.append(f"that {action_text}")
    return " ".join(pieces).strip() + "."


def _edge_rows(parent_index: dict[int, set[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child_id, parent_ids in sorted(parent_index.items()):
        for parent_id in sorted(parent_ids):
            rows.append({"source": parent_id, "target": child_id, "relation": "HAS_CHILD_CARD"})
    return rows


def _caption_rows(records: list[dict[str, Any]], *, require_image: bool) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if not record.get("name"):
            continue
        image = record.get("source", {}).get("art_image", "")
        if require_image and not image:
            continue
        rows.append(
            {
                "card_id": record["card_id"],
                "slug": record.get("slug", ""),
                "name": record.get("name", ""),
                "collectible": record.get("collectible", False),
                "root_collectible_ids": record.get("root_collectible_ids", []),
                "image": image,
                "caption": record.get("lora_caption", ""),
            }
        )
    return rows


def _load_art_index(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    index: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dbf_id = row.get("dbf_id")
            if not isinstance(dbf_id, int):
                continue
            # HearthstoneJSON art metadata dbf_id is the Blizzard card JSON id.
            index[dbf_id] = row
    return index


def _match_art(card: dict[str, Any], art_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    card_id = card.get("id")
    if not isinstance(card_id, int):
        return {}
    return art_index.get(card_id, {})


def _unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
