from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ID_COLLECTIONS = {
    "cardSetId": "sets",
    "cardTypeId": "types",
    "classId": "classes",
    "rarityId": "rarities",
    "spellSchoolId": "spellSchools",
    "minionTypeId": "minionTypes",
    "keywordIds": "keywords",
}


def normalize_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for field_name, collection_name in ID_COLLECTIONS.items():
        maps[field_name] = _normalize_collection(raw.get(collection_name, []))

    return {
        "source": "blizzard_hearthstone_metadata",
        "maps": maps,
        "raw_collection_counts": {
            key: len(value) for key, value in raw.items() if isinstance(value, list)
        },
    }


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"maps": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_entry(metadata: dict[str, Any], field_name: str, value: int | None) -> dict[str, Any] | None:
    if not isinstance(value, int):
        return None
    field_map = metadata.get("maps", {}).get(field_name, {})
    entry = field_map.get(str(value))
    return entry if isinstance(entry, dict) else None


def metadata_name(metadata: dict[str, Any], field_name: str, value: int | None) -> str:
    entry = metadata_entry(metadata, field_name, value)
    if entry and entry.get("name"):
        return str(entry["name"])
    return str(value) if isinstance(value, int) else ""


def _normalize_collection(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue

        entry = {
            "id": item["id"],
            "name": str(item.get("name", item["id"])).strip(),
        }
        if item.get("slug"):
            entry["slug"] = str(item["slug"]).strip()
        if item.get("text"):
            entry["text"] = str(item["text"]).strip()
        if item.get("refText"):
            entry["refText"] = str(item["refText"]).strip()
        if item.get("gameModes"):
            entry["gameModes"] = item["gameModes"]

        normalized[str(item["id"])] = entry
    return normalized

