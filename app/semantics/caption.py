from __future__ import annotations

from typing import Any


def build_lora_caption(record: dict[str, Any], *, enriched: bool = False) -> str:
    identity = record.get("identity", {})
    parts = ["Hearthstone card art"]

    classes = identity.get("card_class") or []
    if classes:
        parts.append("/".join(classes))
    if identity.get("spell_school"):
        parts.append(f"{identity['spell_school']} spell")
    elif identity.get("card_type"):
        parts.append(str(identity["card_type"]))
    if identity.get("minion_type"):
        parts.append(str(identity["minion_type"]))

    parts.extend(record.get("keywords") or [])
    parts.extend(_action_phrases(record.get("actions") or []))
    parts.extend(record.get("mechanic_tags") or [])
    parts.extend(record.get("visual_tags") or [])

    if enriched:
        summary = record.get("semantic_summary")
        if summary:
            parts.append(str(summary))

    return ", ".join(_dedupe(parts))


def _action_phrases(actions: list[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    for action in actions:
        action_type = str(action.get("type", "")).replace("_", " ")
        target = action.get("target")
        if target:
            phrases.append(f"{action_type} {str(target).replace('_', ' ')}")
        elif action_type:
            phrases.append(action_type)
    return phrases


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        normalized = text.lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result

