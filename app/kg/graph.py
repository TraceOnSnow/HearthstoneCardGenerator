from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

from app.kg.metadata import metadata_entry, metadata_name
from app.kg.models import CardRecord


def normalize_name(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return norm or "unknown"


def build_graph(
    cards: list[CardRecord],
    llm_outputs: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {"maps": {}}
    nodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    edge_keys: set[tuple[str, str, str, str]] = set()
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, name: str, **extra: Any) -> None:
        if node_id not in nodes:
            node = {"id": node_id, "type": node_type, "name": name}
            node.update({k: v for k, v in extra.items() if v is not None})
            nodes[node_id] = node

    def add_edge(source: str, predicate: str, target: str, **attributes: Any) -> None:
        clean_attributes = {k: v for k, v in attributes.items() if v is not None}
        attr_key = json.dumps(clean_attributes, ensure_ascii=False, sort_keys=True)
        key = (source, predicate, target, attr_key)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge = {"source": source, "predicate": predicate, "target": target}
        if clean_attributes:
            edge["attributes"] = clean_attributes
        edges.append(edge)

    for card in cards:
        card_id = f"card:{card.id}"
        add_node(
            card_id,
            "card",
            card.name,
            attributes={
                "manaCost": card.manaCost,
                "attack": card.attack,
                "health": card.health,
                "slug": card.slug,
            },
        )
        _add_explicit_metadata(card, metadata, add_node, add_edge)

    _add_llm_outputs(llm_outputs, add_node, add_edge)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {
            "cards": len(cards),
            "nodes": len(nodes),
            "edges": len(edges),
            "llm_batches": len(llm_outputs),
            "llm_ok_batches": sum(1 for row in llm_outputs if row.get("status") == "ok"),
        },
    }


def _add_explicit_metadata(card, metadata, add_node, add_edge) -> None:
    card_node = f"card:{card.id}"
    _add_id_node(card_node, "HAS_CARD_TYPE", "card_type", "cardTypeId", card.cardTypeId, metadata, add_node, add_edge)
    _add_id_node(card_node, "IN_SET", "card_set", "cardSetId", card.cardSetId, metadata, add_node, add_edge)
    _add_id_node(card_node, "HAS_CLASS", "class", "classId", card.classId, metadata, add_node, add_edge)
    _add_id_node(card_node, "HAS_RARITY", "rarity", "rarityId", card.rarityId, metadata, add_node, add_edge)
    _add_id_node(card_node, "HAS_SPELL_SCHOOL", "spell_school", "spellSchoolId", card.spellSchoolId, metadata, add_node, add_edge)
    _add_id_node(card_node, "HAS_MINION_TYPE", "minion_type", "minionTypeId", card.minionTypeId, metadata, add_node, add_edge)

    for class_id in card.multiClassIds:
        _add_id_node(card_node, "HAS_MULTI_CLASS", "class", "classId", class_id, metadata, add_node, add_edge)
    for keyword_id in card.keywordIds:
        _add_id_node(card_node, "HAS_KEYWORD", "keyword", "keywordIds", keyword_id, metadata, add_node, add_edge)
    for child_id in card.childIds:
        child_node = f"card:{child_id}"
        add_node(child_node, "card_ref", str(child_id))
        add_edge(card_node, "HAS_CHILD_CARD", child_node)

    if card.artistName:
        artist_node = f"artist:{normalize_name(card.artistName)}"
        add_node(artist_node, "artist", card.artistName)
        add_edge(card_node, "HAS_ARTIST", artist_node)


def _add_id_node(source: str, predicate: str, kind: str, field_name: str, value, metadata, add_node, add_edge) -> None:
    if not isinstance(value, int):
        return
    node_id = f"{kind}:{value}"
    entry = metadata_entry(metadata, field_name, value)
    extra = {"metadata": entry} if entry else {}
    add_node(node_id, kind, metadata_name(metadata, field_name, value), **extra)
    add_edge(source, predicate, node_id)


def _add_llm_outputs(llm_outputs: list[dict[str, Any]], add_node, add_edge) -> None:
    for out in llm_outputs:
        if out.get("status") not in {"ok", "dry_run"}:
            continue
        raw = out.get("raw_response", "")
        if not raw:
            continue

        parsed = parse_json_response(raw)
        if not parsed:
            continue

        for item in parsed.get("cards", []):
            card_id = item.get("card_id")
            if not isinstance(card_id, int):
                continue
            card_node = f"card:{card_id}"
            add_node(card_node, "card", str(item.get("name") or card_id))

            _add_semantic_card_fields(card_node, item, add_node, add_edge)
            _add_legacy_card_fields(card_node, item, add_node, add_edge)


def _add_semantic_card_fields(card_node: str, item: dict[str, Any], add_node, add_edge) -> None:
    for keyword in item.get("explicit_keywords", []):
        if not isinstance(keyword, str) or not keyword.strip():
            continue
        keyword_node = f"keyword_text:{normalize_name(keyword)}"
        add_node(keyword_node, "keyword_text", keyword.strip())
        add_edge(card_node, "HAS_EXPLICIT_KEYWORD_TEXT", keyword_node)

    for action in item.get("actions", []):
        if not isinstance(action, dict):
            continue
        action_type = _semantic_label(action.get("type"), default="other")
        action_node = f"action:{action_type}"
        add_node(action_node, "action", action_type)
        add_edge(
            card_node,
            "PERFORMS_ACTION",
            action_node,
            amount=_optional_int(action.get("amount")),
            target_label=_semantic_label_or_none(action.get("target")),
            condition=_semantic_label_or_none(action.get("condition")),
            resource=_semantic_label_or_none(action.get("resource")),
            raw_phrase=_optional_str(action.get("raw_phrase")),
        )

        target = _semantic_label_or_none(action.get("target"))
        if target and target != "no_target":
            target_node = f"target:{target}"
            add_node(target_node, "target", target)
            add_edge(card_node, "TARGETS", target_node)
            add_edge(action_node, "CAN_TARGET", target_node)

        condition = _semantic_label_or_none(action.get("condition"))
        if condition:
            condition_node = f"condition:{condition}"
            add_node(condition_node, "condition", condition)
            add_edge(card_node, "HAS_CONDITION", condition_node)
            add_edge(action_node, "REQUIRES_CONDITION", condition_node)

        resource = _semantic_label_or_none(action.get("resource"))
        if resource:
            resource_node = f"resource:{resource}"
            add_node(resource_node, "resource", resource)
            add_edge(card_node, "AFFECTS_RESOURCE", resource_node)
            add_edge(action_node, "AFFECTS_RESOURCE", resource_node)

    for resource in item.get("resources", []):
        if not isinstance(resource, dict):
            continue
        resource_type = _semantic_label(resource.get("type"), default="other")
        resource_node = f"resource:{resource_type}"
        add_node(resource_node, "resource", resource_type)
        add_edge(
            card_node,
            "AFFECTS_RESOURCE",
            resource_node,
            operation=_semantic_label_or_none(resource.get("operation")),
            amount=_optional_int(resource.get("amount")),
        )

    for reference in item.get("tribal_or_school_references", []):
        if not isinstance(reference, str) or not reference.strip():
            continue
        reference_node = f"reference:{normalize_name(reference)}"
        add_node(reference_node, "tribal_or_school_reference", reference.strip())
        add_edge(card_node, "REFERENCES_TRIBE_OR_SCHOOL", reference_node)

    for synergy in item.get("synergy_tags", []):
        synergy_label = _semantic_label_or_none(synergy)
        if not synergy_label:
            continue
        synergy_node = f"synergy:{synergy_label}"
        add_node(synergy_node, "synergy", synergy_label)
        add_edge(card_node, "HAS_SYNERGY", synergy_node)

    for constraint in item.get("constraints", []):
        constraint_label = _semantic_label_or_none(constraint)
        if not constraint_label:
            continue
        constraint_node = f"constraint:{constraint_label}"
        add_node(constraint_node, "constraint", constraint_label)
        add_edge(card_node, "HAS_CONSTRAINT", constraint_node)

    for phrase in item.get("raw_phrases", []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        phrase_node = f"phrase:{normalize_name(phrase)[:80]}"
        add_node(phrase_node, "raw_phrase", phrase.strip())
        add_edge(card_node, "HAS_RAW_PHRASE", phrase_node)


def _add_legacy_card_fields(card_node: str, item: dict[str, Any], add_node, add_edge) -> None:
    for mechanic in item.get("mechanics", []):
        if not isinstance(mechanic, str) or not mechanic.strip():
            continue
        mechanic_node = f"mechanic:{normalize_name(mechanic)}"
        add_node(mechanic_node, "mechanic", mechanic.strip())
        add_edge(card_node, "HAS_MECHANIC", mechanic_node)

    for entity in item.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("type", "other")).strip() or "other"
        entity_name = str(entity.get("name", "")).strip()
        if not entity_name:
            continue
        entity_node = f"entity:{entity_type}:{normalize_name(entity_name)}"
        add_node(entity_node, entity_type, entity_name)
        add_edge(card_node, "HAS_ENTITY", entity_node)


def _semantic_label(value: Any, *, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    return normalize_name(value)


def _semantic_label_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return normalize_name(value)


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def parse_json_response(raw: str) -> dict[str, Any] | None:
    raw = _strip_thinking_blocks(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _strip_thinking_blocks(raw: str) -> str:
    # MiniMax OpenAI-compatible responses may include visible <think> blocks before JSON.
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    return text or raw
