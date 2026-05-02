from __future__ import annotations

import json
import re
from typing import Any

from app.kg.graph import parse_json_response
from app.kg.llm import _api_key_for_provider, _generate_content, load_dotenv


CLASSES = [
    "Death Knight",
    "Demon Hunter",
    "Druid",
    "Hunter",
    "Mage",
    "Paladin",
    "Priest",
    "Rogue",
    "Shaman",
    "Warlock",
    "Warrior",
    "Neutral",
]

CARD_TYPES = ["Spell", "Weapon", "Hero", "Location"]
KEYWORDS = ["Lifesteal", "Taunt", "Deathrattle", "Battlecry", "Discover", "Rush", "Charge", "Divine Shield", "Freeze", "Secret"]
SPELL_SCHOOLS = ["Arcane", "Fire", "Frost", "Nature", "Holy", "Shadow", "Fel"]
MINION_TYPES = ["Beast", "Demon", "Dragon", "Elemental", "Mech", "Murloc", "Naga", "Pirate", "Totem", "Undead"]

ACTION_PATTERNS = [
    ("deal_damage", r"\b(deal|damage|burn|blast|hit|drain|drains|draining)\b"),
    ("heal", r"\b(heal|restore|healing)\b"),
    ("gain_armor", r"\b(armor|armour)\b"),
    ("summon", r"\b(summon|token|tokens)\b"),
    ("draw", r"\b(draw)\b"),
    ("discover", r"\b(discover)\b"),
    ("destroy", r"\b(destroy|kill)\b"),
    ("freeze", r"\b(freeze|frozen|frost)\b"),
    ("silence", r"\b(silence)\b"),
    ("equip", r"\b(equip|weapon)\b"),
    ("give_buff", r"\b(buff|give|gain attack|gain health)\b"),
    ("increase_cost", r"\b(costs? more|increase cost|tax)\b"),
    ("reduce_cost", r"\b(costs? less|reduce cost|discount)\b"),
    ("add_to_hand", r"\b(add .* to (your )?hand|generate)\b"),
    ("cast_spell", r"\b(cast|recast)\b"),
]

TARGET_PATTERNS = [
    ("minion", r"\bminion(s)?\b"),
    ("enemy_minion", r"\benemy minion(s)?\b"),
    ("friendly_minion", r"\bfriendly minion(s)?\b"),
    ("hero", r"\bhero\b"),
    ("all_enemies", r"\ball enemies\b"),
    ("all_minions", r"\ball minions\b"),
]


def parse_query_rule(text: str, *, query_id: str | None = None) -> dict[str, Any]:
    lowered = text.lower()
    query = {
        "query_id": query_id or _query_id(text),
        "text": text,
        "classes": _unique([*_find_names(text, CLASSES), *_class_hints(lowered)]),
        "card_types": _unique([*_infer_card_types(lowered), *_card_type_hints(lowered)]),
        "keywords": _find_names(text, KEYWORDS),
        "actions": _unique([*_find_patterns(lowered, ACTION_PATTERNS), *_action_hints(lowered)]),
        "targets": _find_patterns(lowered, TARGET_PATTERNS),
        "resources": _resource_hints(lowered),
        "spell_schools": _find_names(text, SPELL_SCHOOLS),
        "minion_types": _find_names(text, MINION_TYPES),
        "mechanic_tags": [],
        "constraints": [],
        "generated_roles": [],
        "generated_card_names": [],
        "related_card_names": _related_card_names(lowered),
        "generation_hints": {"visual_tags": _visual_tags(lowered)},
    }
    if "lifesteal" in lowered and "deal_damage" in query["actions"]:
        query["mechanic_tags"].append("lifesteal_damage")
    return query


def parse_query_llm(
    text: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
    query_id: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = _api_key_for_provider(provider)
    prompt = build_query_parse_prompt(text)
    raw = _generate_content(
        prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM query parser did not return a JSON object.")
    parsed.setdefault("query_id", query_id or _query_id(text))
    parsed.setdefault("text", text)
    return normalize_query(parsed)


def normalize_query(query: dict[str, Any]) -> dict[str, Any]:
    normalized = {"query_id": query.get("query_id") or _query_id(str(query.get("text", ""))), "text": query.get("text", "")}
    for field in [
        "classes",
        "card_types",
        "keywords",
        "actions",
        "targets",
        "resources",
        "spell_schools",
        "minion_types",
        "mechanic_tags",
        "constraints",
        "generated_roles",
        "generated_card_names",
        "related_card_names",
        "triggers",
        "conditions",
    ]:
        value = query.get(field) or []
        if isinstance(value, str):
            value = [value]
        normalized[field] = _unique(str(item).strip() for item in value if str(item).strip())
    if isinstance(query.get("generation_hints"), dict):
        normalized["generation_hints"] = query["generation_hints"]
    return normalized


def build_query_parse_prompt(text: str) -> str:
    return f"""You convert a user's Hearthstone artwork request into a structured retrieval query.
Return JSON only. Do not add markdown.

Allowed retrieval fields:
query_id, text, classes, card_types, keywords, actions, targets, resources, spell_schools, minion_types, mechanic_tags, constraints, generated_roles, generated_card_names, related_card_names, triggers, conditions

Optional non-retrieval field:
generation_hints.visual_tags

Allowed actions:
deal_damage, heal, gain_armor, summon, draw, discover, destroy, freeze, silence, equip, give_buff, transform, add_to_hand, other

Use canonical Hearthstone names when possible, for example Warlock, Mage, Spell, Minion, Lifesteal, Deathrattle, Fel, Fire. Do not put pure artwork descriptions into retrieval fields; put them under generation_hints.visual_tags only.
If the user references Hearthstone community memes or Chinese card names, infer the canonical English card family when clear. Examples: 暴怒者 means Rager, 熔岩暴怒者 means Magma Rager. Put these under related_card_names.

User request:
{text}

Return this schema:
{{
  "text": "{text}",
  "classes": [],
  "card_types": [],
  "keywords": [],
  "actions": [],
  "targets": [],
  "resources": [],
  "spell_schools": [],
  "minion_types": [],
  "mechanic_tags": [],
  "constraints": [],
  "generated_roles": [],
  "generated_card_names": [],
  "related_card_names": [],
  "triggers": [],
  "conditions": [],
  "generation_hints": {{"visual_tags": []}}
}}
"""


def _find_names(text: str, names: list[str]) -> list[str]:
    lowered = text.lower()
    return [name for name in names if re.search(rf"\b{re.escape(name.lower())}\b", lowered)]


def _infer_card_types(lowered: str) -> list[str]:
    card_types = _find_names(lowered, CARD_TYPES)
    minion_patterns = [
        r"\bminion (that|with|which|who)\b",
        r"\b(deathrattle|battlecry|taunt|lifesteal|rush|beast|demon|dragon|undead|mech|murloc|naga|pirate) minion\b",
    ]
    if any(re.search(pattern, lowered) for pattern in minion_patterns):
        card_types.append("Minion")
    return _unique(card_types)


def _find_patterns(lowered: str, patterns: list[tuple[str, str]]) -> list[str]:
    return [label for label, pattern in patterns if re.search(pattern, lowered)]


def _visual_tags(lowered: str) -> list[str]:
    tags = []
    for tag in ["dark magic", "fel magic", "holy light", "fire magic", "frost magic", "nature magic", "arcane magic", "healing energy"]:
        if all(part in lowered for part in tag.split()):
            tags.append(tag)
    if "dark" in lowered and "dark magic" not in tags:
        tags.append("dark magic")
    if "fel" in lowered and "fel magic" not in tags:
        tags.append("fel magic")
    return tags


def _related_card_names(lowered: str) -> list[str]:
    names = []
    if "暴怒者" in lowered or "rager" in lowered:
        names.append("Rager")
    if "熔岩暴怒者" in lowered or "magma rager" in lowered:
        names.append("Magma Rager")
    return names


def _class_hints(lowered: str) -> list[str]:
    classes = []
    if "防战" in lowered or "战士" in lowered:
        classes.append("Warrior")
    return classes


def _card_type_hints(lowered: str) -> list[str]:
    card_types = []
    if "暴怒者" in lowered or "rager" in lowered:
        card_types.append("Minion")
    return card_types


def _action_hints(lowered: str) -> list[str]:
    actions = []
    if "护甲" in lowered or "防战" in lowered:
        actions.append("gain_armor")
    return actions


def _resource_hints(lowered: str) -> list[str]:
    resources = []
    if "护甲" in lowered or "armor" in lowered or "armour" in lowered or "防战" in lowered:
        resources.append("armor")
    return resources


def _query_id(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value[:80] or "query"


def _unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
