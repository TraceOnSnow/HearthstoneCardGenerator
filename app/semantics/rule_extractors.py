from __future__ import annotations

import re
from typing import Any

from app.semantics.text import slugify_label


NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

CLASS_VISUAL_TAGS = {
    "death knight": ["death knight", "runes", "undead frost magic"],
    "demon hunter": ["demon hunter", "fel blades", "agile fighter"],
    "druid": ["druid", "nature magic", "wild growth"],
    "hunter": ["hunter", "beasts", "ranged combat"],
    "mage": ["mage", "arcane magic", "spell energy"],
    "paladin": ["paladin", "holy light", "armored champion"],
    "priest": ["priest", "holy magic", "spiritual energy"],
    "rogue": ["rogue", "shadowy assassin", "daggers"],
    "shaman": ["shaman", "elemental magic", "totems"],
    "warlock": ["warlock", "fel magic", "dark ritual"],
    "warrior": ["warrior", "armor", "weapon combat"],
}

SCHOOL_VISUAL_TAGS = {
    "arcane": ["arcane spell", "purple magical energy"],
    "fel": ["fel magic", "green demonic glow"],
    "fire": ["fire magic", "flames"],
    "frost": ["frost magic", "ice shards"],
    "holy": ["holy light", "radiant magic"],
    "nature": ["nature magic", "green growth"],
    "shadow": ["shadow magic", "dark energy"],
}

TRIBE_VISUAL_TAGS = {
    "beast": ["beast creature", "animal monster"],
    "demon": ["demonic creature", "fel glow"],
    "dragon": ["dragon", "draconic fantasy"],
    "elemental": ["elemental creature", "primal energy"],
    "mech": ["mechanical construct", "metal machine"],
    "murloc": ["murloc", "amphibious creature"],
    "naga": ["naga", "serpentine creature"],
    "pirate": ["pirate", "swashbuckler"],
    "totem": ["totem", "shamanic idol"],
    "undead": ["undead", "necromancy"],
}


def extract_actions(clean_text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not clean_text:
        return actions

    for sentence in _sentences(clean_text):
        lowered = sentence.lower()
        trigger = _trigger_for_sentence(lowered)

        damage = re.search(r"deal(?:s)? (\d+) damage(?: to (.*?))?(?:\.|$|,|;)", sentence, re.I)
        if damage:
            actions.append(
                _action(
                    "deal_damage",
                    sentence,
                    amount=int(damage.group(1)),
                    target=_normalize_target(damage.group(2) or "target"),
                    resource="health",
                    trigger=trigger,
                )
            )

        heal = re.search(r"(?:restore|restores) (\d+) health(?: to (.*?))?(?:\.|$|,|;)", sentence, re.I)
        if heal:
            actions.append(
                _action(
                    "heal",
                    sentence,
                    amount=int(heal.group(1)),
                    target=_normalize_target(heal.group(2) or "character"),
                    resource="health",
                    trigger=trigger,
                )
            )

        armor = re.search(r"gain(?:s)? (\d+) armor", sentence, re.I)
        if armor:
            actions.append(
                _action("gain_armor", sentence, amount=int(armor.group(1)), target="hero", resource="armor", trigger=trigger)
            )

        summon = re.search(r"summon(?:s)? (?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+) )?(.*?)(?:\.|$|,|;)", sentence, re.I)
        if summon:
            amount = _number(summon.group(1)) if summon.group(1) else None
            actions.append(_action("summon", sentence, amount=amount, target=_normalize_target(summon.group(2)), resource="board", trigger=trigger))

        draw = re.search(r"draw(?:s)? (?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+) )?card", sentence, re.I)
        if draw:
            actions.append(_action("draw", sentence, amount=_number(draw.group(1)) if draw.group(1) else 1, target="card", resource="hand", trigger=trigger))

        discover = re.search(r"discover(?:s)? (.*?)(?:\.|$|,|;)", sentence, re.I)
        if discover:
            actions.append(_action("discover", sentence, target=_normalize_target(discover.group(1)), resource="hand", trigger=trigger))

        add = re.search(r"add(?:s)? (.*?) to your hand", sentence, re.I)
        if add:
            actions.append(_action("add_to_hand", sentence, target=_normalize_target(add.group(1)), resource="hand", trigger=trigger))

        destroy = re.search(r"destroy(?:s)? (.*?)(?:\.|$|,|;)", sentence, re.I)
        if destroy:
            actions.append(_action("destroy", sentence, target=_normalize_target(destroy.group(1)), resource="board", trigger=trigger))

        if re.search(r"\bfreeze(?:s)?\b", lowered):
            actions.append(_action("freeze", sentence, target=_normalize_target(_after_word(sentence, "freeze")), resource="board", trigger=trigger))

        if re.search(r"\bsilence(?:s)?\b", lowered):
            actions.append(_action("silence", sentence, target=_normalize_target(_after_word(sentence, "silence")), resource="board", trigger=trigger))

        equip = re.search(r"equip(?:s)? (.*?)(?:\.|$|,|;)", sentence, re.I)
        if equip:
            actions.append(_action("equip", sentence, target=_normalize_target(equip.group(1)), resource="weapon", trigger=trigger))

    return _dedupe_actions(actions)


def infer_mechanic_tags(card: dict[str, Any], actions: list[dict[str, Any]], keywords: list[str]) -> list[str]:
    tags: list[str] = []
    action_types = {action["type"] for action in actions}
    keyword_slugs = {slugify_label(keyword) for keyword in keywords}

    for action_type in sorted(action_types):
        tags.append(action_type)
    if "lifesteal" in keyword_slugs and "deal_damage" in action_types:
        tags.append("lifesteal_damage")
    if "summon" in action_types:
        tags.append("token_generation")
    if "draw" in action_types:
        tags.append("card_draw")
    if "gain_armor" in action_types:
        tags.append("armor_synergy")
    if card.get("childIds"):
        tags.append("has_derived_cards")
    return _unique(tags)


def infer_visual_tags(identity: dict[str, Any], actions: list[dict[str, Any]], keywords: list[str]) -> list[str]:
    tags: list[str] = ["hearthstone fantasy art"]
    for class_name in identity.get("card_class", []):
        tags.extend(CLASS_VISUAL_TAGS.get(str(class_name).lower(), [str(class_name).lower()]))
    for field, mapping in [
        ("spell_school", SCHOOL_VISUAL_TAGS),
        ("minion_type", TRIBE_VISUAL_TAGS),
    ]:
        value = identity.get(field)
        if value:
            tags.extend(mapping.get(str(value).lower(), [str(value).lower()]))

    action_types = {action["type"] for action in actions}
    if "deal_damage" in action_types:
        tags.append("magical impact")
    if "heal" in action_types:
        tags.append("healing energy")
    if "summon" in action_types:
        tags.append("summoned creatures")
    if "gain_armor" in action_types:
        tags.append("defensive armor")
    if "freeze" in action_types:
        tags.append("ice magic")
    if "destroy" in action_types:
        tags.append("destructive magic")
    for keyword in keywords:
        if keyword.lower() in {"lifesteal", "deathrattle", "battlecry", "taunt", "divine shield"}:
            tags.append(keyword.lower())
    return _unique(tags)


def _action(action_type: str, raw_phrase: str, **fields: Any) -> dict[str, Any]:
    row = {
        "type": action_type,
        "amount": fields.get("amount"),
        "target": fields.get("target") or None,
        "target_scope": fields.get("target_scope"),
        "resource": fields.get("resource"),
        "condition": fields.get("condition"),
        "trigger": fields.get("trigger") or "on_play",
        "duration": fields.get("duration"),
        "raw_phrase": raw_phrase.strip(),
    }
    return row


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _trigger_for_sentence(lowered: str) -> str:
    if "battlecry" in lowered:
        return "battlecry"
    if "deathrattle" in lowered:
        return "deathrattle"
    if "combo:" in lowered:
        return "combo"
    if "start of game" in lowered:
        return "start_of_game"
    if "end of your turn" in lowered:
        return "end_of_turn"
    return "on_play"


def _number(value: str | None) -> int | None:
    if not value:
        return None
    value = value.lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _normalize_target(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"\([^)]*\)", "", value).strip(" .")
    value = re.sub(r"^(a|an|the|your|another|random) ", "", value, flags=re.I)
    return slugify_label(value) or None


def _after_word(sentence: str, word: str) -> str:
    match = re.search(rf"{word}\s+(.*?)(?:\.|$|,|;)", sentence, re.I)
    return match.group(1) if match else "target"


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for action in actions:
        key = (action["type"], action.get("amount"), action.get("target"), action.get("raw_phrase"))
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result

