import argparse
import html
import json
import re
from pathlib import Path

from PIL import Image

from crop_cards import CARD_TYPE_TO_MASK_NAME, extract_art_with_mask


CLASS_MAP = {
    1: "death_knight",
    2: "druid",
    3: "hunter",
    4: "mage",
    5: "paladin",
    6: "priest",
    7: "rogue",
    8: "shaman",
    9: "warlock",
    10: "warrior",
    12: "neutral",
    14: "demon_hunter",
}

CARD_TYPE_MAP = {
    4: "minion",
    5: "spell",
}

RACE_MAP = {
    2: "draenei",
    11: "undead",
    14: "murloc",
    15: "demon",
    17: "mech",
    18: "elemental",
    20: "beast",
    21: "totem",
    23: "pirate",
    24: "dragon",
    92: "naga",
}

SPELL_SCHOOL_MAP = {
    1: "arcane",
    2: "fel",
    4: "fire",
    5: "holy",
    6: "shadow",
    7: "nature",
}

KEYWORD_MAP = {
    1: "taunt",
    3: "divine_shield",
    4: "stealth",
    5: "secret",
    6: "windfury",
    8: "battlecry",
    10: "freeze",
    11: "charge",
    12: "deathrattle",
    13: "combo",
    14: "overload",
    15: "silence",
    17: "spell_damage",
    21: "discover",
    32: "poisonous",
    38: "lifesteal",
    53: "rush",
    64: "adapt",
    77: "magnetic",
    78: "echo",
    86: "twinspell",
    97: "reborn",
    247: "tradable",
    256: "honorable_kill",
    265: "infuse",
    266: "colossal",
    270: "manathirst",
    297: "quickdraw",
    298: "forge",
}

MECHANIC_PATTERNS = [
    ("aoe", r"\ball\b"),
    ("summon", r"\bsummon\b"),
    ("shuffle_into_deck", r"\bshuffle\b.*\bdeck\b"),
    ("discover", r"\bdiscover\b"),
    ("draw", r"\bdraw\b"),
    ("freeze", r"\bfreeze\b"),
    ("transform", r"\btransform\b"),
    ("copy", r"\bcopy\b"),
    ("buff", r"\bgive\b|\bgrant\b"),
    ("cost_reduction", r"\bcosts?\b.*\bless\b|\breduce\b.*\bcost\b"),
    ("damage", r"\bdeal\b.*\bdamage\b"),
    ("heal", r"\brestore\b|\bheal\b"),
    ("destroy", r"\bdestroy\b"),
    ("resurrect", r"\bresummon\b|\breturn\b.*\bdied\b"),
]

STYLE_HINTS_BY_CLASS = {
    "death_knight": ["undead", "cold", "dark"],
    "druid": ["nature", "green", "wild"],
    "hunter": ["beast", "rugged", "outdoors"],
    "mage": ["arcane", "blue", "glowing"],
    "paladin": ["holy", "gold", "radiant"],
    "priest": ["holy", "soft_light", "mystic"],
    "rogue": ["shadow", "sleek", "stealthy"],
    "shaman": ["elemental", "storm", "totemic"],
    "warlock": ["fel", "purple", "ominous"],
    "warrior": ["armor", "battle", "dramatic"],
    "neutral": ["fantasy", "hearthstone_style"],
    "demon_hunter": ["fel", "agile", "green_glow"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a cropped Hearthstone image subset plus JSON captions for LoRA training."
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--jsonl", type=Path, default=Path("data/cards_collectible.jsonl"))
    parser.add_argument("--card-images-dir", type=Path, default=Path("data/card_images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/sample_img/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/lora_subset_200/train"))
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--alpha-threshold", type=int, default=1)
    return parser.parse_args()


def strip_tags(text: str) -> str:
    unescaped = html.unescape(text or "")
    no_tags = re.sub(r"<[^>]+>", "", unescaped)
    cleaned = re.sub(r"\s+", " ", no_tags).strip()
    return cleaned


def extract_keywords(card: dict) -> list[str]:
    found: list[str] = []
    for keyword_id in card.get("keywordIds") or []:
        keyword = KEYWORD_MAP.get(keyword_id)
        if keyword and keyword not in found:
            found.append(keyword)

    for match in re.findall(r"<b>(.*?)</b>", card.get("text", ""), flags=re.IGNORECASE):
        normalized = re.sub(r"[^a-z0-9]+", "_", html.unescape(match).lower()).strip("_")
        if normalized and normalized not in found:
            found.append(normalized)

    return found


def extract_mechanics(description: str) -> list[str]:
    lowered = description.lower()
    mechanics: list[str] = []
    for name, pattern in MECHANIC_PATTERNS:
        if re.search(pattern, lowered) and name not in mechanics:
            mechanics.append(name)
    return mechanics


def build_style(card_class: str, spell_school: str | None, race: str | None) -> list[str]:
    style: list[str] = []
    for token in STYLE_HINTS_BY_CLASS.get(card_class, ["fantasy"]):
        if token not in style:
            style.append(token)
    if spell_school and spell_school not in style:
        style.append(spell_school)
    if race and race not in style:
        style.append(race)
    if "fantasy" not in style:
        style.append("fantasy")
    if "hearthstone_style" not in style:
        style.append("hearthstone_style")
    return style


def build_structured_payload(card: dict, image_path: Path) -> dict:
    card_class = CLASS_MAP.get(card.get("classId"), "unknown")
    card_type = CARD_TYPE_MAP.get(card.get("cardTypeId"), "unknown")
    race = RACE_MAP.get(card.get("minionTypeId"), "none")
    spell_school = SPELL_SCHOOL_MAP.get(card.get("spellSchoolId"))
    description = strip_tags(str(card.get("text", "")))
    keywords = extract_keywords(card)
    mechanics = extract_mechanics(description)
    style = build_style(card_class, spell_school, None if race == "none" else race)

    return {
        "structured_semantics": {
            "card_type": card_type,
            "class": card_class,
            "race": race,
            "keywords": keywords,
            "mechanics": mechanics,
            "description": description,
            "style": style,
        },
        "reference_images": [str(image_path).replace("\\", "/")],
    }


def load_cards_by_id(path: Path) -> dict[int, dict]:
    cards: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            card_id = raw.get("id")
            card_type = raw.get("cardTypeId")
            if isinstance(card_id, int) and card_type in CARD_TYPE_MAP:
                cards[card_id] = raw
    return cards


def load_masks(mask_dir: Path) -> dict[int, Image.Image]:
    masks: dict[int, Image.Image] = {}
    for type_id, mask_name in CARD_TYPE_TO_MASK_NAME.items():
        mask_path = mask_dir / f"{mask_name}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {mask_name}: {mask_path}")
        masks[type_id] = Image.open(mask_path).convert("RGBA")
    return masks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = [int(path.stem) for path in sorted(args.card_images_dir.glob("*.png"))[: args.limit]]
    if not selected_ids:
        raise RuntimeError(f"No source images found in {args.card_images_dir}")

    cards_by_id = load_cards_by_id(args.jsonl)
    masks = load_masks(args.mask_dir)

    metadata_rows: list[dict[str, str]] = []
    processed = 0

    for card_id in selected_ids:
        card = cards_by_id.get(card_id)
        if not card:
            continue

        source_path = args.card_images_dir / f"{card_id}.png"
        output_image = args.output_dir / f"{card_id}.png"
        output_text = args.output_dir / f"{card_id}.txt"

        card_img = Image.open(source_path).convert("RGBA")
        art = extract_art_with_mask(
            card_img=card_img,
            mask_img=masks[card["cardTypeId"]],
            alpha_threshold=args.alpha_threshold,
            add_padding=args.padding,
            keep_transparency=True,
            invert_mask=True,
            auto_fix_polarity=False,
        )
        art.save(output_image)

        payload = build_structured_payload(card, output_image)
        output_text.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata_rows.append({"file_name": output_image.name, "text": json.dumps(payload, ensure_ascii=False)})
        processed += 1

    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Prepared subset. requested={args.limit} processed={processed} output={args.output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
