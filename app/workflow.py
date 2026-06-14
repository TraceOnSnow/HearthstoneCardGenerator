from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.card_design.kg_designer import design_card_from_kg


DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}

DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, card frame, UI, typography, blurry, low quality, cropped"


@dataclass(frozen=True)
class GenerateOptions:
    request_text: str
    out_dir: Path
    card_index_path: Path = Path("data/semantic_kg/card_index.jsonl")
    semantics_path: Path = Path("data/semantics_enriched_current/cards_semantics_enriched.jsonl")
    top_k: int = 8
    parse_with_llm: bool = False
    provider: str = "minimax"
    model: str | None = None
    temperature: float = 0.3
    timeout_seconds: int = 180
    query_id: str | None = None
    mock_design: bool = False
    image_provider: str = "mock"
    seed: int = 42
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_dir: Path = Path("models/sd15-hearthstone-lora")
    steps: int = 30
    guidance_scale: float = 7.5
    lora_scale: float = 1.0
    width: int = 512
    height: int = 512


def run_generate(options: GenerateOptions) -> dict[str, Any]:
    out_dir = _resolve_out_dir(options)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _mock_design(options) if options.mock_design else _real_design(options)
    design = result["design"]
    card = design.get("card", {}) if isinstance(design, dict) else {}
    art_prompt = _art_prompt(result)

    art_path: Path | None = None
    if options.image_provider == "mock":
        art_path = out_dir / "art.png"
        _write_mock_art(art_path, art_prompt=art_prompt, card=card)
    elif options.image_provider == "lora":
        art_path = out_dir / "art.png"
        _write_lora_art(art_path, art_prompt=art_prompt, options=options)
    elif options.image_provider == "none":
        art_path = None
    else:
        raise ValueError(f"Unsupported image provider: {options.image_provider}")

    final_card_path: Path | None = None
    if art_path is not None:
        final_card_path = out_dir / "final_card.png"
        _compose_card(final_card_path, card=card, art_path=art_path)

    artifacts = {
        "out_dir": out_dir.as_posix(),
        "input": (out_dir / "input.json").as_posix(),
        "query": (out_dir / "query.json").as_posix(),
        "retrieved_cards": (out_dir / "retrieved_cards.json").as_posix(),
        "card": (out_dir / "card.json").as_posix(),
        "design": (out_dir / "design.json").as_posix(),
        "art_prompt": (out_dir / "art_prompt.txt").as_posix(),
        "art": art_path.as_posix() if art_path else None,
        "final_card": final_card_path.as_posix() if final_card_path else None,
        "run": (out_dir / "run.json").as_posix(),
    }
    run = {
        "request": options.request_text,
        "mock_design": options.mock_design,
        "image_provider": options.image_provider,
        "artifacts": artifacts,
    }

    _write_json(out_dir / "input.json", {"request": options.request_text})
    _write_json(out_dir / "query.json", result.get("query", {}))
    _write_json(out_dir / "retrieved_cards.json", result.get("retrieval_results", []))
    _write_json(out_dir / "evidence_package.json", result.get("evidence_package", {}))
    _write_json(out_dir / "card.json", card)
    _write_json(out_dir / "design.json", design)
    (out_dir / "art_prompt.txt").write_text(art_prompt + "\n", encoding="utf-8")
    _write_json(out_dir / "run.json", run)
    (out_dir / "summary.md").write_text(_summary_markdown(options.request_text, card, artifacts), encoding="utf-8")

    return {
        "out_dir": out_dir,
        "card": card,
        "artifacts": artifacts,
    }


def _real_design(options: GenerateOptions) -> dict[str, Any]:
    return design_card_from_kg(
        request_text=options.request_text,
        card_index_path=options.card_index_path,
        semantics_path=options.semantics_path,
        top_k=options.top_k,
        parse_with_llm=options.parse_with_llm,
        provider=options.provider,
        model=options.model or DEFAULT_MODELS[options.provider],
        temperature=options.temperature,
        timeout_seconds=options.timeout_seconds,
        query_id=options.query_id,
    )


def _mock_design(options: GenerateOptions) -> dict[str, Any]:
    from app.semantic_kg.query_parser import parse_query_rule
    from app.semantic_kg.retrieval import retrieve_one
    from app.kg.io import read_jsonl
    from app.card_design.kg_designer import build_evidence_package

    query = parse_query_rule(options.request_text, query_id=options.query_id)
    cards = read_jsonl(options.card_index_path)
    retrieval_results = retrieve_one(cards, query=query, top_k=options.top_k, require_image=False)
    evidence = [
        {
            "rank": row.get("rank"),
            "score": row.get("score"),
            "reasons": row.get("reasons", []),
            "card_id": row.get("card_id"),
            "name": row.get("card_name"),
        }
        for row in retrieval_results
    ]
    card_type = (query.get("card_types") or ["Minion"])[0]
    classes = query.get("classes") or ["Neutral"]
    is_minion = card_type == "Minion"
    actions = query.get("actions") or ["discover"]
    rules_text = _mock_rules_text(actions)
    design = {
        "card": {
            "name": _mock_card_name(query),
            "mana_cost": 4,
            "card_type": card_type,
            "class": classes[:1],
            "rarity": "Rare",
            "minion_type": (query.get("minion_types") or [None])[0],
            "attack": 3 if is_minion else None,
            "health": 5 if is_minion else None,
            "durability": 2 if card_type == "Weapon" else None,
            "keywords": query.get("keywords", [])[:2],
            "rules_text": rules_text,
            "flavor_text": "A local mock design for testing the full workflow.",
        },
        "structured_semantics": {
            "actions": actions,
            "mechanic_tags": query.get("mechanic_tags", []),
            "constraints": query.get("constraints", []),
            "related_card_refs": query.get("related_card_names", []),
            "semantic_summary": f"A {classes[0]} {card_type} focused on {', '.join(actions[:2])}.",
            "lora_caption": _caption_from_query(options.request_text, query),
        },
        "kg_usage": {
            "retrieved_cards_used": [{"card_id": row.get("card_id"), "name": row.get("card_name"), "used_for": "mock context"} for row in retrieval_results[:3]],
            "design_rationale": "Mock card generated without an external LLM.",
        },
        "balance_notes": "Mock balance note.",
    }
    return {
        "request": options.request_text,
        "query": query,
        "retrieval_results": retrieval_results,
        "evidence": evidence,
        "evidence_package": build_evidence_package(query=query, evidence=evidence),
        "design": design,
        "raw_response": json.dumps(design, ensure_ascii=False),
    }


def _resolve_out_dir(options: GenerateOptions) -> Path:
    if options.out_dir.name:
        return options.out_dir
    slug = _slug(options.query_id or options.request_text)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{slug}_{stamp}"


def _art_prompt(result: dict[str, Any]) -> str:
    design = result.get("design", {})
    structured = design.get("structured_semantics", {}) if isinstance(design, dict) else {}
    caption = structured.get("lora_caption") if isinstance(structured, dict) else ""
    if caption:
        return str(caption)
    card = design.get("card", {}) if isinstance(design, dict) else {}
    parts = ["Hearthstone card art", card.get("name"), card.get("card_type")]
    classes = card.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    parts.extend(classes)
    parts.append(card.get("minion_type"))
    return ", ".join(str(part) for part in parts if part)


def _write_lora_art(path: Path, *, art_prompt: str, options: GenerateOptions) -> None:
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit("Missing diffusion dependencies. Run `uv sync --extra diffusion` before using --image-provider lora.") from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        options.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_lora_weights(str(options.lora_dir))
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=pipe.device).manual_seed(options.seed)
    image = pipe(
        prompt=art_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=options.steps,
        guidance_scale=options.guidance_scale,
        width=options.width,
        height=options.height,
        generator=generator,
        cross_attention_kwargs={"scale": options.lora_scale},
    ).images[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_mock_art(path: Path, *, art_prompt: str, card: dict[str, Any]) -> None:
    palette = _class_palette(card)
    image = Image.new("RGB", (512, 512), palette["bg"])
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _fonts(34, 22, 16)
    draw.rounded_rectangle([18, 18, 494, 494], radius=34, outline=palette["fg"], width=5)
    for idx, color in enumerate(palette["bands"]):
        draw.ellipse([80 + idx * 70, 100 - idx * 12, 360 + idx * 40, 390 + idx * 18], fill=color)
    draw.text((42, 42), str(card.get("name") or "Custom Card")[:24], fill=palette["fg"], font=title_font)
    y = 354
    for line in _wrap(art_prompt, 42)[:4]:
        draw.text((42, y), line, fill=palette["fg"], font=small_font)
        y += 22
    draw.text((42, 458), "mock art", fill=palette["fg"], font=body_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _compose_card(path: Path, *, card: dict[str, Any], art_path: Path) -> None:
    width, height = 744, 1038
    palette = _class_palette(card)
    canvas = Image.new("RGB", (width, height), palette["outer"])
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = _fonts(38, 28, 22)
    rules_font = _font(25)
    draw.rounded_rectangle([26, 24, width - 26, height - 24], radius=52, fill=palette["frame"], outline="#2b1b0d", width=8)
    draw.rounded_rectangle([74, 82, width - 74, 172], radius=26, fill="#ead8a8", outline="#4b351f", width=4)
    draw.text((106, 104), str(card.get("name") or "Custom Card")[:28], fill="#27170c", font=title_font)
    _draw_gem(draw, center=(72, 78), text=str(card.get("mana_cost", "?")), fill="#2d72d9", font=title_font)

    art = Image.open(art_path).convert("RGB").resize((560, 420))
    canvas.paste(art, (92, 190))
    draw.rounded_rectangle([86, 184, 658, 616], radius=24, outline="#3b2614", width=7)

    type_line = _type_line(card)
    draw.rounded_rectangle([92, 642, 652, 708], radius=18, fill="#d7c083", outline="#4b351f", width=4)
    draw.text((118, 659), type_line[:38], fill="#27170c", font=body_font)

    draw.rounded_rectangle([94, 734, 650, 910], radius=18, fill="#eee1c0", outline="#4b351f", width=4)
    y = 760
    for line in _wrap(str(card.get("rules_text") or ""), 34)[:5]:
        draw.text((124, y), line, fill="#27170c", font=rules_font)
        y += 33
    flavor = str(card.get("flavor_text") or "")
    if flavor:
        for line in _wrap(flavor, 42)[:2]:
            draw.text((124, y + 6), line, fill="#6d5637", font=small_font)
            y += 27

    if card.get("attack") is not None:
        _draw_gem(draw, center=(82, 944), text=str(card.get("attack")), fill="#d98b2d", font=title_font)
    if card.get("health") is not None:
        _draw_gem(draw, center=(662, 944), text=str(card.get("health")), fill="#c64035", font=title_font)
    elif card.get("durability") is not None:
        _draw_gem(draw, center=(662, 944), text=str(card.get("durability")), fill="#7c9bb8", font=title_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _draw_gem(draw: ImageDraw.ImageDraw, *, center: tuple[int, int], text: str, fill: str, font: ImageFont.ImageFont) -> None:
    x, y = center
    draw.ellipse([x - 48, y - 48, x + 48, y + 48], fill=fill, outline="#1c1208", width=5)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 2), text, fill="white", font=font)


def _type_line(card: dict[str, Any]) -> str:
    card_type = str(card.get("card_type") or "Card")
    minion_type = card.get("minion_type")
    rarity = card.get("rarity")
    parts = [card_type]
    if minion_type:
        parts.append(str(minion_type))
    if rarity:
        parts.append(str(rarity))
    return " - ".join(parts)


def _class_palette(card: dict[str, Any]) -> dict[str, Any]:
    classes = card.get("class") or ["Neutral"]
    if isinstance(classes, str):
        classes = [classes]
    cls = str(classes[0] if classes else "Neutral")
    palettes = {
        "Mage": {"outer": "#163c63", "frame": "#8fb9d6", "bg": "#b8d7ea", "fg": "#18334a", "bands": ["#76a9d6", "#d7efff", "#4984b3"]},
        "Warlock": {"outer": "#321f3f", "frame": "#9d7aa8", "bg": "#d1bad6", "fg": "#2e1b38", "bands": ["#9d6aba", "#47325a", "#c8a6d4"]},
        "Warrior": {"outer": "#55331d", "frame": "#c28a58", "bg": "#ddb58a", "fg": "#3d2415", "bands": ["#c06b3e", "#8c3d2d", "#e3a86b"]},
        "Druid": {"outer": "#31471f", "frame": "#a0b96f", "bg": "#c9daa4", "fg": "#263719", "bands": ["#6f9c4c", "#d0d88a", "#496d35"]},
        "Paladin": {"outer": "#6d5b24", "frame": "#d6c26f", "bg": "#efe4aa", "fg": "#4b3c13", "bands": ["#ead071", "#fff3ba", "#b99c34"]},
        "Rogue": {"outer": "#30302d", "frame": "#aaa064", "bg": "#d8d0a2", "fg": "#29271d", "bands": ["#6f7048", "#c4bd7b", "#41412f"]},
        "Priest": {"outer": "#4a4359", "frame": "#c8c1d7", "bg": "#e5e0ef", "fg": "#332d40", "bands": ["#cfc7ea", "#ffffff", "#8f82b8"]},
        "Hunter": {"outer": "#324521", "frame": "#91ad61", "bg": "#c5d9a1", "fg": "#26341a", "bands": ["#69904b", "#bacf7a", "#405d31"]},
        "Shaman": {"outer": "#1d4450", "frame": "#7eb5bd", "bg": "#b8dce0", "fg": "#19383f", "bands": ["#4c98a3", "#bedb8d", "#2f6f78"]},
    }
    return palettes.get(cls, {"outer": "#4a3824", "frame": "#b99b6b", "bg": "#d9c7a9", "fg": "#2f251a", "bands": ["#b98e55", "#e2c98f", "#7e6041"]})


def _mock_card_name(query: dict[str, Any]) -> str:
    classes = query.get("classes") or ["Neutral"]
    minion_types = query.get("minion_types") or []
    actions = query.get("actions") or ["Spark"]
    noun = minion_types[0] if minion_types else "Adept"
    action = str(actions[0]).replace("_", " ").title().replace(" ", "")
    return f"{classes[0]} {action} {noun}"


def _mock_rules_text(actions: list[str]) -> str:
    if "discover" in actions:
        return "Battlecry: Discover a spell. It costs (1) less."
    if "gain_armor" in actions:
        return "Battlecry: Gain 3 Armor. If you control a damaged minion, gain +2/+2."
    if "summon" in actions:
        return "Battlecry: Summon two 1/1 recruits."
    if "deal_damage" in actions:
        return "Battlecry: Deal 2 damage. If this kills a minion, draw a card."
    return "Battlecry: Add a random class card to your hand."


def _caption_from_query(request_text: str, query: dict[str, Any]) -> str:
    parts = ["Hearthstone card art"]
    for field in ["classes", "card_types", "spell_schools", "minion_types", "keywords", "actions", "related_card_names"]:
        parts.extend(query.get(field, [])[:3])
    parts.extend((query.get("generation_hints") or {}).get("visual_tags", [])[:4])
    parts.append(request_text)
    return ", ".join(str(part) for part in parts if part)


def _summary_markdown(request_text: str, card: dict[str, Any], artifacts: dict[str, Any]) -> str:
    lines = [
        "# HearthGen Run",
        "",
        f"Request: {request_text}",
        "",
        "## Card",
        "",
        f"- Name: {card.get('name')}",
        f"- Type: {_type_line(card)}",
        f"- Cost: {card.get('mana_cost')}",
        f"- Stats: {card.get('attack')}/{card.get('health')}",
        f"- Text: {card.get('rules_text')}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in artifacts.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:48] or "card"


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _fonts(title_size: int, body_size: int, small_size: int) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    return _font(title_size), _font(body_size), _font(small_size)


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
