from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_lora_rows(
    *,
    metadata_path: Path,
    image_root: Path | None,
    caption_column: str = "text",
    image_column: str = "file_name",
    trigger_token: str = "hsart",
    limit: int | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Return trainable image/caption rows and missing-image diagnostics.

    The repo can produce two useful JSONL shapes:
    - data/semantics/lora_captions.jsonl: image + caption
    - data/hf_hearthstone_art_512/metadata.jsonl: file_name + text/card metadata
    This normalizer accepts both, resolves relative image paths, and builds a
    simple fallback caption when a row does not already contain one.
    """
    rows = read_jsonl(metadata_path)
    base_dirs = _base_dirs(metadata_path=metadata_path, image_root=image_root)
    normalized: list[dict[str, str]] = []
    missing_images: list[str] = []

    for row in rows:
        image_value = _first_text(row, [image_column, "image", "file_name"])
        if not image_value:
            missing_images.append("<missing image field>")
            continue

        image_path = resolve_image_path(image_value, base_dirs)
        if image_path is None:
            missing_images.append(image_value)
            continue

        caption = build_caption(row, caption_column=caption_column, trigger_token=trigger_token)
        if not caption:
            continue

        normalized.append({"image_path": str(image_path), "caption": caption})
        if limit is not None and len(normalized) >= limit:
            break

    return normalized, missing_images


def resolve_image_path(image_value: str, base_dirs: list[Path]) -> Path | None:
    path = Path(image_value)
    if path.is_absolute():
        return path if path.exists() else None

    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return None


def build_caption(row: dict[str, Any], *, caption_column: str, trigger_token: str) -> str:
    caption = _first_text(row, [caption_column, "text", "caption", "lora_caption"])
    if not caption:
        caption = _fallback_caption(row)

    caption = caption.strip()
    trigger_token = trigger_token.strip()
    if trigger_token and not caption.lower().startswith(trigger_token.lower()):
        caption = f"{trigger_token} {caption}"
    return caption


def _base_dirs(*, metadata_path: Path, image_root: Path | None) -> list[Path]:
    base_dirs: list[Path] = []
    if image_root is not None:
        base_dirs.append(image_root)
    base_dirs.append(metadata_path.parent)
    return [path.resolve() for path in base_dirs]


def _first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _fallback_caption(row: dict[str, Any]) -> str:
    parts = [
        "Hearthstone card art",
        row.get("name"),
        row.get("card_class"),
        row.get("type"),
        row.get("set"),
        row.get("artist"),
    ]
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in parts:
        text = str(part).strip() if part is not None else ""
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return ", ".join(cleaned)
