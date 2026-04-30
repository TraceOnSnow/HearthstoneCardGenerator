from __future__ import annotations

import html
import re


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_card_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", ". ", text, flags=re.IGNORECASE)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def slugify_label(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")

