#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LoRA-ready HF metadata from structured semantics.")
    parser.add_argument("--captions", type=Path, default=Path("data/semantics/lora_captions.jsonl"))
    parser.add_argument("--semantics", type=Path, default=Path("data/semantics/cards_semantics_base.jsonl"))
    parser.add_argument("--derived-edges", type=Path, default=Path("data/semantics/derived_edges.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/semantics/summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/lora_hf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    caption_rows = read_jsonl(args.captions)
    lora_rows = [_to_lora_row(row) for row in caption_rows if row.get("image") and row.get("caption")]
    lora_rows.sort(key=lambda row: (str(row["file_name"]), int(row["card_id"])))

    write_jsonl(args.out_dir / "metadata.jsonl", lora_rows)
    _copy_if_exists(args.semantics, args.out_dir / "cards_semantics_base.jsonl")
    _copy_if_exists(args.derived_edges, args.out_dir / "derived_edges.jsonl")
    _copy_if_exists(args.summary, args.out_dir / "summary.json")
    _write_readme(args.out_dir, row_count=len(lora_rows))

    print(f"Prepared {args.out_dir}")
    print(f"metadata_rows={len(lora_rows)}")
    print(f"semantics={args.out_dir / 'cards_semantics_base.jsonl'}")
    print(f"derived_edges={args.out_dir / 'derived_edges.jsonl'}")


def _to_lora_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": row["image"],
        "text": row["caption"],
        "card_id": row.get("card_id"),
        "name": row.get("name"),
        "slug": row.get("slug"),
        "collectible": row.get("collectible", False),
        "root_collectible_ids": row.get("root_collectible_ids", []),
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def _write_readme(out_dir: Path, *, row_count: int) -> None:
    readme = f"""# LoRA Training Metadata

This folder contains LoRA-ready metadata derived from the structured Hearthstone
semantics pipeline.

## Files

- `metadata.jsonl`: one row per trainable image with `file_name` and `text`.
- `cards_semantics_base.jsonl`: full non-special-mode structured semantics.
- `derived_edges.jsonl`: parent/child card links after special-mode filtering.
- `summary.json`: generation statistics.

## Training Rows

`metadata.jsonl` contains {row_count} rows. Each row points to an image already
stored in the dataset root `images/` folder.

## Notes

Battlegrounds, Mercenaries, hero skins, and other special-mode cards are excluded
from the structured semantics used here.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

