#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.semantics.builder import build_semantics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based structured semantics for all Hearthstone cards.")
    parser.add_argument("--cards", type=Path, default=Path("data/cards_all.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/hearthstone_metadata.json"))
    parser.add_argument("--art-metadata", type=Path, default=Path("data/hf_hearthstone_art_512/metadata.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/semantics"))
    parser.add_argument(
        "--include-special-modes",
        action="store_true",
        help="Keep Battlegrounds, Mercenaries, hero skins, and other special-mode cards.",
    )
    parser.add_argument("--limit", type=int, help="Build only first N cards for a quick test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = args.metadata if args.metadata.exists() else None
    art_metadata = args.art_metadata if args.art_metadata.exists() else None
    if metadata is None:
        print(f"Warning: metadata not found at {args.metadata}; numeric IDs will be used as fallback names.")
    if art_metadata is None:
        print(f"Warning: art metadata not found at {args.art_metadata}; lora captions will not include image paths.")
    stats = build_semantics(
        cards_path=args.cards,
        metadata_path=metadata,
        art_metadata_path=art_metadata,
        exclude_special_modes=not args.include_special_modes,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print("Done.")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
