#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.semantic_kg.build import build_semantic_kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic KG from structured Hearthstone semantics.")
    parser.add_argument("--semantics", type=Path, default=Path("data/semantics/cards_semantics_base.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/semantic_kg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_semantic_kg(semantics_path=args.semantics, out_dir=args.out_dir)
    print("Done.")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

