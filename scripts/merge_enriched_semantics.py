#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple enriched semantics JSONL files by card_id.")
    parser.add_argument("--base", type=Path, required=True, help="Base JSONL that defines output order and card set.")
    parser.add_argument("--overlay", type=Path, action="append", required=True, help="Enriched JSONL overlay. Can be repeated.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base)
    overlays: dict[int, dict[str, Any]] = {}
    overlay_sources: dict[int, str] = {}

    for overlay_path in args.overlay:
        for row in read_jsonl(overlay_path):
            card_id = row.get("card_id")
            if not isinstance(card_id, int):
                continue
            if row.get("enrichment", {}).get("status") != "enriched":
                continue
            overlays[card_id] = row
            overlay_sources[card_id] = str(overlay_path)

    merged = [overlays.get(row.get("card_id"), row) for row in base_rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, merged)

    summary = {
        "base_cards": len(base_rows),
        "overlay_files": [str(path) for path in args.overlay],
        "enriched_cards": sum(1 for row in merged if row.get("enrichment", {}).get("status") == "enriched"),
        "base_only_cards": sum(1 for row in merged if row.get("enrichment", {}).get("status") != "enriched"),
        "overlay_sources": overlay_sources,
        "out": str(args.out),
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    for key, value in summary.items():
        if key == "overlay_sources":
            continue
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
