#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "query_id",
    "method",
    "rank",
    "card_id",
    "card_name",
    "image",
    "query_text",
    "class_match",
    "action_match",
    "keyword_match",
    "overall_relevance",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV template for human retrieval judging.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/judging_template.csv"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def template_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row.get("query_id", ""),
        "method": row.get("method", ""),
        "rank": row.get("rank", ""),
        "card_id": row.get("card_id", ""),
        "card_name": row.get("card_name", ""),
        "image": row.get("image", ""),
        "query_text": row.get("query_text", ""),
        "class_match": "",
        "action_match": "",
        "keyword_match": "",
        "overall_relevance": "",
        "comment": "",
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(template_row(row) for row in read_jsonl(path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
