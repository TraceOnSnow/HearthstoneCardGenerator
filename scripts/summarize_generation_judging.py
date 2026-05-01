#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FIELDS = ["prompt_alignment", "hearthstone_style", "reference_consistency", "overall_quality"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize filled generation judging CSV by method.")
    parser.add_argument("--input", type=Path, default=Path("results/generation_eval/generation_judging_template.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/generation_eval/generation_summary.csv"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)

    summary = []
    for method, method_rows in sorted(grouped.items()):
        item = {"method": method, "rows": len(method_rows)}
        for field in FIELDS:
            item[f"{field}_mean"] = _mean(method_rows, field)
        summary.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["method"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"out={args.out}")


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


if __name__ == "__main__":
    main()
