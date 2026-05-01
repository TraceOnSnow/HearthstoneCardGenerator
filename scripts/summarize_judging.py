#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SCORE_FIELDS = [
    "class_match",
    "action_match",
    "keyword_match",
    "overall_relevance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize filled retrieval judging CSV by method.")
    parser.add_argument("--input", type=Path, default=Path("results/retrieval_eval/judging_template.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/summary.csv"))
    return parser.parse_args()


def parse_score(value: str) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def main() -> None:
    args = parse_args()
    main_with_paths(args.input, args.out)


def main_with_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Judging CSV not found: {input_path}")

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if not method:
                continue
            for field in SCORE_FIELDS:
                score = parse_score(row.get(field, ""))
                if score is not None:
                    values[method][field].append(score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "class_match_at_5",
            "action_match_at_5",
            "keyword_match_at_5",
            "overall_relevance_at_5",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in sorted(values):
            writer.writerow(
                {
                    "method": method,
                    "class_match_at_5": average(values[method]["class_match"]),
                    "action_match_at_5": average(values[method]["action_match"]),
                    "keyword_match_at_5": average(values[method]["keyword_match"]),
                    "overall_relevance_at_5": average(values[method]["overall_relevance"]),
                }
            )
    print(f"Wrote summary to {output_path}")


def average(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


if __name__ == "__main__":
    main()
