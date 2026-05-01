#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.evaluation import summarize_judging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize filled human judging CSV by retrieval method.")
    parser.add_argument("--input", type=Path, default=Path("results/retrieval_eval/judging_template.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_judging(args.input, args.out)
    print("Done.")
    print(f"methods={len(rows)}")
    print(f"out={args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
