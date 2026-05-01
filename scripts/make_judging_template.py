#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.evaluation import read_result_files, write_judging_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV template for human retrieval relevance judging.")
    parser.add_argument("--inputs", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/judging_template.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_result_files(args.inputs)
    write_judging_template(rows, args.out)
    print("Done.")
    print(f"rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
