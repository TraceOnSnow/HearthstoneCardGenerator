#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.comparison import write_generation_judging_template  # noqa: E402
from app.kg.io import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a human judging CSV for generated images.")
    parser.add_argument("--plan", type=Path, default=Path("results/generation_eval/generation_plan.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/generation_eval/generation_judging_template.csv"))
    args = parser.parse_args()
    write_generation_judging_template(read_jsonl(args.plan), args.out)
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
