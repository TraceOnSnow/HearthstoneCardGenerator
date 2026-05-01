#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.evaluation import read_result_files, render_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render retrieval results as a simple side-by-side HTML grid.")
    parser.add_argument("--inputs", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/retrieval_grid.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_result_files(args.inputs)
    render_grid(rows, out_path=args.out, image_root=args.image_root)
    print("Done.")
    print(f"rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
