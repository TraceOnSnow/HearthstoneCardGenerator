#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import write_jsonl
from app.retrieval.clip_baseline import DEFAULT_CLIP_MODEL, retrieve_clip
from app.retrieval.common import load_caption_corpus, load_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP text-to-image nearest-neighbor retrieval baseline.")
    parser.add_argument("--queries", type=Path, default=Path("configs/retrieval_queries.json"))
    parser.add_argument("--captions", type=Path, default=Path("data/semantics/lora_captions.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--cache", type=Path, default=Path("results/retrieval_eval/clip_image_cache.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/clip_results.jsonl"))
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional corpus limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", help="cuda or cpu. Defaults automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    corpus = load_caption_corpus(args.captions)
    rows = retrieve_clip(
        corpus=corpus,
        queries=queries,
        image_root=args.image_root,
        cache_path=args.cache,
        top_k=args.top_k,
        model_name=args.model,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    write_jsonl(args.out, rows)
    print("Done.")
    print(f"queries={len(queries)}")
    print(f"corpus={len(corpus) if args.limit is None else min(len(corpus), args.limit)}")
    print(f"results={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
