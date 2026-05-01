#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import write_jsonl
from app.retrieval.common import load_caption_corpus, load_queries
from app.retrieval.tfidf import retrieve_tfidf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF retrieval baseline over LoRA captions.")
    parser.add_argument("--queries", type=Path, default=Path("configs/retrieval_queries.json"))
    parser.add_argument("--captions", type=Path, default=Path("data/semantics/lora_captions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/tfidf_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    corpus = load_caption_corpus(args.captions)
    rows = retrieve_tfidf(corpus=corpus, queries=queries, top_k=args.top_k)
    write_jsonl(args.out, rows)
    print("Done.")
    print(f"queries={len(queries)}")
    print(f"corpus={len(corpus)}")
    print(f"results={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
