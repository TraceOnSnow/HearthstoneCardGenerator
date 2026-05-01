#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import write_jsonl
from app.semantic_kg.query_parser import parse_query_llm, parse_query_rule
from app.semantic_kg.retrieval import load_structured_queries, retrieve_many


DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k cards from the semantic KG.")
    parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    parser.add_argument("--queries", type=Path, help="JSON file with structured retrieval queries.")
    parser.add_argument("--query-text", help="Natural-language query. Parsed by rules unless --parse-with-llm is set.")
    parser.add_argument("--query-id", help="Optional query id for --query-text.")
    parser.add_argument("--out", type=Path, default=Path("results/kg_retrieval/kg_results.jsonl"))
    parser.add_argument("--parsed-query-out", type=Path, help="Optional JSON path for parsed natural-language query.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-missing-images", action="store_true", help="Return cards without art_image paths.")
    parser.add_argument("--parse-with-llm", action="store_true")
    parser.add_argument("--provider", default="minimax", choices=["google", "minimax"])
    parser.add_argument("--model", help="LLM model name. Defaults by provider.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = _queries_from_args(args)
    results = retrieve_many(
        card_index_path=args.card_index,
        queries=queries,
        top_k=args.top_k,
        out_path=args.out,
        require_image=not args.allow_missing_images,
    )
    print("Done.")
    print(f"queries={len(queries)}")
    print(f"results={len(results)}")
    print(f"out={args.out}")
    for row in results[: min(10, len(results))]:
        print(f"{row['query_id']} rank={row['rank']} score={row['score']} card={row['card_name']} reasons={'; '.join(row['reasons'])}")


def _queries_from_args(args: argparse.Namespace) -> list[dict]:
    if args.queries:
        return load_structured_queries(args.queries)
    if not args.query_text:
        raise SystemExit("Provide --queries or --query-text.")
    if args.parse_with_llm:
        model = args.model or DEFAULT_MODELS[args.provider]
        query = parse_query_llm(
            args.query_text,
            provider=args.provider,
            model=model,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            query_id=args.query_id,
        )
    else:
        query = parse_query_rule(args.query_text, query_id=args.query_id)
    if args.parsed_query_out:
        args.parsed_query_out.parent.mkdir(parents=True, exist_ok=True)
        args.parsed_query_out.write_text(json.dumps(query, ensure_ascii=False, indent=2), encoding="utf-8")
    return [query]


if __name__ == "__main__":
    main()

