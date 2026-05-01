#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.semantic_kg.query_parser import parse_query_llm, parse_query_rule


DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse natural language into a structured KG retrieval query.")
    parser.add_argument("text")
    parser.add_argument("--query-id")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parse-with-llm", action="store_true")
    parser.add_argument("--provider", default="minimax", choices=["google", "minimax"])
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parse_with_llm:
        model = args.model or DEFAULT_MODELS[args.provider]
        query = parse_query_llm(
            args.text,
            provider=args.provider,
            model=model,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            query_id=args.query_id,
        )
    else:
        query = parse_query_rule(args.text, query_id=args.query_id)

    text = json.dumps(query, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

