#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.semantics.enrichment import DEFAULT_PROMPT_TEMPLATE, run_enrichment


DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-enrich structured Hearthstone semantics.")
    parser.add_argument("--semantics", type=Path, default=Path("data/semantics/cards_semantics_base.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/semantics_enriched"))
    parser.add_argument("--prompt-template", type=Path, default=Path(DEFAULT_PROMPT_TEMPLATE))
    parser.add_argument("--limit", type=int, help="Enrich only first N semantic records.")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--chunk-strategy", default="set_class", choices=["set_class", "sequential"])
    parser.add_argument("--set-name", help="Only enrich cards from this expansion/set name.")
    parser.add_argument("--class-name", help="Only enrich cards containing this Hearthstone class name.")
    parser.add_argument("--card-id", type=int, action="append", help="Only enrich these card IDs. Can be repeated.")
    parser.add_argument("--collectible-only", action="store_true", help="Only enrich collectible cards.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and merged base outputs without API calls.")
    parser.add_argument("--provider", default="minimax", choices=["google", "minimax"])
    parser.add_argument("--model", help="LLM model name. Defaults by provider.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    stats = run_enrichment(
        semantics_path=args.semantics,
        out_dir=args.out_dir,
        prompt_template=args.prompt_template,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_strategy=args.chunk_strategy,
        set_name=args.set_name,
        class_name=args.class_name,
        card_ids=set(args.card_id or []) or None,
        collectible_only=args.collectible_only,
        dry_run=args.dry_run,
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
        force_llm=args.force_llm,
    )
    print("Done.")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
