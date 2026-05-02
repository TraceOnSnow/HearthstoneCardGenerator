#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.card_design.kg_designer import design_card_from_kg, write_design_outputs  # noqa: E402


DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design a custom Hearthstone card with KG retrieval evidence.")
    parser.add_argument("--request", required=True, help="Natural-language DIY card request.")
    parser.add_argument("--query-id", help="Stable id for outputs.")
    parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    parser.add_argument("--semantics", type=Path, default=Path("data/semantics/cards_semantics_base.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/kg_card_design"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--parse-with-llm", action="store_true")
    parser.add_argument("--provider", default="minimax", choices=["google", "minimax"])
    parser.add_argument("--model", help="LLM model name. Defaults by provider.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    result = design_card_from_kg(
        request_text=args.request,
        card_index_path=args.card_index,
        semantics_path=args.semantics,
        top_k=args.top_k,
        parse_with_llm=args.parse_with_llm,
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        query_id=args.query_id,
    )
    write_design_outputs(result, out_dir=args.out_dir)
    card = result["design"].get("card", {})
    print(f"name={card.get('name')}")
    print(f"type={card.get('card_type')} cost={card.get('mana_cost')} stats={card.get('attack')}/{card.get('health')}")
    print(f"text={card.get('rules_text')}")
    print(f"out={args.out_dir}")


if __name__ == "__main__":
    main()
