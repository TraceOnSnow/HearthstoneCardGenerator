from __future__ import annotations

import argparse
from pathlib import Path

from app.workflow import DEFAULT_MODELS, GenerateOptions, run_generate


def main() -> None:
    parser = argparse.ArgumentParser(prog="hs-cardgen", description="Local-first Hearthstone custom card generation workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Check local data and model configuration.")
    init_parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    init_parser.add_argument("--semantics", type=Path, default=Path("data/semantics_enriched_current/cards_semantics_enriched.jsonl"))
    init_parser.add_argument("--lora-dir", type=Path, default=Path("models/sd15-hearthstone-lora"))

    generate_parser = subparsers.add_parser("generate", help="Generate a custom card, art prompt, art image, and final card PNG.")
    generate_parser.add_argument("request", nargs="?", help="Natural-language custom card request.")
    generate_parser.add_argument("--request", dest="request_flag", help="Natural-language custom card request.")
    generate_parser.add_argument("--out-dir", type=Path, default=Path(""), help="Output directory. Defaults to runs/<slug>_<timestamp>.")
    generate_parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    generate_parser.add_argument("--semantics", type=Path, default=Path("data/semantics_enriched_current/cards_semantics_enriched.jsonl"))
    generate_parser.add_argument("--top-k", type=int, default=8)
    generate_parser.add_argument("--parse-with-llm", action="store_true", help="Use the configured LLM to parse retrieval fields.")
    generate_parser.add_argument("--provider", choices=["minimax", "google"], default="minimax")
    generate_parser.add_argument("--model", help="LLM model name. Defaults by provider.")
    generate_parser.add_argument("--temperature", type=float, default=0.3)
    generate_parser.add_argument("--timeout-seconds", type=int, default=180)
    generate_parser.add_argument("--query-id", help="Stable id for this run.")
    generate_parser.add_argument("--mock-design", action="store_true", help="Use a deterministic local mock card design instead of an external LLM.")
    generate_parser.add_argument("--image-provider", choices=["mock", "lora", "none"], default="mock")
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--pretrained-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    generate_parser.add_argument("--lora-dir", type=Path, default=Path("models/sd15-hearthstone-lora"))
    generate_parser.add_argument("--steps", type=int, default=30)
    generate_parser.add_argument("--guidance-scale", type=float, default=7.5)
    generate_parser.add_argument("--lora-scale", type=float, default=1.0)
    generate_parser.add_argument("--width", type=int, default=512)
    generate_parser.add_argument("--height", type=int, default=512)

    args = parser.parse_args()
    if args.command == "init":
        _run_init(args)
        return
    if args.command == "generate":
        _run_generate(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


def _run_init(args: argparse.Namespace) -> None:
    checks = [
        ("card_index", args.card_index),
        ("semantics", args.semantics),
        ("lora_dir", args.lora_dir),
    ]
    ok = True
    for label, path in checks:
        exists = path.exists()
        ok = ok and (exists or label == "lora_dir")
        status = "ok" if exists else "missing"
        optional = " (optional unless --image-provider lora)" if label == "lora_dir" else ""
        print(f"{label}: {status} {path}{optional}")
    print("llm_models:")
    for provider, model in DEFAULT_MODELS.items():
        print(f"  {provider}: {model}")
    if not ok:
        raise SystemExit(1)


def _run_generate(args: argparse.Namespace) -> None:
    request_text = args.request_flag or args.request
    if not request_text:
        raise SystemExit("Pass a request as a positional argument or with --request.")
    result = run_generate(
        GenerateOptions(
            request_text=request_text,
            out_dir=args.out_dir,
            card_index_path=args.card_index,
            semantics_path=args.semantics,
            top_k=args.top_k,
            parse_with_llm=args.parse_with_llm,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            query_id=args.query_id,
            mock_design=args.mock_design,
            image_provider=args.image_provider,
            seed=args.seed,
            pretrained_model=args.pretrained_model,
            lora_dir=args.lora_dir,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            lora_scale=args.lora_scale,
            width=args.width,
            height=args.height,
        )
    )
    card = result["card"]
    print(f"name={card.get('name')}")
    print(f"type={card.get('card_type')} cost={card.get('mana_cost')} stats={card.get('attack')}/{card.get('health')}")
    print(f"text={card.get('rules_text')}")
    print(f"out={result['out_dir']}")
    final_card = result["artifacts"].get("final_card")
    if final_card:
        print(f"final_card={final_card}")


if __name__ == "__main__":
    main()
