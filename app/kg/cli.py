from __future__ import annotations

import argparse
from pathlib import Path

from app.kg.models import KgRunConfig
from app.kg.pipeline import run_pipeline


DEFAULT_PROMPT = "app/kg/prompts/kg_entity_extraction_prompt.md"
DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "minimax": "MiniMax-M2.7",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Hearthstone collectible-card KG.")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="Build KG from collectible cards.")
    build.add_argument("--source", default="data/cards_collectible.jsonl")
    build.add_argument("--out-dir", default="data/kg_collectible")
    build.add_argument("--metadata", default="data/hearthstone_metadata.json")
    build.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    build.add_argument("--limit", type=int, help="Use first N collectible cards.")
    build.add_argument("--sample-size", type=int, help="Randomly sample N collectible cards.")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--chunk-size", type=int, default=50)
    build.add_argument("--dry-run", action="store_true", help="Skip LLM calls; build explicit metadata graph only.")
    build.add_argument("--provider", default="google", choices=["google", "minimax"])
    build.add_argument("--model", help="LLM model name. Defaults by provider.")
    build.add_argument("--temperature", type=float, default=0.1)
    build.add_argument("--timeout-seconds", type=int, default=60)
    build.add_argument("--no-resume", action="store_true", help="Ignore previous successful LLM batches.")
    build.add_argument("--force-llm", action="store_true", help="Rerun every LLM batch and overwrite outputs.")
    build.add_argument("--visualize", action="store_true", help="Write graph_vis.html after graph build.")

    smoke = subparsers.add_parser("smoke", help="Run a minimal local test without LLM calls.")
    smoke.add_argument("--source", default="data/cards_collectible.jsonl")
    smoke.add_argument("--out-dir", default="data/kg_smoke")
    smoke.add_argument("--metadata", default="data/hearthstone_metadata.json")
    smoke.add_argument("--limit", type=int, default=5)
    smoke.add_argument("--visualize", action="store_true")

    viz = subparsers.add_parser("visualize", help="Render graph JSON to HTML.")
    viz.add_argument("--input", default="data/kg_collectible/graph.json")
    viz.add_argument("--output", default="data/kg_collectible/graph_vis.html")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "build":
        model = args.model or DEFAULT_MODELS[args.provider]
        config = KgRunConfig(
            source_jsonl=args.source,
            out_dir=args.out_dir,
            metadata_json=args.metadata,
            prompt_template=args.prompt_template,
            limit=args.limit,
            sample_size=args.sample_size,
            random_seed=args.seed,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
            provider=args.provider,
            model=model,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            resume=not args.no_resume,
            force_llm=args.force_llm,
            visualize=args.visualize,
        )
        stats = run_pipeline(config)
    elif args.command == "smoke":
        config = KgRunConfig(
            source_jsonl=args.source,
            out_dir=args.out_dir,
            metadata_json=args.metadata,
            prompt_template=DEFAULT_PROMPT,
            limit=args.limit,
            sample_size=None,
            random_seed=42,
            chunk_size=max(1, args.limit),
            dry_run=True,
            provider="google",
            model="gemini-2.5-flash-lite",
            temperature=0.1,
            timeout_seconds=60,
            resume=False,
            force_llm=True,
            visualize=args.visualize,
        )
        stats = run_pipeline(config)
    elif args.command == "visualize":
        from app.kg.visualize import load_graph_json, save_graph_html

        graph = load_graph_json(Path(args.input))
        save_graph_html(graph, Path(args.output))
        stats = {"input": args.input, "html": args.output}
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print("Done.")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
