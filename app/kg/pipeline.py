from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.kg.graph import build_graph
from app.kg.io import load_cards, read_jsonl, select_cards, write_jsonl
from app.kg.llm import run_llm_batches
from app.kg.metadata import load_metadata
from app.kg.models import KgRunConfig
from app.kg.prompting import build_prompt_rows, load_prompt_template


def run_pipeline(config: KgRunConfig) -> dict[str, int | str]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_cards_path = out_dir / "cards_selected.jsonl"
    prompts_path = out_dir / "prompts.jsonl"
    outputs_path = out_dir / "llm_outputs.jsonl"
    graph_path = out_dir / "graph.json"
    html_path = out_dir / "graph_vis.html"
    run_config_path = out_dir / "run_config.json"

    cards = load_cards(Path(config.source_jsonl), collectible_only=True)
    selected_cards = select_cards(
        cards,
        limit=config.limit,
        sample_size=config.sample_size,
        seed=config.random_seed,
    )

    template = load_prompt_template(Path(config.prompt_template))
    prompt_rows = build_prompt_rows(
        selected_cards,
        template=template,
        chunk_size=config.chunk_size,
    )

    write_jsonl(selected_cards_path, (card.to_json_dict() for card in selected_cards))
    write_jsonl(prompts_path, prompt_rows)
    run_config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    existing_outputs = read_jsonl(outputs_path) if config.resume and not config.force_llm else []
    if not config.resume or config.force_llm:
        outputs_path.write_text("", encoding="utf-8")

    llm_outputs = run_llm_batches(
        prompt_rows,
        output_path=outputs_path,
        existing_outputs=existing_outputs,
        dry_run=config.dry_run,
        provider=config.provider,
        model=config.model,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
        resume=config.resume,
        force=config.force_llm,
    )

    # Re-read from disk so graph generation reflects exactly what was persisted.
    llm_outputs = read_jsonl(outputs_path)
    metadata = load_metadata(Path(config.metadata_json) if config.metadata_json else None)
    graph = build_graph(selected_cards, llm_outputs, metadata=metadata)
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    if config.visualize:
        from app.kg.visualize import save_graph_html

        save_graph_html(graph, html_path)

    stats = {
        "cards": len(selected_cards),
        "batches": len(prompt_rows),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "out_dir": str(out_dir),
        "graph": str(graph_path),
    }
    if config.visualize:
        stats["html"] = str(html_path)
    return stats
