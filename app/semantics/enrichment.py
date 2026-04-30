from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kg.graph import parse_json_response
from app.kg.io import read_jsonl, write_jsonl
from app.kg.llm import run_llm_batches
from app.semantics.caption import build_lora_caption


DEFAULT_PROMPT_TEMPLATE = "app/semantics/prompts/enrich_semantics_prompt.md"


def build_enrichment_prompts(
    records: list[dict[str, Any]],
    *,
    template: str,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(0, len(records), chunk_size):
        chunk = records[idx : idx + chunk_size]
        payload = [_prompt_card(record) for record in chunk]
        rows.append(
            {
                "batch_id": idx // chunk_size + 1,
                "card_count": len(chunk),
                "card_ids": [record["card_id"] for record in chunk],
                "prompt": template.replace("{{CARDS_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2)),
            }
        )
    return rows


def run_enrichment(
    *,
    semantics_path: Path,
    out_dir: Path,
    prompt_template: Path = Path(DEFAULT_PROMPT_TEMPLATE),
    limit: int | None = None,
    chunk_size: int = 20,
    dry_run: bool = False,
    provider: str = "minimax",
    model: str = "MiniMax-M2.7",
    temperature: float = 0.1,
    timeout_seconds: int = 90,
    resume: bool = True,
    force_llm: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "enrichment_prompts.jsonl"
    outputs_path = out_dir / "enrichment_llm_outputs.jsonl"
    enriched_path = out_dir / "cards_semantics_enriched.jsonl"
    captions_path = out_dir / "lora_captions_enriched.jsonl"

    records = read_jsonl(semantics_path)
    if limit is not None:
        records = records[:limit]

    template = prompt_template.read_text(encoding="utf-8")
    prompt_rows = build_enrichment_prompts(records, template=template, chunk_size=chunk_size)
    write_jsonl(prompts_path, prompt_rows)

    existing_outputs = read_jsonl(outputs_path) if resume and not force_llm else []
    if not resume or force_llm:
        outputs_path.write_text("", encoding="utf-8")

    run_llm_batches(
        prompt_rows,
        output_path=outputs_path,
        existing_outputs=existing_outputs,
        dry_run=dry_run,
        provider=provider,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        resume=resume,
        force=force_llm,
    )

    outputs = read_jsonl(outputs_path)
    enriched_by_id = _parse_enrichment_outputs(outputs)
    merged = [_merge_record(record, enriched_by_id.get(record["card_id"])) for record in records]
    write_jsonl(enriched_path, merged)
    write_jsonl(captions_path, _caption_rows(merged))
    (out_dir / "enrichment_summary.json").write_text(
        json.dumps(
            {
                "cards": len(records),
                "batches": len(prompt_rows),
                "enriched_cards": len(enriched_by_id),
                "dry_run": dry_run,
                "provider": provider,
                "model": model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "cards": len(records),
        "batches": len(prompt_rows),
        "enriched_cards": len(enriched_by_id),
        "out_dir": str(out_dir),
    }


def _prompt_card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": record["card_id"],
        "name": record.get("name"),
        "collectible": record.get("collectible"),
        "identity": record.get("identity"),
        "stats": record.get("stats"),
        "text": record.get("text"),
        "keywords": record.get("keywords"),
        "actions": record.get("actions"),
        "mechanic_tags": record.get("mechanic_tags"),
        "visual_tags": record.get("visual_tags"),
        "child_card_ids": record.get("child_card_ids"),
        "derived_cards": record.get("derived_cards"),
    }


def _parse_enrichment_outputs(outputs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for output in outputs:
        if output.get("status") not in {"ok", "dry_run"}:
            continue
        raw_response = output.get("raw_response") or ""
        if output.get("status") == "dry_run":
            raw_response = json.dumps({"cards": []})
        try:
            parsed = parse_json_response(raw_response)
        except (json.JSONDecodeError, ValueError):
            continue
        for card in parsed.get("cards", []):
            card_id = card.get("card_id")
            if isinstance(card_id, int):
                rows[card_id] = card
    return rows


def _merge_record(record: dict[str, Any], enriched: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(record)
    if enriched:
        for key in ["actions", "mechanic_tags", "visual_tags", "derived_cards", "semantic_summary"]:
            if key in enriched and enriched[key] not in (None, "", []):
                merged[key] = enriched[key]
    merged["enrichment"] = {
        "status": "enriched" if enriched else "base_only",
        "source": "llm" if enriched else "rules",
    }
    merged["lora_caption"] = build_lora_caption(merged, enriched=bool(enriched))
    return merged


def _caption_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        image = record.get("source", {}).get("art_image", "")
        if not image:
            continue
        rows.append(
            {
                "card_id": record["card_id"],
                "slug": record.get("slug", ""),
                "name": record.get("name", ""),
                "collectible": record.get("collectible", False),
                "root_collectible_ids": record.get("root_collectible_ids", []),
                "image": image,
                "caption": record.get("lora_caption", ""),
                "enrichment_status": record.get("enrichment", {}).get("status", "base_only"),
            }
        )
    return rows
