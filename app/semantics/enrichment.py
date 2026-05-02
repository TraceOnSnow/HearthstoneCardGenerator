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
    chunk_strategy: str = "set_class",
    context_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup_records = context_records or records
    record_by_id = {record.get("card_id"): record for record in lookup_records if isinstance(record.get("card_id"), int)}
    batch_id = 1
    for chunk, chunk_key in _iter_chunks(records, chunk_size=chunk_size, chunk_strategy=chunk_strategy):
        payload = [_prompt_card(record, record_by_id) for record in chunk]
        rows.append(
            {
                "batch_id": batch_id,
                "chunk_key": chunk_key,
                "chunk_strategy": chunk_strategy,
                "card_count": len(chunk),
                "card_ids": [record["card_id"] for record in chunk],
                "prompt": template.replace("{{CARDS_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2)),
            }
        )
        batch_id += 1
    return rows


def run_enrichment(
    *,
    semantics_path: Path,
    out_dir: Path,
    prompt_template: Path = Path(DEFAULT_PROMPT_TEMPLATE),
    limit: int | None = None,
    chunk_size: int = 20,
    chunk_strategy: str = "set_class",
    set_name: str | None = None,
    class_name: str | None = None,
    card_ids: set[int] | None = None,
    collectible_only: bool = False,
    dry_run: bool = False,
    provider: str = "minimax",
    model: str = "MiniMax-M2.7",
    temperature: float = 0.1,
    timeout_seconds: int = 90,
    resume: bool = True,
    force_llm: bool = False,
    concurrency: int = 1,
    max_retries: int = 0,
    retry_backoff_seconds: float = 10.0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "enrichment_prompts.jsonl"
    outputs_path = out_dir / "enrichment_llm_outputs.jsonl"
    enriched_path = out_dir / "cards_semantics_enriched.jsonl"
    captions_path = out_dir / "lora_captions_enriched.jsonl"

    all_records = read_jsonl(semantics_path)
    records = _filter_records(all_records, set_name=set_name, class_name=class_name, card_ids=card_ids, collectible_only=collectible_only)
    if limit is not None:
        records = records[:limit]

    template = prompt_template.read_text(encoding="utf-8")
    prompt_rows = build_enrichment_prompts(
        records,
        template=template,
        chunk_size=chunk_size,
        chunk_strategy=chunk_strategy,
        context_records=all_records,
    )
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
        concurrency=concurrency,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
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
                "chunk_strategy": chunk_strategy,
                "set_name": set_name,
                "class_name": class_name,
                "card_ids": sorted(card_ids) if card_ids else None,
                "collectible_only": collectible_only,
                "concurrency": concurrency,
                "max_retries": max_retries,
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


def _filter_records(
    records: list[dict[str, Any]],
    *,
    set_name: str | None,
    class_name: str | None,
    card_ids: set[int] | None,
    collectible_only: bool,
) -> list[dict[str, Any]]:
    filtered = records
    if card_ids:
        filtered = [record for record in filtered if record.get("card_id") in card_ids]
    if set_name:
        filtered = [record for record in filtered if record.get("identity", {}).get("set") == set_name]
    if class_name:
        filtered = [
            record
            for record in filtered
            if class_name in (record.get("identity", {}).get("card_class") or [])
        ]
    if collectible_only:
        filtered = [record for record in filtered if record.get("collectible")]
    return filtered


def _iter_chunks(
    records: list[dict[str, Any]],
    *,
    chunk_size: int,
    chunk_strategy: str,
) -> list[tuple[list[dict[str, Any]], str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_strategy == "sequential":
        return [
            (records[idx : idx + chunk_size], f"sequential:{idx // chunk_size + 1}")
            for idx in range(0, len(records), chunk_size)
        ]
    if chunk_strategy != "set_class":
        raise ValueError(f"Unsupported chunk_strategy: {chunk_strategy}")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    bucket_order: list[tuple[str, str]] = []
    for record in records:
        key = _set_class_key(record)
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(record)

    chunks: list[tuple[list[dict[str, Any]], str]] = []
    for set_value, class_value in bucket_order:
        bucket = buckets[(set_value, class_value)]
        for idx in range(0, len(bucket), chunk_size):
            chunk_key = f"set={set_value}|class={class_value}|part={idx // chunk_size + 1}"
            chunks.append((bucket[idx : idx + chunk_size], chunk_key))
    return chunks


def _set_class_key(record: dict[str, Any]) -> tuple[str, str]:
    identity = record.get("identity", {})
    set_value = str(identity.get("set") or "unknown_set")
    classes = identity.get("card_class") or ["unknown_class"]
    class_value = "+".join(str(value) for value in classes) or "unknown_class"
    return set_value, class_value


def _prompt_card(record: dict[str, Any], record_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
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
        "child_cards": [_child_prompt_card(record_by_id[child_id]) for child_id in record.get("child_card_ids") or [] if child_id in record_by_id],
        "derived_cards": record.get("derived_cards"),
    }


def _child_prompt_card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": record.get("card_id"),
        "name": record.get("name"),
        "identity": record.get("identity"),
        "stats": record.get("stats"),
        "text": record.get("text"),
        "keywords": record.get("keywords"),
        "actions": record.get("actions"),
        "mechanic_tags": record.get("mechanic_tags"),
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
        if not isinstance(parsed, dict):
            continue
        for card in parsed.get("cards", []):
            card_id = card.get("card_id")
            if isinstance(card_id, int):
                rows[card_id] = card
    return rows


def _merge_record(record: dict[str, Any], enriched: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(record)
    if enriched:
        for key in [
            "actions",
            "action_groups",
            "mechanic_tags",
            "constraints",
            "generated_card_refs",
            "related_card_refs",
            "semantic_summary",
            "generation_hints",
        ]:
            if key in enriched and enriched[key] not in (None, "", []):
                merged[key] = enriched[key]
        if enriched.get("visual_tags") not in (None, "", []):
            merged["visual_tags"] = enriched["visual_tags"]
        if enriched.get("derived_cards") not in (None, "", []):
            merged["derived_cards"] = _merge_derived_cards(record.get("derived_cards") or [], enriched["derived_cards"])
    merged["enrichment"] = {
        "status": "enriched" if enriched else "base_only",
        "source": "llm" if enriched else "rules",
    }
    merged["lora_caption"] = build_lora_caption(merged, enriched=bool(enriched))
    return merged


def _merge_derived_cards(base: list[dict[str, Any]], enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []
    for row in base + enriched:
        if not isinstance(row, dict) or not isinstance(row.get("card_id"), int):
            continue
        card_id = row["card_id"]
        if card_id not in by_id:
            by_id[card_id] = {}
            ordered_ids.append(card_id)
        by_id[card_id].update({key: value for key, value in row.items() if value not in (None, "", [])})
        by_id[card_id].setdefault("relation", "HAS_CHILD_CARD")
    return [by_id[card_id] for card_id in ordered_ids]


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
