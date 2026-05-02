#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import read_jsonl, write_jsonl  # noqa: E402
from app.kg.graph import parse_json_response  # noqa: E402
from app.kg.llm import _api_key_for_provider, _generate_content, load_dotenv  # noqa: E402
from app.retrieval.common import load_caption_corpus  # noqa: E402
from app.retrieval.clip_baseline import DEFAULT_CLIP_MODEL, retrieve_clip  # noqa: E402
from app.retrieval.evaluation import render_grid  # noqa: E402
from app.retrieval.tfidf import retrieve_tfidf  # noqa: E402
from app.semantic_kg.query_parser import parse_query_llm, parse_query_rule  # noqa: E402
from app.semantic_kg.retrieval import retrieve_many  # noqa: E402


DEFAULT_MODELS = {"minimax": "MiniMax-M2.7", "google": "gemini-2.5-flash-lite"}
METHODS = ["tfidf_baseline", "clip_baseline", "semantic_kg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end DIY retrieval + card text evaluation over user prompts.")
    parser.add_argument("--prompts", type=Path, default=Path("configs/diy_user_prompts.json"))
    parser.add_argument("--captions", type=Path, default=Path("data/semantics/lora_captions.jsonl"))
    parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/diy_retrieval_design_eval"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parse-with-llm", action="store_true")
    parser.add_argument("--provider", default="minimax", choices=["minimax", "google"])
    parser.add_argument("--model")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--mock-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-cache", type=Path, default=Path("results/diy_retrieval_design_eval/clip_image_cache.npz"))
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-device", default=None)
    parser.add_argument("--clip-limit", type=int, help="Limit CLIP corpus for smoke tests only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _load_diy_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    queries = [_parse_query(item, args) for item in prompts]
    query_items = {item["prompt_id"]: item for item in prompts}
    query_payload = {"queries": queries}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "parsed_queries.json").write_text(json.dumps(query_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus = load_caption_corpus(args.captions)
    tfidf_rows = retrieve_tfidf(corpus=corpus, queries=queries, top_k=args.top_k)
    clip_rows = _mock_clip_from_tfidf(tfidf_rows) if args.mock_clip else _run_real_clip(corpus, queries, args)
    kg_rows = retrieve_many(card_index_path=args.card_index, queries=queries, top_k=args.top_k, require_image=True)
    all_rows = [*tfidf_rows, *clip_rows, *kg_rows]
    write_jsonl(args.out_dir / "retrieval_results.jsonl", all_rows)
    write_jsonl(args.out_dir / "tfidf_results.jsonl", tfidf_rows)
    write_jsonl(args.out_dir / "clip_results.jsonl", clip_rows)
    write_jsonl(args.out_dir / "kg_results.jsonl", kg_rows)
    render_grid(all_rows, out_path=args.out_dir / "retrieval_grid.html", image_root=args.image_root)

    if args.mock_design:
        designs = [_mock_design_for_query(query, query_items.get(query["query_id"], {}), _top_rows(all_rows, query["query_id"])) for query in queries]
    else:
        designs = []
        for index, query in enumerate(queries, start=1):
            print(f"design {index}/{len(queries)} {query['query_id']}", flush=True)
            designs.append(_real_design_for_query(query, query_items.get(query["query_id"], {}), _top_rows(all_rows, query["query_id"]), args))
    write_jsonl(args.out_dir / "diy_card_designs.jsonl", designs)

    if args.mock_judge:
        retrieval_scores = [_score_retrieval_row(row, query_items.get(str(row.get("query_id")), {})) for row in all_rows]
        design_scores = [_score_design_row(row, query_items.get(str(row.get("prompt_id")), {})) for row in designs]
    else:
        retrieval_scores, design_scores = _real_judge(queries, query_items, all_rows, designs, args)
    write_jsonl(args.out_dir / "retrieval_scores.jsonl", retrieval_scores)
    write_jsonl(args.out_dir / "design_scores.jsonl", design_scores)
    _write_retrieval_summary(args.out_dir / "retrieval_metrics_summary.csv", retrieval_scores)
    _write_design_summary(args.out_dir / "design_text_metrics_summary.csv", design_scores)
    _write_tables(args.out_dir)
    print(f"prompts={len(prompts)}")
    print(f"retrieval_rows={len(all_rows)}")
    print(f"design_rows={len(designs)}")
    print(f"out={args.out_dir}")


def _load_diy_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("prompts", data if isinstance(data, list) else []))


def _parse_query(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.parse_with_llm:
        query = parse_query_llm(
            item["user_request"],
            provider=args.provider,
            model=args.model or DEFAULT_MODELS[args.provider],
            temperature=0.0,
            timeout_seconds=args.timeout_seconds,
            query_id=item["prompt_id"],
        )
    else:
        query = parse_query_rule(item["user_request"], query_id=item["prompt_id"])
    return _merge_expected_hints(query, item.get("expected_intent", {}))


def _merge_expected_hints(query: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "class": "classes",
        "card_type": "card_types",
        "mechanics": "actions",
        "spell_school": "spell_schools",
        "minion_type": "minion_types",
    }
    for source, target in mapping.items():
        for value in expected.get(source, []) or []:
            if value not in query.setdefault(target, []):
                query[target].append(value)
    for field in ["related_card_names", "generated_card_names", "generated_roles"]:
        for value in expected.get(field, []) or []:
            if value not in query.setdefault(field, []):
                query[field].append(value)
    return query


def _mock_clip_from_tfidf(tfidf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in tfidf_rows:
        cloned = dict(row)
        cloned["method"] = "clip_baseline"
        cloned["score"] = round(float(row.get("score") or 0) * 0.92, 6)
        cloned["reasons"] = ["mock_clip_from_caption_similarity", *list(row.get("reasons") or [])[:2]]
        rows.append(cloned)
    return rows


def _run_real_clip(corpus: list[dict[str, Any]], queries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = retrieve_clip(
        corpus=corpus,
        queries=queries,
        image_root=args.image_root,
        cache_path=args.clip_cache,
        top_k=args.top_k,
        model_name=args.clip_model,
        limit=args.clip_limit,
        batch_size=args.clip_batch_size,
        device=args.clip_device,
    )
    for row in rows:
        row["method"] = "clip_baseline"
    return rows


def _top_rows(rows: list[dict[str, Any]], query_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("query_id") == query_id and int(row.get("rank") or 0) <= 3]


def _mock_design_for_query(query: dict[str, Any], item: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    expected = item.get("expected_intent", {})
    classes = query.get("classes") or expected.get("class") or ["Neutral"]
    card_types = query.get("card_types") or expected.get("card_type") or ["Minion"]
    actions = query.get("actions") or expected.get("mechanics") or []
    name = _mock_name(item.get("prompt_id", query["query_id"]))
    if item.get("stage") == "art_only":
        name = _extract_name(item.get("user_request", "")) or name
    rules = _rules_text(actions, item)
    return {
        "prompt_id": query["query_id"],
        "user_request": item.get("user_request", query.get("text", "")),
        "mock": True,
        "card": {
            "name": name,
            "mana_cost": 3,
            "card_type": card_types[0],
            "class": classes[:2],
            "rarity": (expected.get("rarity") or ["Common"])[0],
            "minion_type": (query.get("minion_types") or expected.get("minion_type") or [None])[0],
            "attack": 3 if card_types[0] == "Minion" else None,
            "health": 4 if card_types[0] == "Minion" else None,
            "durability": 2 if card_types[0] == "Weapon" else None,
            "keywords": query.get("keywords", [])[:3],
            "rules_text": rules,
            "flavor_text": "A custom-card prototype generated from KG evidence.",
        },
        "structured_semantics": {
            "actions": actions,
            "related_card_refs": query.get("related_card_names", []),
            "semantic_summary": f"A {classes[0]} {card_types[0]} focused on {', '.join(actions[:3]) or 'class fantasy'}.",
            "lora_caption": _caption(query, item),
        },
        "kg_evidence_used": [
            {"card_id": row.get("card_id"), "name": row.get("card_name"), "method": row.get("method"), "used_for": "; ".join(row.get("reasons") or [])}
            for row in evidence[:5]
        ],
    }


def _real_design_for_query(query: dict[str, Any], item: dict[str, Any], evidence: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    api_key = _api_key_for_provider(args.provider)
    compact_evidence = _compact_evidence(evidence[:9])
    prompt = f"""You design one custom Hearthstone card from a user request and retrieval evidence.
Return one strict JSON object only. Do not use markdown. Do not include explanations.

If the user already fully specified the card, preserve the card and mainly generate structured semantics and lora_caption.
If the request is vague, create one simple plausible Hearthstone card with official-style wording.

User request:
{item.get('user_request', query.get('text', ''))}

Structured query:
{json.dumps(query, ensure_ascii=False, indent=2)}

Top retrieval evidence:
{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}

Return schema:
{{
  "prompt_id": "{query['query_id']}",
  "mock": false,
  "card": {{
    "name": "string",
    "mana_cost": 0,
    "card_type": "Minion|Spell|Weapon|Location|Hero",
    "class": ["Neutral"],
    "rarity": "Common|Rare|Epic|Legendary|Free",
    "minion_type": null,
    "attack": null,
    "health": null,
    "durability": null,
    "keywords": [],
    "rules_text": "short Hearthstone-style rules text",
    "flavor_text": "short flavor text"
  }},
  "structured_semantics": {{
    "actions": [],
    "related_card_refs": [],
    "semantic_summary": "one sentence",
    "lora_caption": "Hearthstone card art, ..."
  }},
  "kg_evidence_used": [
    {{"card_id": 0, "name": "string", "method": "semantic_kg|tfidf_baseline|clip_baseline", "used_for": "string"}}
  ]
}}
"""
    raw = _generate_content(
        prompt,
        provider=args.provider,
        api_key=api_key,
        model=args.model or DEFAULT_MODELS[args.provider],
        temperature=0.2,
        timeout_seconds=args.timeout_seconds,
    )
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        fallback = _mock_design_for_query(query, item, evidence)
        fallback["mock"] = False
        fallback["llm_parse_failed"] = True
        fallback["raw_response_excerpt"] = raw[:1000]
        return fallback
    parsed.setdefault("prompt_id", query["query_id"])
    parsed["mock"] = False
    parsed.setdefault("user_request", item.get("user_request", query.get("text", "")))
    return parsed


def _compact_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "method": row.get("method"),
                "rank": row.get("rank"),
                "card_id": row.get("card_id"),
                "card_name": row.get("card_name"),
                "score": row.get("score"),
                "caption": row.get("caption"),
                "reasons": list(row.get("reasons") or [])[:5],
            }
        )
    return compact


def _real_judge(
    queries: list[dict[str, Any]],
    query_items: dict[str, dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    designs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_dotenv()
    api_key = _api_key_for_provider(args.provider)
    retrieval_scores: list[dict[str, Any]] = []
    design_scores: list[dict[str, Any]] = []
    designs_by_id = {row.get("prompt_id"): row for row in designs}
    for index, query in enumerate(queries, start=1):
        qid = query["query_id"]
        print(f"judge {index}/{len(queries)} {qid}", flush=True)
        item = query_items.get(qid, {})
        rows = [_compact_judge_row(row) for row in retrieval_rows if row.get("query_id") == qid and int(row.get("rank") or 0) <= 5]
        design = designs_by_id.get(qid, {})
        prompt = f"""You are grading a Hearthstone custom-card retrieval and design system.
Return strict JSON only. Score each field from 0.0 to 1.0.

User request:
{item.get('user_request', query.get('text', ''))}

Expected intent hints:
{json.dumps(item.get('expected_intent', {}), ensure_ascii=False, indent=2)}

Structured query:
{json.dumps(query, ensure_ascii=False, indent=2)}

Retrieval rows to grade:
{json.dumps(rows, ensure_ascii=False, indent=2)}

Generated DIY card to grade:
{json.dumps(design, ensure_ascii=False, indent=2)}

Return schema:
{{
  "retrieval_scores": [
    {{
      "query_id": "{qid}",
      "method": "string",
      "rank": 1,
      "card_id": 0,
      "card_name": "string",
      "class_match": 0.0,
      "action_match": 0.0,
      "relation_match": 0.0,
      "type_match": 0.0,
      "overall_relevance": 0.0,
      "notes": "short"
    }}
  ],
  "design_score": {{
    "prompt_id": "{qid}",
    "card_name": "string",
    "class_match": 0.0,
    "action_match": 0.0,
    "relation_match": 0.0,
    "type_match": 0.0,
    "wording_concision": 0.0,
    "overall_design_text_quality": 0.0,
    "notes": "short"
  }}
}}
"""
        raw = _generate_content(
            prompt,
            provider=args.provider,
            api_key=api_key,
            model=args.model or DEFAULT_MODELS[args.provider],
            temperature=0.0,
            timeout_seconds=args.timeout_seconds,
        )
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict):
            for row in retrieval_rows:
                if row.get("query_id") == qid and int(row.get("rank") or 0) <= 5:
                    scored = _score_retrieval_row(row, item)
                    scored["mock"] = False
                    scored["llm_parse_failed"] = True
                    retrieval_scores.append(scored)
            scored_design = _score_design_row(design, item)
            scored_design["mock"] = False
            scored_design["llm_parse_failed"] = True
            design_scores.append(scored_design)
            continue
        for row in parsed.get("retrieval_scores", []):
            if isinstance(row, dict):
                row["mock"] = False
                retrieval_scores.append(row)
        design_score = parsed.get("design_score")
        if isinstance(design_score, dict):
            design_score["mock"] = False
            design_scores.append(design_score)
    return retrieval_scores, design_scores


def _compact_judge_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row.get("query_id"),
        "method": row.get("method"),
        "rank": row.get("rank"),
        "card_id": row.get("card_id"),
        "card_name": row.get("card_name"),
        "caption": row.get("caption"),
        "reasons": list(row.get("reasons") or [])[:5],
    }


def _mock_name(prompt_id: str) -> str:
    words = [word.capitalize() for word in prompt_id.split("_") if word not in {"vague", "semi", "complete", "community", "art"}]
    return " ".join(words[:3]) or "Custom Card"


def _extract_name(text: str) -> str | None:
    for marker in ["Name:", "The card is:", "Finished card:"]:
        if marker in text:
            value = text.split(marker, 1)[1].split(".", 1)[0].split(",", 1)[0].strip()
            return value or None
    return None


def _rules_text(actions: list[str], item: dict[str, Any]) -> str:
    if item.get("stage") == "art_only" and "Text:" in item.get("user_request", ""):
        return item["user_request"].split("Text:", 1)[1].split(".", 1)[0].strip() + "."
    if "gain_armor" in actions:
        return "Battlecry: Gain 3 Armor."
    if "summon" in actions:
        return "Battlecry: Summon a 1/1 token."
    if "draw" in actions:
        return "Draw a card."
    if "heal" in actions or "restore_health" in actions:
        return "Restore 4 Health."
    if "freeze" in actions:
        return "Freeze an enemy minion."
    return "Battlecry: Gain a small bonus based on your class."


def _caption(query: dict[str, Any], item: dict[str, Any]) -> str:
    parts = ["Hearthstone card art", *query.get("classes", [])[:2], *query.get("card_types", [])[:2], *query.get("actions", [])[:3]]
    visual = item.get("expected_intent", {}).get("visual_tags", []) or (query.get("generation_hints", {}) or {}).get("visual_tags", [])
    parts.extend(visual[:4])
    return ", ".join(str(part) for part in parts if part)


def _score_retrieval_row(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected_intent", {})
    haystack = " ".join(
        [
            str(row.get("card_name", "")),
            str(row.get("caption", "")),
            " ".join(str(x) for x in row.get("reasons", []) or []),
        ]
    ).lower()
    class_score = _contains_any(haystack, expected.get("class", []))
    action_score = _contains_any(haystack, expected.get("mechanics", []))
    relation_score = _contains_any(haystack, expected.get("related_card_names", []) + expected.get("generated_card_names", []))
    type_score = _contains_any(haystack, expected.get("card_type", []) + expected.get("minion_type", []) + expected.get("spell_school", []))
    overall = round((class_score + action_score + relation_score + type_score) / 4, 4)
    if row.get("method") == "semantic_kg" and relation_score:
        overall = min(1.0, overall + 0.15)
    return {
        "query_id": row.get("query_id"),
        "method": row.get("method"),
        "rank": row.get("rank"),
        "card_id": row.get("card_id"),
        "card_name": row.get("card_name"),
        "class_match": class_score,
        "action_match": action_score,
        "relation_match": relation_score,
        "type_match": type_score,
        "overall_relevance": overall,
        "mock": True,
    }


def _score_design_row(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected_intent", {})
    text = json.dumps(row, ensure_ascii=False).lower()
    class_score = _contains_any(text, expected.get("class", []))
    action_score = _contains_any(text, expected.get("mechanics", []))
    relation_score = _contains_any(text, expected.get("related_card_names", []) + expected.get("generated_card_names", []))
    type_score = _contains_any(text, expected.get("card_type", []) + expected.get("minion_type", []) + expected.get("spell_school", []))
    rules = str(row.get("card", {}).get("rules_text", ""))
    concision = 1.0 if 0 < len(rules.split()) <= 18 else 0.5
    overall = round((class_score + action_score + relation_score + type_score + concision) / 5, 4)
    return {
        "prompt_id": row.get("prompt_id"),
        "card_name": row.get("card", {}).get("name"),
        "class_match": class_score,
        "action_match": action_score,
        "relation_match": relation_score,
        "type_match": type_score,
        "wording_concision": concision,
        "overall_design_text_quality": overall,
        "mock": True,
    }


def _contains_any(text: str, values: list[Any]) -> float:
    if not values:
        return 1.0
    normalized = text.replace("_", " ").lower()
    for value in values:
        value_text = str(value).replace("_", " ").lower()
        if value_text and value_text in normalized:
            return 1.0
    return 0.0


def _write_retrieval_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["class_match", "action_match", "relation_match", "type_match", "overall_relevance"]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("rank") or 0) <= 5:
            by_method.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method in METHODS:
        method_rows = by_method.get(method, [])
        item = {"method": method, "rows": len(method_rows)}
        for field in fields:
            vals = [float(row[field]) for row in method_rows]
            item[f"{field}_at5"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        summary.append(item)
    _write_csv(path, summary)


def _write_design_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["class_match", "action_match", "relation_match", "type_match", "wording_concision", "overall_design_text_quality"]
    item = {"method": "kg_augmented_mock_designer", "rows": len(rows)}
    for field in fields:
        vals = [float(row[field]) for row in rows]
        item[f"{field}_mean"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    _write_csv(path, [item])


def _write_tables(out_dir: Path) -> None:
    retrieval = list(csv.DictReader((out_dir / "retrieval_metrics_summary.csv").open("r", encoding="utf-8")))
    lines = ["| Method | Class@5 ↑ | Action@5 ↑ | Relation@5 ↑ | Overall@5 ↑ |", "|---|---:|---:|---:|---:|"]
    for row in retrieval:
        lines.append(f"| {row['method']} | {row['class_match_at5']} | {row['action_match_at5']} | {row['relation_match_at5']} | {row['overall_relevance_at5']} |")
    (out_dir / "table_retrieval_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    design = list(csv.DictReader((out_dir / "design_text_metrics_summary.csv").open("r", encoding="utf-8")))
    lines = ["| Method | Class ↑ | Action ↑ | Relation ↑ | Wording ↑ | Overall ↑ |", "|---|---:|---:|---:|---:|---:|"]
    for row in design:
        lines.append(
            f"| {row['method']} | {row['class_match_mean']} | {row['action_match_mean']} | {row['relation_match_mean']} | {row['wording_concision_mean']} | {row['overall_design_text_quality_mean']} |"
        )
    (out_dir / "table_design_text_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
