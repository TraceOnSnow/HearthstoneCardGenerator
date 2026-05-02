from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kg.graph import parse_json_response
from app.kg.io import read_jsonl, write_jsonl
from app.kg.llm import _api_key_for_provider, _generate_content, load_dotenv
from app.semantic_kg.query_parser import parse_query_llm, parse_query_rule
from app.semantic_kg.retrieval import retrieve_one


def design_card_from_kg(
    *,
    request_text: str,
    card_index_path: Path,
    semantics_path: Path,
    top_k: int,
    parse_with_llm: bool,
    provider: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
    query_id: str | None = None,
) -> dict[str, Any]:
    query = (
        parse_query_llm(
            request_text,
            provider=provider,
            model=model,
            temperature=0.0,
            timeout_seconds=timeout_seconds,
            query_id=query_id,
        )
        if parse_with_llm
        else parse_query_rule(request_text, query_id=query_id)
    )
    card_index = read_jsonl(card_index_path)
    retrieval_results = retrieve_one(card_index, query=query, top_k=top_k, require_image=False)
    retrieval_results = _add_related_name_matches(card_index, query=query, results=retrieval_results, top_k=top_k)
    semantics_by_id = {row.get("card_id"): row for row in read_jsonl(semantics_path)}
    evidence = [_evidence_row(row, semantics_by_id.get(row.get("card_id"))) for row in retrieval_results]
    evidence_package = build_evidence_package(query=query, evidence=evidence)

    load_dotenv()
    api_key = _api_key_for_provider(provider)
    raw_response = _generate_content(
        _build_design_prompt(request_text=request_text, query=query, evidence_package=evidence_package),
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
    parsed = parse_json_response(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Card designer did not return a JSON object.")
    return {
        "request": request_text,
        "query": query,
        "retrieval_results": retrieval_results,
        "evidence": evidence,
        "evidence_package": evidence_package,
        "design": parsed,
        "raw_response": raw_response,
    }


def write_design_outputs(result: dict[str, Any], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parsed_query.json").write_text(json.dumps(result["query"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "retrieved_cards.jsonl", result["retrieval_results"])
    (out_dir / "evidence_package.json").write_text(json.dumps(result["evidence_package"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "design.json").write_text(json.dumps(result["design"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "raw_response.txt").write_text(str(result["raw_response"]), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")


def _evidence_row(retrieval_row: dict[str, Any], semantics: dict[str, Any] | None) -> dict[str, Any]:
    semantics = semantics or {}
    return {
        "rank": retrieval_row.get("rank"),
        "score": retrieval_row.get("score"),
        "reasons": retrieval_row.get("reasons", []),
        "card_id": retrieval_row.get("card_id"),
        "name": retrieval_row.get("card_name"),
        "collectible": semantics.get("collectible", retrieval_row.get("collectible")),
        "identity": semantics.get("identity", {}),
        "stats": semantics.get("stats", {}),
        "text": (semantics.get("text") or {}).get("clean") or retrieval_row.get("text", ""),
        "keywords": semantics.get("keywords", []),
        "actions": semantics.get("actions", []),
        "mechanic_tags": semantics.get("mechanic_tags", []),
        "semantic_summary": semantics.get("semantic_summary", ""),
    }


def build_evidence_package(*, query: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Split KG evidence into design facets instead of one global rank.

    A single score-sorted list over-rewards repeated overlaps. For design, one
    lower-ranked family card can be more useful than five near-duplicate armor
    cards. The package keeps separate evidence strings for mechanics, identity,
    named/family anchors, and a diversified shortlist.
    """

    family = _facet_rows(
        evidence,
        predicate=lambda row: _is_family_or_named_anchor(row, query),
        limit=8,
    )
    mechanics = _facet_rows(
        evidence,
        predicate=lambda row: _has_reason_prefix(row, ("actions=", "resources=", "keywords=", "mechanic_tags=")),
        limit=8,
    )
    identity = _facet_rows(
        evidence,
        predicate=lambda row: _has_reason_prefix(row, ("classes=", "card_types=", "minion_types=", "spell_schools=")),
        limit=8,
    )
    diversified = _diverse_rows([*family, *mechanics, *identity, *evidence], limit=12)
    return {
        "retrieval_policy": {
            "ranking_note": "Do not treat rank as the only importance signal. Use high-score mechanic matches for mechanics, and family/name matches as style/statline anchors even if their score is lower.",
            "lora_reference_note": "For image generation, prefer one primary reference plus prompt-only semantic hints. If img2img is used, use a higher denoise strength around 0.75-0.85 so the result is influenced by the reference without becoming a literal fusion of multiple references.",
        },
        "facets": {
            "family_or_named_anchors": family,
            "mechanic_matches": mechanics,
            "identity_matches": identity,
            "diversified_shortlist": diversified,
        },
    }


def _add_related_name_matches(
    cards: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    related_names = [str(name).lower() for name in query.get("related_card_names", []) if str(name).strip()]
    if not related_names:
        return results
    existing_ids = {row.get("card_id") for row in results}
    additions = []
    for card in cards:
        card_id = card.get("card_id")
        if card_id in existing_ids:
            continue
        name = str(card.get("name", ""))
        lowered = name.lower()
        if not any(related in lowered for related in related_names):
            continue
        additions.append(
            {
                "query_id": query.get("query_id", ""),
                "query_text": query.get("text", ""),
                "method": "semantic_kg_related_name_expansion",
                "card_id": card_id,
                "card_name": name,
                "image": card.get("image", ""),
                "score": 0.0,
                "reasons": ["related_name_match"],
            }
        )
    # Keep the semantic top-k and append explicit family/name matches. This is
    # useful for meme/card-family requests where exact names are design anchors
    # but not necessarily high-scoring mechanic matches.
    combined = results[:top_k] + sorted(additions, key=lambda row: (str(row.get("card_name", "")), int(row.get("card_id") or 0)))[:10]
    for idx, row in enumerate(combined, start=1):
        row["rank"] = idx
    return combined


def _build_design_prompt(*, request_text: str, query: dict[str, Any], evidence_package: dict[str, Any]) -> str:
    return f"""You are designing one new custom Hearthstone card from a user's natural-language request.
Use the retrieved KG evidence as design context, not as a list to copy blindly.

Return strict JSON only. Do not use markdown.

Design goals:
- Produce a plausible Hearthstone card, not just an artwork prompt.
- Respect Hearthstone conventions: mana, attack, health, rarity, class, card type, keywords, concise rules text, flavor text.
- If the request references a meme family, preserve the recognizable meme pattern while making a playable DIY card.
- Use KG evidence to justify the design. Mention which retrieved cards influenced stats/mechanics.
- Avoid direct power creep unless the user explicitly asks for it.
- Prefer one clean mechanic over a complicated custom essay.
- The KG evidence is multi-facet. Do not blindly copy rank #1. A lower-ranked family/name anchor may be more important than several near-duplicate high-score mechanic matches.
- For LoRA/image reference, choose at most one primary visual reference and use the rest as text-only semantic hints.

User request:
{request_text}

Structured retrieval query:
{json.dumps(query, ensure_ascii=False, indent=2)}

Retrieved KG evidence package:
{json.dumps(evidence_package, ensure_ascii=False, indent=2)}

Return this JSON schema:
{{
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
    "rules_text": "short Hearthstone-style text",
    "flavor_text": "short optional flavor text"
  }},
  "structured_semantics": {{
    "actions": [],
    "mechanic_tags": [],
    "constraints": [],
    "related_card_refs": [],
    "semantic_summary": "one sentence",
    "lora_caption": "Hearthstone card art, ..."
  }},
  "kg_usage": {{
    "retrieved_cards_used": [
      {{"card_id": 1653, "name": "Magma Rager", "used_for": "statline/meme family"}}
    ],
    "design_rationale": "concise explanation"
  }},
  "balance_notes": "concise balance note"
}}
"""


def _summary_markdown(result: dict[str, Any]) -> str:
    design = result.get("design", {})
    card = design.get("card", {}) if isinstance(design, dict) else {}
    lines = [
        "# KG-Augmented Card Design",
        "",
        f"Request: {result.get('request', '')}",
        "",
        "## Parsed Query",
        "",
        "```json",
        json.dumps(result.get("query", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Retrieved Cards",
        "",
    ]
    for row in result.get("retrieval_results", []):
        lines.append(f"- #{row.get('rank')} {row.get('card_name')} ({row.get('card_id')}), score={row.get('score')}, reasons={'; '.join(row.get('reasons', []))}")
    lines.extend(
        [
            "",
            "## Evidence Package",
            "",
            "```json",
            json.dumps(result.get("evidence_package", {}), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Designed Card",
            "",
            f"**{card.get('name', '')}**",
            "",
            f"Cost/Class/Type: {card.get('mana_cost')} / {', '.join(card.get('class', []) or [])} / {card.get('card_type')}",
            "",
            f"Stats: {card.get('attack')}/{card.get('health')}",
            "",
            f"Text: {card.get('rules_text', '')}",
            "",
            f"Flavor: {card.get('flavor_text', '')}",
            "",
            "## Full Design JSON",
            "",
            "```json",
            json.dumps(design, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _facet_rows(evidence: list[dict[str, Any]], *, predicate, limit: int) -> list[dict[str, Any]]:
    return _diverse_rows([row for row in evidence if predicate(row)], limit=limit)


def _diverse_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected = []
    seen_cards = set()
    seen_signatures = set()
    for row in rows:
        card_id = row.get("card_id")
        if card_id in seen_cards:
            continue
        signature = _reason_signature(row)
        if signature in seen_signatures:
            continue
        selected.append(row)
        seen_cards.add(card_id)
        seen_signatures.add(signature)
        if len(selected) >= limit:
            break
    return selected


def _reason_signature(row: dict[str, Any]) -> tuple[str, ...]:
    reasons = row.get("reasons") or []
    normalized = []
    for reason in reasons:
        text = str(reason)
        if text.startswith("text_overlap="):
            continue
        normalized.append(text)
    if not normalized:
        normalized = [str(row.get("name") or row.get("card_name") or "")]
    return tuple(sorted(normalized))


def _has_reason_prefix(row: dict[str, Any], prefixes: tuple[str, ...]) -> bool:
    return any(str(reason).startswith(prefixes) for reason in row.get("reasons") or [])


def _is_family_or_named_anchor(row: dict[str, Any], query: dict[str, Any]) -> bool:
    name = str(row.get("name") or row.get("card_name") or "").lower()
    related = [str(item).lower() for item in query.get("related_card_names") or [] if str(item).strip()]
    if any(item in name for item in related):
        return True
    return "related_name_match" in {str(reason) for reason in row.get("reasons") or []}
