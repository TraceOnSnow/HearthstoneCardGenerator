from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from app.kg_demo import config


@dataclass(slots=True)
class CardLite:
    id: int
    name: str
    text: str
    manaCost: int | None
    attack: int | None
    health: int | None


def load_card_lite_records(path: Path) -> list[CardLite]:
    cards: list[CardLite] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            card_id = raw.get("id")
            if not isinstance(card_id, int):
                continue
            cards.append(
                CardLite(
                    id=card_id,
                    name=str(raw.get("name", "")).strip(),
                    text=str(raw.get("text", "")).strip(),
                    manaCost=raw.get("manaCost"),
                    attack=raw.get("attack"),
                    health=raw.get("health"),
                )
            )
    return cards


def sample_cards(cards: list[CardLite], sample_size: int, seed: int) -> list[CardLite]:
    if sample_size >= len(cards):
        return cards
    rng = random.Random(seed)
    return rng.sample(cards, sample_size)


def chunked(items: list[CardLite], chunk_size: int) -> Iterable[list[CardLite]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, cards: list[CardLite]) -> str:
    cards_json = json.dumps([asdict(c) for c in cards], ensure_ascii=False, indent=2)
    return template.replace("{{CARDS_JSON}}", cards_json)


def _google_generate_content(prompt: str, api_key: str, model: str, temperature: float) -> str:
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    url = f"{base}?{urllib.parse.urlencode({'key': api_key})}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Empty candidates from model: {body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Missing content parts from model: {body}")

    text = parts[0].get("text", "")
    if not text:
        raise RuntimeError(f"Empty model text output: {body}")

    return text


def run_llm_batches(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    load_dotenv()
    outputs: list[dict[str, Any]] = []

    api_key = ""
    if not config.DRY_RUN:
        import os

        api_key = os.getenv(config.GOOGLE_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing env: {config.GOOGLE_API_KEY_ENV}. Set it in .env or shell."
            )

    for row in prompts:
        batch_id = row["batch_id"]
        prompt = row["prompt"]

        if config.DRY_RUN:
            result_text = json.dumps({"cards": []}, ensure_ascii=False)
            status = "dry_run"
            error = ""
        else:
            try:
                result_text = _google_generate_content(
                    prompt=prompt,
                    api_key=api_key,
                    model=config.GOOGLE_MODEL,
                    temperature=config.LLM_TEMPERATURE,
                )
                status = "ok"
                error = ""
            except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                result_text = ""
                status = "failed"
                error = str(exc)

        outputs.append(
            {
                "batch_id": batch_id,
                "status": status,
                "error": error,
                "raw_response": result_text,
            }
        )

    return outputs


def _normalize_name(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return norm or "unknown"


def build_graph_from_outputs(outputs: list[dict[str, Any]], sampled_cards: list[CardLite]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    # Ensure sampled cards always exist as graph nodes.
    for card in sampled_cards:
        card_node_id = f"card:{card.id}"
        nodes[card_node_id] = {
            "id": card_node_id,
            "type": "card",
            "name": card.name,
            "attributes": {
                "manaCost": card.manaCost,
                "attack": card.attack,
                "health": card.health,
            },
        }

    for out in outputs:
        if out.get("status") != "ok" and out.get("status") != "dry_run":
            continue

        raw = out.get("raw_response", "")
        if not raw:
            continue

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for item in parsed.get("cards", []):
            card_id = item.get("card_id")
            if not isinstance(card_id, int):
                continue
            card_node_id = f"card:{card_id}"
            if card_node_id not in nodes:
                nodes[card_node_id] = {
                    "id": card_node_id,
                    "type": "card",
                    "name": item.get("name", ""),
                    "attributes": item.get("attributes", {}),
                }

            for mechanic in item.get("mechanics", []):
                if not isinstance(mechanic, str) or not mechanic.strip():
                    continue
                mechanic_node_id = f"mechanic:{_normalize_name(mechanic)}"
                if mechanic_node_id not in nodes:
                    nodes[mechanic_node_id] = {
                        "id": mechanic_node_id,
                        "type": "mechanic",
                        "name": mechanic.strip(),
                    }
                edges.append(
                    {
                        "source": card_node_id,
                        "predicate": "HAS_MECHANIC",
                        "target": mechanic_node_id,
                    }
                )

            for entity in item.get("entities", []):
                entity_type = str(entity.get("type", "other")).strip() or "other"
                entity_name = str(entity.get("name", "")).strip()
                if not entity_name:
                    continue
                entity_node_id = f"entity:{entity_type}:{_normalize_name(entity_name)}"
                if entity_node_id not in nodes:
                    nodes[entity_node_id] = {
                        "id": entity_node_id,
                        "type": entity_type,
                        "name": entity_name,
                    }
                edges.append(
                    {
                        "source": card_node_id,
                        "predicate": "HAS_ENTITY",
                        "target": entity_node_id,
                    }
                )

    return {"nodes": list(nodes.values()), "edges": edges}


def run_pipeline() -> None:
    cards = load_card_lite_records(config.SOURCE_JSONL)
    sampled = sample_cards(cards, config.SAMPLE_SIZE, config.RANDOM_SEED)

    write_jsonl(config.SAMPLED_JSONL, (asdict(c) for c in sampled))

    template = load_prompt_template(config.PROMPT_TEMPLATE_PATH)
    prompt_rows: list[dict[str, Any]] = []
    for batch_idx, chunk in enumerate(chunked(sampled, config.CHUNK_SIZE), start=1):
        prompt_rows.append(
            {
                "batch_id": batch_idx,
                "card_count": len(chunk),
                "cards": [asdict(c) for c in chunk],
                "prompt": build_prompt(template, chunk),
            }
        )

    write_jsonl(config.PROMPTS_JSONL, prompt_rows)

    outputs = run_llm_batches(prompt_rows)
    write_jsonl(config.LLM_OUTPUT_JSONL, outputs)

    graph = build_graph_from_outputs(outputs, sampled)
    config.GRAPH_JSON.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done. sampled={len(sampled)} batches={len(prompt_rows)} "
        f"nodes={len(graph['nodes'])} edges={len(graph['edges'])} dry_run={config.DRY_RUN}"
    )
