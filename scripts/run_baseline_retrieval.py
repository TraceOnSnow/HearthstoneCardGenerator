#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


METHOD = "tfidf_baseline"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dependency-free TF-IDF retrieval over LoRA captions.")
    parser.add_argument("--queries", type=Path, default=Path("configs/retrieval_queries.json"))
    parser.add_argument("--captions", type=Path, default=Path("data/semantics/lora_captions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/baseline_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Query file must be a JSON list.")
    for row in rows:
        if not isinstance(row, dict) or not row.get("query_id") or not row.get("text"):
            raise ValueError("Each query must contain query_id and text.")
    return rows


def load_captions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Caption file not found: {path}. Run scripts/build_semantics.py first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("card_id") is not None and row.get("caption"):
                    rows.append(row)
    if not rows:
        raise ValueError(f"No caption rows found in {path}")
    return rows


def query_text(query: dict[str, Any]) -> str:
    parts = [str(query.get("text", ""))]
    for key in [
        "classes",
        "card_types",
        "keywords",
        "actions",
        "targets",
        "spell_schools",
        "mechanic_tags",
        "visual_tags",
    ]:
        value = query.get(key) or []
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return " ".join(parts)


def document_text(row: dict[str, Any]) -> str:
    return " ".join([str(row.get("name", "")), str(row.get("caption", ""))])


def tokenize(text: str) -> list[str]:
    normalized = text.replace("_", " ").replace("-", " ").lower()
    return TOKEN_RE.findall(normalized)


def build_index(captions: list[dict[str, Any]]) -> tuple[list[Counter[str]], dict[str, float]]:
    doc_terms = [Counter(tokenize(document_text(row))) for row in captions]
    dfs: Counter[str] = Counter()
    for terms in doc_terms:
        dfs.update(terms.keys())
    total_docs = len(doc_terms)
    idf = {term: math.log((total_docs + 1) / (df + 1)) + 1.0 for term, df in dfs.items()}
    return doc_terms, idf


def score(query_terms: Counter[str], doc_terms: Counter[str], idf: dict[str, float]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    total = 0.0
    for term, q_count in query_terms.items():
        d_count = doc_terms.get(term, 0)
        if not d_count:
            continue
        weight = idf.get(term, 1.0)
        total += (1.0 + math.log(q_count)) * (1.0 + math.log(d_count)) * weight * weight
    return total


def retrieve(
    queries: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    doc_terms, idf = build_index(captions)
    results: list[dict[str, Any]] = []
    for query in queries:
        q_text = query_text(query)
        q_terms = Counter(tokenize(q_text))
        scored = [
            (score(q_terms, terms, idf), idx)
            for idx, terms in enumerate(doc_terms)
        ]
        scored.sort(key=lambda item: (-item[0], captions[item[1]].get("card_id", 0)))
        for rank, (value, idx) in enumerate(scored[:top_k], start=1):
            row = captions[idx]
            results.append(
                {
                    "query_id": query["query_id"],
                    "method": METHOD,
                    "rank": rank,
                    "card_id": row.get("card_id"),
                    "card_name": row.get("name", ""),
                    "image": row.get("image", ""),
                    "score": round(value, 6),
                    "query_text": query.get("text", ""),
                    "caption": row.get("caption", ""),
                }
            )
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    captions = load_captions(args.captions)
    results = retrieve(queries, captions, top_k=args.top_k)
    write_jsonl(args.out, results)
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
