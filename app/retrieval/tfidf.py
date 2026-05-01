from __future__ import annotations

import math
from collections import Counter
from typing import Any

from app.retrieval.common import query_to_text, result_row, tokenize


def retrieve_tfidf(
    *,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    docs = [_doc_text(row) for row in corpus]
    doc_tokens = [tokenize(text) for text in docs]
    idf = _idf(doc_tokens)
    doc_vectors = [_tfidf(tokens, idf) for tokens in doc_tokens]
    doc_norms = [_norm(vector) for vector in doc_vectors]

    rows: list[dict[str, Any]] = []
    for query in queries:
        query_vector = _tfidf(tokenize(query_to_text(query)), idf)
        query_norm = _norm(query_vector)
        scored = []
        for card, doc_vector, doc_norm in zip(corpus, doc_vectors, doc_norms, strict=True):
            score = _cosine(query_vector, query_norm, doc_vector, doc_norm)
            if score <= 0:
                continue
            overlap = sorted(set(query_vector) & set(doc_vector))
            scored.append((score, card, overlap))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("card_name")), int(item[1].get("card_id") or 0)))
        for rank, (score, card, overlap) in enumerate(scored[:top_k], start=1):
            rows.append(
                result_row(
                    query=query,
                    method="tfidf_baseline",
                    rank=rank,
                    card=card,
                    score=score,
                    reasons=[f"token_overlap={','.join(overlap[:12])}"] if overlap else [],
                )
            )
    return rows


def _doc_text(row: dict[str, Any]) -> str:
    return " ".join([str(row.get("card_name", "")), str(row.get("caption", ""))])


def _idf(docs: list[list[str]]) -> dict[str, float]:
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))
    total = max(len(docs), 1)
    return {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in df.items()}


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)
    return {token: (count / total) * idf.get(token, 1.0) for token, count in counts.items()}


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(a: dict[str, float], a_norm: float, b: dict[str, float], b_norm: float) -> float:
    if a_norm <= 0 or b_norm <= 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    return dot / (a_norm * b_norm)
