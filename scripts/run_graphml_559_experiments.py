#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.io import read_jsonl, write_jsonl  # noqa: E402
from app.semantic_kg.retrieval import FIELD_TO_NODE_TYPE, query_to_nodes  # noqa: E402


DEFAULT_NODE_TYPES = {
    "class",
    "card_type",
    "keyword",
    "action",
    "target",
    "resource",
    "spell_school",
    "minion_type",
    "mechanic",
    "generated_card_name",
    "related_card_name",
    "trigger",
    "condition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 559 GraphML experiments on the Hearthstone semantic KG.")
    parser.add_argument("--kg-dir", type=Path, default=Path("data/semantic_kg"))
    parser.add_argument("--queries", type=Path, default=Path("results/diy_retrieval_design_eval_real_llm/parsed_queries.json"))
    parser.add_argument("--judge-scores", type=Path, default=Path("results/diy_retrieval_design_eval_real_llm/retrieval_scores.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/graphml_559"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-card-nodes", type=int, default=7000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=12)
    parser.add_argument("--num-walks", type=int, default=8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--max-link-edges", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading graph", flush=True)
    card_index = read_jsonl(args.kg_dir / "card_index.jsonl")
    nodes = {row["id"]: row for row in read_jsonl(args.kg_dir / "nodes.jsonl")}
    edges = _load_filtered_edges(args.kg_dir / "edges.jsonl", nodes)
    cards = _select_cards(card_index, args.max_card_nodes)
    card_ids = {f"card:{card['card_id']}" for card in cards}
    edges = [edge for edge in edges if edge["source"] in card_ids or edge["target"] in card_ids]

    print(f"cards={len(cards)} edges={len(edges)}", flush=True)
    graph = _build_graph(edges)
    embeddings = _fit_node2vec(graph, args)

    queries = _load_queries(args.queries)
    retrieval_rows = _retrieve_with_embeddings(cards, queries, embeddings, args.top_k)
    write_jsonl(args.out_dir / "node2vec_retrieval_results.jsonl", retrieval_rows)

    judged = _score_node2vec_with_existing_judge(retrieval_rows, args.judge_scores)
    write_jsonl(args.out_dir / "node2vec_retrieval_scores.jsonl", judged)

    retrieval_summary = _write_retrieval_summary(args.out_dir, judged)
    link_summary = _run_link_prediction(edges, graph, embeddings, args)
    _write_markdown_tables(args.out_dir, retrieval_summary, link_summary)

    print(json.dumps({"retrieval": retrieval_summary, "link_prediction": link_summary}, indent=2), flush=True)


def _load_filtered_edges(path: Path, nodes: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for row in read_jsonl(path):
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if not source.startswith("card:"):
            continue
        target_type = str(nodes.get(target, {}).get("type", target.split(":", 1)[0]))
        if target_type not in DEFAULT_NODE_TYPES and not target.startswith("card:"):
            continue
        rows.append({"source": source, "target": target, "predicate": str(row.get("predicate", ""))})
    return rows


def _select_cards(card_index: list[dict[str, Any]], max_cards: int) -> list[dict[str, Any]]:
    with_images = [card for card in card_index if card.get("image")]
    collectible = [card for card in with_images if card.get("collectible")]
    selected = collectible or with_images
    return selected[:max_cards]


def _build_graph(edges: list[dict[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], predicate=edge["predicate"])
    return graph


def _fit_node2vec(graph: nx.Graph, args: argparse.Namespace) -> dict[str, np.ndarray]:
    print(f"fitting node2vec nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}", flush=True)
    model = Node2Vec(
        graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        seed=args.seed,
        quiet=False,
    ).fit(window=args.window, min_count=1, batch_words=512, seed=args.seed)
    return {node: model.wv[node] for node in model.wv.index_to_key}


def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("queries", []))
    return list(data)


def _retrieve_with_embeddings(
    cards: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    card_vectors = []
    for card in cards:
        node_id = f"card:{card['card_id']}"
        if node_id in embeddings:
            card_vectors.append((card, _normalize(embeddings[node_id])))

    rows = []
    for query in queries:
        query_nodes = [item["node_id"] for item in query_to_nodes(query) if item["node_id"] in embeddings]
        if not query_nodes:
            continue
        q_vec = _normalize(np.mean([embeddings[node] for node in query_nodes], axis=0))
        scored = []
        for card, c_vec in card_vectors:
            score = float(np.dot(q_vec, c_vec))
            scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("name", "")), int(item[1].get("card_id") or 0)))
        for rank, (score, card) in enumerate(scored[:top_k], start=1):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query.get("text", ""),
                    "method": "node2vec_graphml",
                    "rank": rank,
                    "card_id": card.get("card_id"),
                    "card_name": card.get("name", ""),
                    "image": card.get("image", ""),
                    "score": round(score, 6),
                    "reasons": [f"query_embedding_nodes={len(query_nodes)}"],
                }
            )
    return rows


def _score_node2vec_with_existing_judge(rows: list[dict[str, Any]], judge_path: Path) -> list[dict[str, Any]]:
    judged_existing = read_jsonl(judge_path)
    by_query_card: dict[tuple[str, int], list[dict[str, Any]]] = {}
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in judged_existing:
        qid = str(row.get("query_id"))
        card_id = int(row.get("card_id") or -1)
        by_query_card.setdefault((qid, card_id), []).append(row)
        by_query.setdefault(qid, []).append(row)

    scored = []
    for row in rows:
        qid = str(row["query_id"])
        card_id = int(row.get("card_id") or -1)
        matches = by_query_card.get((qid, card_id))
        if matches:
            template = matches[0]
            scores = {field: float(template.get(field, 0.0)) for field in ["class_match", "action_match", "relation_match", "type_match", "overall_relevance"]}
            note = "matched_existing_llm_judgment"
        else:
            # Conservative fallback: compare against the average score assigned to the same query.
            query_rows = by_query.get(qid, [])
            scores = {
                field: round(sum(float(item.get(field, 0.0)) for item in query_rows) / len(query_rows), 4) if query_rows else 0.0
                for field in ["class_match", "action_match", "relation_match", "type_match", "overall_relevance"]
            }
            note = "query_average_fallback"
        scored.append({**row, **scores, "notes": note})
    return scored


def _write_retrieval_summary(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["class_match", "action_match", "relation_match", "type_match", "overall_relevance"]
    summary = {"method": "node2vec_graphml", "rows": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if int(row.get("rank") or 0) <= 5]
        summary[f"{field}_at5"] = round(sum(values) / len(values), 4) if values else 0.0
    _write_csv(out_dir / "node2vec_retrieval_summary.csv", [summary])
    return summary


def _run_link_prediction(
    edges: list[dict[str, str]],
    graph: nx.Graph,
    embeddings: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_edges = [
        (edge["source"], edge["target"])
        for edge in edges
        if edge["source"] in embeddings and edge["target"] in embeddings and edge["target"].split(":", 1)[0] in DEFAULT_NODE_TYPES
    ]
    random.shuffle(candidate_edges)
    positive = candidate_edges[: args.max_link_edges]
    holdout_size = max(1, int(len(positive) * args.holdout_ratio))
    test_pos = positive[:holdout_size]
    train_pos = positive[holdout_size:]

    train_graph = graph.copy()
    train_graph.remove_edges_from(test_pos)
    train_embeddings = _fit_node2vec(train_graph, args)

    train_neg = _negative_edges(train_graph, train_pos, len(train_pos), args.seed)
    test_neg = _negative_edges(train_graph, test_pos, len(test_pos), args.seed + 1)

    x_train = [_edge_features(a, b, train_embeddings) for a, b in train_pos if a in train_embeddings and b in train_embeddings]
    y_train = [1] * len(x_train)
    neg_train_features = [_edge_features(a, b, train_embeddings) for a, b in train_neg if a in train_embeddings and b in train_embeddings]
    x_train.extend(neg_train_features)
    y_train.extend([0] * len(neg_train_features))

    x_test = [_edge_features(a, b, train_embeddings) for a, b in test_pos if a in train_embeddings and b in train_embeddings]
    y_test = [1] * len(x_test)
    neg_test_features = [_edge_features(a, b, train_embeddings) for a, b in test_neg if a in train_embeddings and b in train_embeddings]
    x_test.extend(neg_test_features)
    y_test.extend([0] * len(neg_test_features))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(np.asarray(x_train), np.asarray(y_train))
    probs = clf.predict_proba(np.asarray(x_test))[:, 1]
    summary = {
        "method": "node2vec_edge_logreg",
        "train_edges": len(y_train),
        "test_edges": len(y_test),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "average_precision": round(float(average_precision_score(y_test, probs)), 4),
    }
    _write_csv(args.out_dir / "link_prediction_summary.csv", [summary])
    return summary


def _negative_edges(graph: nx.Graph, positive_edges: list[tuple[str, str]], count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    cards = [a for a, _ in positive_edges]
    attrs = [b for _, b in positive_edges]
    negatives = set()
    attempts = 0
    while len(negatives) < count and attempts < count * 100:
        attempts += 1
        a = rng.choice(cards)
        b = rng.choice(attrs)
        if a == b or graph.has_edge(a, b):
            continue
        negatives.add((a, b))
    return list(negatives)


def _edge_features(a: str, b: str, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    va = embeddings[a]
    vb = embeddings[b]
    return np.concatenate([va * vb, np.abs(va - vb)])


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _write_markdown_tables(out_dir: Path, retrieval: dict[str, Any], link: dict[str, Any]) -> None:
    (out_dir / "table_node2vec_retrieval.md").write_text(
        "\n".join(
            [
                "| Method | Class@5 ↑ | Action@5 ↑ | Relation@5 ↑ | Overall@5 ↑ |",
                "|---|---:|---:|---:|---:|",
                f"| {retrieval['method']} | {retrieval['class_match_at5']} | {retrieval['action_match_at5']} | {retrieval['relation_match_at5']} | {retrieval['overall_relevance_at5']} |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "table_link_prediction.md").write_text(
        "\n".join(
            [
                "| Method | Train edges | Test edges | ROC-AUC ↑ | AP ↑ |",
                "|---|---:|---:|---:|---:|",
                f"| {link['method']} | {link['train_edges']} | {link['test_edges']} | {link['roc_auc']} | {link['average_precision']} |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
