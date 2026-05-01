from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.kg.io import read_jsonl


JUDGING_FIELDS = [
    "query_id",
    "method",
    "rank",
    "card_id",
    "card_name",
    "query_text",
    "image",
    "score",
    "class_match",
    "action_match",
    "keyword_match",
    "overall_relevance",
    "notes",
]


def read_result_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_judging_template(results: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGING_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r.get("query_id", ""), r.get("method", ""), int(r.get("rank") or 0))):
            writer.writerow(
                {
                    "query_id": row.get("query_id", ""),
                    "method": row.get("method", ""),
                    "rank": row.get("rank", ""),
                    "card_id": row.get("card_id", ""),
                    "card_name": row.get("card_name", ""),
                    "query_text": row.get("query_text", ""),
                    "image": row.get("image", ""),
                    "score": row.get("score", ""),
                    "class_match": "",
                    "action_match": "",
                    "keyword_match": "",
                    "overall_relevance": "",
                    "notes": "",
                }
            )


def summarize_judging(input_path: Path, out_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)

    summary = []
    for method, method_rows in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "rows": len(method_rows),
                "class_match_at_5": _mean(method_rows, "class_match"),
                "action_match_at_5": _mean(method_rows, "action_match"),
                "keyword_match_at_5": _mean(method_rows, "keyword_match"),
                "overall_relevance_at_5": _mean(method_rows, "overall_relevance"),
                "hit_at_5": _hit_at_5(method_rows),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["method"])
        writer.writeheader()
        writer.writerows(summary)
    return summary


def render_grid(results: list[dict[str, Any]], *, out_path: Path, image_root: Path) -> None:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_query[str(row.get("query_id", ""))].append(row)

    sections = []
    for query_id, rows in sorted(by_query.items()):
        query_text = rows[0].get("query_text", "") if rows else ""
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(rows, key=lambda r: (r.get("method", ""), int(r.get("rank") or 0))):
            by_method[str(row.get("method", ""))].append(row)
        method_columns = []
        for method, method_rows in sorted(by_method.items()):
            cards = "\n".join(_card_html(row, image_root=image_root) for row in method_rows)
            method_columns.append(f"<section class='method'><h3>{html.escape(method)}</h3>{cards}</section>")
        sections.append(
            f"<section class='query'><h2>{html.escape(query_id)}</h2><p>{html.escape(str(query_text))}</p><div class='methods'>{''.join(method_columns)}</div></section>"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Retrieval Evaluation Grid</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #f7f3ea; color: #1f2933; }
.query { margin-bottom: 36px; padding: 20px; background: white; border: 1px solid #ddd2bd; border-radius: 16px; }
.methods { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
.method { background: #fbfaf7; border-radius: 12px; padding: 12px; }
.card { display: grid; grid-template-columns: 88px 1fr; gap: 10px; margin: 10px 0; align-items: start; }
.card img { width: 88px; height: 88px; object-fit: cover; border-radius: 8px; background: #e5e7eb; }
.meta { font-size: 13px; line-height: 1.35; }
.name { font-weight: 700; }
.reason { color: #58616f; font-size: 12px; }
</style>
</head>
<body>
<h1>Retrieval Evaluation Grid</h1>
""" + "\n".join(sections) + "\n</body>\n</html>\n",
        encoding="utf-8",
    )


def _card_html(row: dict[str, Any], *, image_root: Path) -> str:
    image = str(row.get("image", ""))
    image_path = image_root / image
    image_src = image_path.as_posix() if image_path.exists() else ""
    reasons = row.get("reasons") or []
    if isinstance(reasons, list):
        reason_text = "; ".join(str(item) for item in reasons[:3])
    else:
        reason_text = str(reasons)
    return f"""<div class="card">
<img src="{html.escape(image_src)}" alt="">
<div class="meta">
<div class="name">#{html.escape(str(row.get("rank", "")))} {html.escape(str(row.get("card_name", "")))}</div>
<div>card_id={html.escape(str(row.get("card_id", "")))} score={html.escape(str(row.get("score", "")))}</div>
<div class="reason">{html.escape(reason_text)}</div>
</div>
</div>"""


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value == "":
            continue
        values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


def _hit_at_5(rows: list[dict[str, str]]) -> float:
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_query[row.get("query_id", "")].append(row)
    hits = []
    for query_rows in by_query.values():
        hit = any(float(row.get("overall_relevance") or 0) >= 2 for row in query_rows)
        hits.append(1.0 if hit else 0.0)
    return round(sum(hits) / len(hits), 4) if hits else 0.0
