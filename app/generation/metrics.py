from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NUMERIC_METRICS = [
    "image_quality_score",
    "clip_prompt_alignment",
    "style_similarity",
    "reference_similarity",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method:
            grouped[method].append(row)

    summary: list[dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        item: dict[str, Any] = {"method": method, "rows": len(method_rows)}
        for metric in NUMERIC_METRICS:
            item[f"{metric}_mean"] = _mean(method_rows, metric)
        summary.append(item)
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize_metric_rows(rows)
    fieldnames = ["method", "rows"] + [f"{metric}_mean" for metric in NUMERIC_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def collect_style_reference_paths(
    *,
    metadata_path: Path | None,
    image_root: Path,
    limit: int,
) -> list[Path]:
    if metadata_path is None or not metadata_path.exists():
        candidates = sorted((image_root / "images").glob("*"))
    else:
        candidates = []
        for row in read_jsonl(metadata_path):
            file_name = row.get("file_name") or row.get("image")
            if isinstance(file_name, str) and file_name.strip():
                candidates.append(image_root / file_name)
    return [path for path in candidates if path.exists()][:limit]


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 6) if values else 0.0
