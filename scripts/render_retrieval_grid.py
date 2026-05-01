#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render retrieval results into a simple HTML comparison grid.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_eval/retrieval_grid.html"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row.get("query_id", ""))][str(row.get("method", ""))].append(row)
    for methods in grouped.values():
        for method_rows in methods.values():
            method_rows.sort(key=lambda row: int(row.get("rank") or 0))
    return grouped


def image_src(row: dict[str, Any], *, output_path: Path, image_root: Path) -> str:
    image = str(row.get("image") or "").strip()
    if not image:
        return ""
    path = image_root / image
    relative = os.path.relpath(path.resolve(), output_path.parent.resolve())
    return html.escape(Path(relative).as_posix())


def render_card(row: dict[str, Any], *, output_path: Path, image_root: Path) -> str:
    src = image_src(row, output_path=output_path, image_root=image_root)
    img = f'<img src="{src}" alt="">' if src else '<div class="missing">No image path</div>'
    reasons = row.get("reasons") or row.get("matched_terms") or ""
    return f"""
      <article class="card">
        {img}
        <div class="meta">
          <div class="rank">#{html.escape(str(row.get("rank", "")))} · score {html.escape(str(row.get("score", "")))}</div>
          <h4>{html.escape(str(row.get("card_name", "")))}</h4>
          <p>{html.escape(str(row.get("caption", "")))}</p>
          <p class="reasons">{html.escape(str(reasons))}</p>
        </div>
      </article>
    """


def render_html(rows: list[dict[str, Any]], *, output_path: Path, image_root: Path) -> str:
    grouped = group_rows(rows)
    sections = []
    for query_id in sorted(grouped):
        methods = grouped[query_id]
        query_text = next((row.get("query_text", "") for rows in methods.values() for row in rows), "")
        method_blocks = []
        for method in sorted(methods):
            cards = "\n".join(render_card(row, output_path=output_path, image_root=image_root) for row in methods[method])
            method_blocks.append(
                f"""
                <section class="method">
                  <h3>{html.escape(method)}</h3>
                  <div class="cards">{cards}</div>
                </section>
                """
            )
        sections.append(
            f"""
            <section class="query">
              <h2>{html.escape(query_id)}</h2>
              <p class="query-text">{html.escape(str(query_text))}</p>
              <div class="methods">{''.join(method_blocks)}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Retrieval Evaluation Grid</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1 {{ margin-bottom: 4px; }}
    .query {{ margin: 28px 0; padding: 20px; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; }}
    .query-text {{ color: #4b5563; }}
    .methods {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .method h3 {{ margin: 0 0 10px; }}
    .cards {{ display: grid; gap: 12px; }}
    .card {{ display: grid; grid-template-columns: 96px 1fr; gap: 12px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #ffffff; }}
    .card img {{ width: 96px; min-height: 96px; object-fit: cover; border-radius: 6px; background: #e5e7eb; }}
    .missing {{ width: 96px; min-height: 96px; display: grid; place-items: center; background: #e5e7eb; border-radius: 6px; color: #6b7280; font-size: 12px; text-align: center; }}
    .rank {{ color: #6b7280; font-size: 12px; }}
    h4 {{ margin: 3px 0 6px; }}
    p {{ margin: 0 0 6px; line-height: 1.35; }}
    .reasons {{ color: #0369a1; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Retrieval Evaluation Grid</h1>
  {''.join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(rows, output_path=args.out, image_root=args.image_root), encoding="utf-8")
    print(f"Wrote HTML grid to {args.out}")


if __name__ == "__main__":
    main()
