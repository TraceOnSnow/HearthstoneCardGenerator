from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.kg.io import read_jsonl, write_jsonl


GENERATION_METHODS = [
    "sd_text_only",
    "sd_reference",
    "lora_text_only",
    "lora_reference",
]

JUDGING_FIELDS = [
    "prompt_id",
    "method",
    "prompt",
    "reference_method",
    "reference_card_name",
    "reference_image",
    "output_image",
    "prompt_alignment",
    "hearthstone_style",
    "reference_consistency",
    "overall_quality",
    "notes",
]


@dataclass(frozen=True)
class GenerationPrompt:
    prompt_id: str
    query_id: str
    prompt: str
    negative_prompt: str | None = None


def load_generation_prompts(path: Path) -> list[GenerationPrompt]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", data if isinstance(data, list) else [])
    rows: list[GenerationPrompt] = []
    for item in prompts:
        rows.append(
            GenerationPrompt(
                prompt_id=str(item["prompt_id"]),
                query_id=str(item.get("query_id") or item["prompt_id"]),
                prompt=str(item["prompt"]),
                negative_prompt=item.get("negative_prompt"),
            )
        )
    return rows


def load_top_references(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            query_id = str(row.get("query_id", ""))
            method = str(row.get("method", ""))
            rank = int(row.get("rank") or 0)
            if rank != 1:
                continue
            refs[(query_id, method)] = row
    return refs


def build_generation_plan(
    *,
    prompts: list[GenerationPrompt],
    refs: dict[tuple[str, str], dict[str, Any]],
    reference_method: str,
    out_dir: Path,
    image_root: Path,
    negative_prompt: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        ref = refs.get((prompt.query_id, reference_method), {})
        reference_image = str(ref.get("image", ""))
        reference_path = image_root / reference_image if reference_image else Path()
        reference_exists = bool(reference_image and reference_path.exists())

        for method in GENERATION_METHODS:
            output_image = out_dir / "images" / prompt.prompt_id / f"{method}.png"
            uses_lora = method.startswith("lora")
            uses_reference = method.endswith("reference")
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "query_id": prompt.query_id,
                    "method": method,
                    "prompt": prompt.prompt,
                    "negative_prompt": prompt.negative_prompt or negative_prompt,
                    "seed": seed,
                    "uses_lora": uses_lora,
                    "uses_reference": uses_reference,
                    "reference_method": reference_method if uses_reference else "",
                    "reference_card_id": ref.get("card_id", "") if uses_reference else "",
                    "reference_card_name": ref.get("card_name", "") if uses_reference else "",
                    "reference_image": reference_image if uses_reference else "",
                    "reference_path": reference_path.as_posix() if uses_reference and reference_exists else "",
                    "reference_score": ref.get("score", "") if uses_reference else "",
                    "output_image": output_image.as_posix(),
                    "status": "planned",
                }
            )
    return rows


def write_generation_judging_template(plan_rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGING_FIELDS)
        writer.writeheader()
        for row in plan_rows:
            writer.writerow(
                {
                    "prompt_id": row.get("prompt_id", ""),
                    "method": row.get("method", ""),
                    "prompt": row.get("prompt", ""),
                    "reference_method": row.get("reference_method", ""),
                    "reference_card_name": row.get("reference_card_name", ""),
                    "reference_image": row.get("reference_image", ""),
                    "output_image": row.get("output_image", ""),
                    "prompt_alignment": "",
                    "hearthstone_style": "",
                    "reference_consistency": "",
                    "overall_quality": "",
                    "notes": "",
                }
            )


def render_generation_grid(plan_rows: list[dict[str, Any]], *, out_path: Path) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in plan_rows:
        by_prompt.setdefault(str(row["prompt_id"]), []).append(row)

    sections = []
    for prompt_id, rows in by_prompt.items():
        prompt = rows[0].get("prompt", "") if rows else ""
        cards = "\n".join(_generation_card_html(row) for row in rows)
        sections.append(
            f"<section class='prompt'><h2>{html.escape(prompt_id)}</h2><p>{html.escape(str(prompt))}</p><div class='methods'>{cards}</div></section>"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Generation Comparison Grid</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #f5f1e8; color: #1f2933; }
.prompt { margin-bottom: 36px; padding: 20px; background: white; border: 1px solid #d8cbb2; border-radius: 16px; }
.methods { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
.method { background: #fbfaf6; border-radius: 12px; padding: 12px; border: 1px solid #eadfca; }
.method img.main { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 10px; background: #e5e7eb; }
.ref { display: grid; grid-template-columns: 48px 1fr; gap: 8px; align-items: center; margin-top: 8px; font-size: 12px; color: #58616f; }
.ref img { width: 48px; height: 48px; object-fit: cover; border-radius: 6px; background: #e5e7eb; }
.missing { display: grid; place-items: center; width: 100%; aspect-ratio: 1 / 1; border-radius: 10px; background: #ece6d8; color: #766b5c; font-size: 13px; text-align: center; }
h3 { margin: 0 0 8px; }
</style>
</head>
<body>
<h1>Generation Comparison Grid</h1>
""" + "\n".join(sections) + "\n</body>\n</html>\n",
        encoding="utf-8",
    )


def make_contact_sheet(plan_rows: list[dict[str, Any]], *, out_path: Path, cell_size: int = 256) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    prompt_ids = list(dict.fromkeys(str(row["prompt_id"]) for row in plan_rows))
    methods = GENERATION_METHODS
    width = cell_size * len(methods)
    header_h = 80
    row_h = cell_size + 70
    height = header_h + row_h * len(prompt_ids)
    sheet = Image.new("RGB", (width, height), (245, 241, 232))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for col, method in enumerate(methods):
        draw.text((col * cell_size + 8, 12), method, fill=(31, 41, 51), font=font)

    rows_by_key = {(str(row["prompt_id"]), str(row["method"])): row for row in plan_rows}
    for row_idx, prompt_id in enumerate(prompt_ids):
        y = header_h + row_idx * row_h
        draw.text((8, y + 4), prompt_id, fill=(31, 41, 51), font=font)
        for col, method in enumerate(methods):
            row = rows_by_key.get((prompt_id, method), {})
            image_path = Path(str(row.get("output_image", "")))
            x = col * cell_size
            if image_path.exists():
                image = Image.open(image_path).convert("RGB").resize((cell_size, cell_size))
                sheet.paste(image, (x, y + 28))
            else:
                draw.rectangle((x, y + 28, x + cell_size - 1, y + 28 + cell_size - 1), fill=(236, 230, 216), outline=(210, 199, 178))
                draw.text((x + 16, y + 28 + cell_size // 2), "not generated", fill=(118, 107, 92), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def _generation_card_html(row: dict[str, Any]) -> str:
    output = Path(str(row.get("output_image", "")))
    if output.exists():
        main = f'<img class="main" src="{html.escape(output.as_posix())}" alt="">'
    else:
        main = f'<div class="missing">not generated<br>{html.escape(str(row.get("status", "planned")))}</div>'

    ref_html = ""
    reference_path = str(row.get("reference_path", ""))
    if reference_path:
        ref_html = f"""<div class="ref">
<img src="{html.escape(reference_path)}" alt="">
<div>{html.escape(str(row.get("reference_card_name", "")))}<br>{html.escape(str(row.get("reference_method", "")))}</div>
</div>"""
    return f"""<section class="method">
<h3>{html.escape(str(row.get("method", "")))}</h3>
{main}
{ref_html}
</section>"""
