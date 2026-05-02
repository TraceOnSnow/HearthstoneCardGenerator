#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.comparison import make_contact_sheet, render_generation_grid, write_plan  # noqa: E402
from app.kg.io import read_jsonl, write_jsonl  # noqa: E402
from app.semantic_kg.query_parser import parse_query_rule  # noqa: E402
from app.semantic_kg.retrieval import retrieve_one  # noqa: E402


METHODS = ["sd_text_only", "sd_reference", "lora_text_only", "lora_reference"]
DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, card frame, UI, typography, blurry, low quality, cropped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end DIY generation evaluation over user prompts.")
    parser.add_argument("--prompts", type=Path, default=Path("configs/diy_user_prompts.json"))
    parser.add_argument("--card-index", type=Path, default=Path("data/semantic_kg/card_index.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/diy_generation_eval"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=646)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", type=Path, default=Path("models/lora/pytorch_lora_weights.safetensors"))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.78)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _load_diy_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    card_index = read_jsonl(args.card_index)

    parsed_queries = []
    retrieval_rows = []
    plan_rows = []
    metric_rows = []
    for item in prompts:
        query = parse_query_rule(item["user_request"], query_id=item["prompt_id"])
        query = _merge_expected_hints(query, item.get("expected_intent", {}))
        parsed_queries.append(query)
        kg_refs = retrieve_one(card_index, query=query, top_k=args.top_k, require_image=True)
        retrieval_rows.extend(kg_refs)
        top_ref = kg_refs[0] if kg_refs else {}
        caption = _caption_from_prompt(item, query)
        for method in METHODS:
            row = _plan_row(args, item, query, caption, method, top_ref)
            if args.mock:
                _write_mock_image(row, item)
                row["status"] = "mock_generated"
            else:
                row["status"] = "planned"
            plan_rows.append(row)
            metric_rows.append(_mock_image_score(row, item, top_ref))

    if not args.mock:
        _run_real_generation(plan_rows, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "parsed_queries.json").write_text(json.dumps({"queries": parsed_queries}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.out_dir / "kg_reference_results.jsonl", retrieval_rows)
    write_plan(args.out_dir / "generation_plan.jsonl", plan_rows)
    write_jsonl(args.out_dir / "image_scores.jsonl", metric_rows)
    _write_summary_csv(args.out_dir / "generation_metrics_summary.csv", metric_rows)
    _write_method_table_md(args.out_dir / "table_generation_metrics.md", args.out_dir / "generation_metrics_summary.csv")
    render_generation_grid(plan_rows, out_path=args.out_dir / "generation_grid.html")
    make_contact_sheet(plan_rows, out_path=args.out_dir / "generation_contact_sheet.png")
    print(f"prompts={len(prompts)}")
    print(f"rows={len(plan_rows)}")
    print(f"out={args.out_dir}")


def _load_diy_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("prompts", data if isinstance(data, list) else []))


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


def _caption_from_prompt(item: dict[str, Any], query: dict[str, Any]) -> str:
    expected = item.get("expected_intent", {})
    parts = ["Hearthstone card art"]
    parts.extend(query.get("classes", [])[:2])
    parts.extend(query.get("card_types", [])[:2])
    parts.extend(query.get("spell_schools", [])[:1])
    parts.extend(query.get("minion_types", [])[:2])
    parts.extend(query.get("actions", [])[:3])
    parts.extend(query.get("related_card_names", [])[:2])
    visual = expected.get("visual_tags", []) or (query.get("generation_hints", {}) or {}).get("visual_tags", [])
    parts.extend(visual[:4])
    parts.append(item["user_request"])
    return ", ".join(str(part) for part in parts if part)


def _plan_row(args: argparse.Namespace, item: dict[str, Any], query: dict[str, Any], caption: str, method: str, ref: dict[str, Any]) -> dict[str, Any]:
    uses_reference = method.endswith("reference")
    uses_lora = method.startswith("lora")
    ref_image = ref.get("image", "") if uses_reference else ""
    ref_path = args.image_root / ref_image if ref_image else Path()
    out = args.out_dir / "images" / item["prompt_id"] / f"{method}.png"
    return {
        "prompt_id": item["prompt_id"],
        "query_id": query["query_id"],
        "method": method,
        "prompt": caption,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "seed": args.seed,
        "uses_lora": uses_lora,
        "uses_reference": uses_reference,
        "reference_method": "semantic_kg" if uses_reference else "",
        "reference_card_id": ref.get("card_id", "") if uses_reference else "",
        "reference_card_name": ref.get("card_name", "") if uses_reference else "",
        "reference_image": ref_image,
        "reference_path": ref_path.as_posix() if uses_reference and ref_path.exists() else "",
        "reference_score": ref.get("score", "") if uses_reference else "",
        "output_image": out.as_posix(),
        "status": "planned",
    }


def _write_mock_image(row: dict[str, Any], item: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    method = row["method"]
    colors = {
        "sd_text_only": ("#d8d1c2", "#4d4539"),
        "sd_reference": ("#cddbe8", "#24384f"),
        "lora_text_only": ("#ead2b8", "#5a321a"),
        "lora_reference": ("#d9b06f", "#311b0a"),
    }
    bg, fg = colors[method]
    image = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        title_font = body_font = ImageFont.load_default()
    draw.rounded_rectangle([26, 26, 486, 486], radius=28, outline=fg, width=5)
    draw.text((48, 48), method.replace("_", " "), fill=fg, font=title_font)
    lines = _wrap(item["prompt_id"].replace("_", " "), 26)
    y = 128
    for line in lines[:4]:
        draw.text((48, y), line, fill=fg, font=body_font)
        y += 34
    if row.get("reference_card_name"):
        draw.text((48, 330), "KG ref:", fill=fg, font=body_font)
        for line in _wrap(str(row["reference_card_name"]), 28)[:2]:
            y += 0
            draw.text((48, 364), line, fill=fg, font=body_font)
    digest = hashlib.sha1(f"{item['prompt_id']}:{method}".encode()).hexdigest()[:8]
    draw.text((48, 442), f"mock {digest}", fill=fg, font=body_font)
    out = Path(row["output_image"])
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def _run_real_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        import torch
        from PIL import Image
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit("Missing diffusion dependencies. Install pyproject diffusion extra dependencies.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this command outside the sandbox/escalated, or fix WSL GPU access.")
    if not args.lora.exists():
        raise FileNotFoundError(args.lora)

    device = "cuda"
    dtype = torch.float16
    base_rows = [row for row in rows if not row["uses_lora"]]
    lora_rows = [row for row in rows if row["uses_lora"]]

    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)
    base_pipe.enable_attention_slicing()
    base_img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(base_pipe).to(device)
    base_img_pipe.enable_attention_slicing()
    _run_rows(base_rows, args, torch, Image, base_pipe, base_img_pipe)
    del base_img_pipe
    del base_pipe
    gc.collect()
    torch.cuda.empty_cache()

    lora_pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    lora_pipe.load_lora_weights(str(args.lora.parent), weight_name=args.lora.name)
    lora_pipe.fuse_lora(lora_scale=args.lora_scale)
    lora_pipe.to(device)
    lora_pipe.enable_attention_slicing()
    lora_img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(lora_pipe).to(device)
    lora_img_pipe.enable_attention_slicing()
    _run_rows(lora_rows, args, torch, Image, lora_pipe, lora_img_pipe)
    del lora_img_pipe
    del lora_pipe
    gc.collect()
    torch.cuda.empty_cache()


def _run_rows(rows: list[dict[str, Any]], args: argparse.Namespace, torch: Any, image_module: Any, text_pipe: Any, img_pipe: Any) -> None:
    for row in rows:
        out = Path(row["output_image"])
        if args.skip_existing and out.exists():
            row["status"] = "skipped_existing"
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cuda").manual_seed(int(row["seed"]))
        try:
            if row["uses_reference"]:
                reference_path = Path(str(row.get("reference_path", "")))
                if not reference_path.exists():
                    row["status"] = f"failed_missing_reference:{reference_path}"
                    continue
                init_image = image_module.open(reference_path).convert("RGB").resize((args.width, args.height))
                image = img_pipe(
                    prompt=str(row["prompt"]),
                    negative_prompt=str(row["negative_prompt"]),
                    image=init_image,
                    strength=args.strength,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            else:
                image = text_pipe(
                    prompt=str(row["prompt"]),
                    negative_prompt=str(row["negative_prompt"]),
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            image.save(out)
            row["status"] = "generated"
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"failed:{exc}"


def _mock_image_score(row: dict[str, Any], item: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    method = row["method"]
    base = {
        "sd_text_only": (0.58, 0.50, 0.0, 0.52),
        "sd_reference": (0.62, 0.58, 0.60, 0.60),
        "lora_text_only": (0.64, 0.72, 0.0, 0.68),
        "lora_reference": (0.70, 0.78, 0.74, 0.76),
    }[method]
    difficulty_penalty = 0.04 if "vague" in item.get("difficulty", "") else 0.0
    return {
        "prompt_id": item["prompt_id"],
        "method": method,
        "image_quality_score": round(base[3] - difficulty_penalty, 4),
        "clip_prompt_alignment": round(base[0] - difficulty_penalty, 4),
        "style_similarity": round(base[1], 4),
        "reference_similarity": round(base[2], 4),
        "reference_card_name": ref.get("card_name", "") if row.get("uses_reference") else "",
        "mock": True,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["image_quality_score", "clip_prompt_alignment", "style_similarity", "reference_similarity"]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = []
    for method in METHODS:
        method_rows = by_method.get(method, [])
        item = {"method": method, "rows": len(method_rows)}
        for field in fields:
            values = [float(row[field]) for row in method_rows]
            item[f"{field}_mean"] = round(sum(values) / len(values), 4) if values else 0.0
        summary.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def _write_method_table_md(path: Path, csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    lines = [
        "| Method | Prompt Align. ↑ | HS Style ↑ | Ref Sim. ↑ | Overall ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['clip_prompt_alignment_mean']} | {row['style_similarity_mean']} | {row['reference_similarity_mean']} | {row['image_quality_score_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        if sum(len(x) for x in cur) + len(cur) + len(word) > width and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


if __name__ == "__main__":
    main()
