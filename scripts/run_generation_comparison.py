#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.comparison import (  # noqa: E402
    GENERATION_METHODS,
    build_generation_plan,
    load_generation_prompts,
    load_top_references,
    make_contact_sheet,
    render_generation_grid,
    write_generation_judging_template,
    write_prompt_review_template,
    write_plan,
)


DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, blurry, low quality, cropped, frame, card border, UI, typography"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run side-by-side SD/LoRA generation comparison.")
    parser.add_argument("--prompts", type=Path, default=Path("configs/generation_prompts.json"))
    parser.add_argument("--retrieval-results", type=Path, nargs="+", default=[Path("results/retrieval_eval/kg_results.jsonl")])
    parser.add_argument("--reference-method", default="semantic_kg")
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/generation_eval"))
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", type=Path, default=Path("models/lora/pytorch_lora_weights.safetensors"))
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.55, help="Img2img denoise strength for reference-conditioned variants.")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Only process the first N prompts.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=GENERATION_METHODS,
        default=GENERATION_METHODS,
        help="Generation methods to include. Use `--methods sd_text_only lora_text_only` for LoRA-vs-base text-only.",
    )
    parser.add_argument("--run", action="store_true", help="Actually run diffusion. Without this, only writes plan, HTML, contact sheet, and prompt review CSV.")
    parser.add_argument("--write-judging-template", action="store_true", help="Also write the old human scoring template.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if output files already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_generation_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    refs = load_top_references(args.retrieval_results)
    rows = build_generation_plan(
        prompts=prompts,
        refs=refs,
        reference_method=args.reference_method,
        out_dir=args.out_dir,
        image_root=args.image_root,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    rows = [row for row in rows if row["method"] in set(args.methods)]

    if args.run:
        rows = _run_generation(rows, args)

    write_plan(args.out_dir / "generation_plan.jsonl", rows)
    write_prompt_review_template(prompts, args.out_dir / "prompt_review.csv")
    if args.write_judging_template:
        write_generation_judging_template(rows, args.out_dir / "generation_judging_template.csv")
    render_generation_grid(rows, out_path=args.out_dir / "generation_grid.html")
    make_contact_sheet(rows, out_path=args.out_dir / "generation_contact_sheet.png")
    print(f"prompts={len(prompts)}")
    print(f"rows={len(rows)}")
    print(f"plan={args.out_dir / 'generation_plan.jsonl'}")
    print(f"grid={args.out_dir / 'generation_grid.html'}")
    print(f"prompt_review={args.out_dir / 'prompt_review.csv'}")
    if args.write_judging_template:
        print(f"judging={args.out_dir / 'generation_judging_template.csv'}")


def _run_generation(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    deps = _require_diffusion_deps()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation script in the current WSL setup.")
    if not args.lora.exists():
        raise FileNotFoundError(args.lora)

    device = "cuda"
    dtype = torch.float16
    base_rows = [row for row in rows if not row["uses_lora"]]
    lora_rows = [row for row in rows if row["uses_lora"]]

    text_pipe = deps["StableDiffusionPipeline"].from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    text_pipe.enable_attention_slicing()
    img_pipe = deps["StableDiffusionImg2ImgPipeline"].from_pipe(text_pipe).to(device)
    img_pipe.enable_attention_slicing()
    _run_rows(base_rows, args, torch, text_pipe, img_pipe)
    del img_pipe
    del text_pipe
    gc.collect()
    torch.cuda.empty_cache()

    lora_text_pipe = deps["StableDiffusionPipeline"].from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    lora_text_pipe.load_lora_weights(str(args.lora.parent), weight_name=args.lora.name)
    lora_text_pipe.fuse_lora(lora_scale=args.lora_scale)
    lora_text_pipe.to(device)
    lora_text_pipe.enable_attention_slicing()

    lora_img_pipe = deps["StableDiffusionImg2ImgPipeline"].from_pipe(lora_text_pipe).to(device)
    lora_img_pipe.enable_attention_slicing()
    _run_rows(lora_rows, args, torch, lora_text_pipe, lora_img_pipe)
    del lora_img_pipe
    del lora_text_pipe
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _run_rows(rows: list[dict[str, Any]], args: argparse.Namespace, torch: Any, text_pipe: Any, img_pipe: Any) -> None:
    device = "cuda"
    for row in rows:
        output = Path(str(row["output_image"]))
        if args.skip_existing and not args.overwrite and output.exists():
            row["status"] = "skipped_existing"
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
        prompt = str(row["prompt"])
        negative_prompt = str(row["negative_prompt"])
        try:
            if row["method"] == "sd_text_only":
                image = text_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                ).images[0]
            elif row["method"] == "sd_reference":
                image = _run_img2img(img_pipe, row, args, generator, prompt, negative_prompt)
            elif row["method"] == "lora_text_only":
                image = text_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                ).images[0]
            elif row["method"] == "lora_reference":
                image = _run_img2img(img_pipe, row, args, generator, prompt, negative_prompt)
            else:
                raise ValueError(f"Unknown method: {row['method']}")
            image.save(output)
            row["status"] = "generated"
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"failed: {exc}"


def _run_img2img(pipe: Any, row: dict[str, Any], args: argparse.Namespace, generator: Any, prompt: str, negative_prompt: str) -> Any:
    from PIL import Image

    reference_path = Path(str(row.get("reference_path", "")))
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image missing for {row['prompt_id']}: {reference_path}")
    image = Image.open(reference_path).convert("RGB").resize((args.width, args.height))
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]


def _require_diffusion_deps() -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit("Missing diffusion dependencies. Install the diffusion extra/deps first.") from exc
    return {
        "Image": Image,
        "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "torch": torch,
    }


if __name__ == "__main__":
    main()
