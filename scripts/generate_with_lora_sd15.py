#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample image with a trained SD 1.5 LoRA adapter.")
    parser.add_argument("--pretrained-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--lora-dir", type=Path, default=Path("models/sd15-hearthstone-lora"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, low quality, text, watermark, cropped")
    parser.add_argument("--output", type=Path, default=Path("outputs/lora_sample.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = _require_generation_deps()
    torch = deps.torch
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    pipe = deps.StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_lora_weights(str(args.lora_dir))
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
        cross_attention_kwargs={"scale": args.lora_scale},
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved: {args.output}")


def _require_generation_deps():
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit("Missing generation dependencies. Install torch and diffusers.") from exc
    return argparse.Namespace(StableDiffusionPipeline=StableDiffusionPipeline, torch=torch)


if __name__ == "__main__":
    main()

