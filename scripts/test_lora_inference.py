#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one test image with the Hearthstone LoRA.")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", type=Path, default=Path("models/lora/pytorch_lora_weights.safetensors"))
    parser.add_argument("--out", type=Path, default=Path("results/lora_smoke/test_lora.png"))
    parser.add_argument(
        "--prompt",
        default="Hearthstone card art, Warlock Fel spell, dark fantasy magic, green demonic glow, high quality fantasy illustration",
    )
    parser.add_argument(
        "--negative-prompt",
        default="text, watermark, logo, blurry, low quality, cropped, frame, card border",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Stable Diffusion inference needs GPU here.")
    if not args.lora.exists():
        raise FileNotFoundError(args.lora)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(args.lora.parent), weight_name=args.lora.name)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
