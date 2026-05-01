#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diffusion.lora_data import normalize_lora_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 1.5 with LoRA on Hearthstone art.")
    parser.add_argument("--pretrained-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None, help="Optional HF weight variant, e.g. fp16.")
    parser.add_argument("--metadata", type=Path, default=Path("data/hf_hearthstone_art_512/metadata.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--caption-column", default="text")
    parser.add_argument("--image-column", default="file_name")
    parser.add_argument("--trigger-token", default="hsart")
    parser.add_argument("--output-dir", type=Path, default=Path("models/sd15-hearthstone-lora"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--target-modules",
        default="to_q,to_k,to_v,to_out.0",
        help="Comma-separated UNet attention modules to train.",
    )
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--best-checkpoint-name", default="best", help="Subdirectory for the best train-loss adapter.")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_every <= 0:
        raise SystemExit("--log-every must be greater than 0.")
    train(args)


def train(args: argparse.Namespace) -> None:
    deps = _require_training_deps()
    torch = deps.torch

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = deps.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    if args.seed is not None:
        deps.set_seed(args.seed)
        random.seed(args.seed)

    rows, missing_images = normalize_lora_rows(
        metadata_path=args.metadata,
        image_root=args.image_root,
        caption_column=args.caption_column,
        image_column=args.image_column,
        trigger_token=args.trigger_token,
        limit=args.max_train_samples,
    )
    if not rows:
        raise SystemExit(
            f"No trainable rows found. Check --metadata {args.metadata} and --image-root {args.image_root}."
        )
    if accelerator.is_main_process:
        print(f"Train rows={len(rows)} missing_images={len(missing_images)}")
        if missing_images[:5]:
            print("First missing images: " + ", ".join(missing_images[:5]))

    tokenizer = deps.CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
    )
    noise_scheduler = deps.DDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
        revision=args.revision,
    )
    text_encoder = deps.CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = deps.AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = deps.UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_config = deps.LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    unet.add_adapter(lora_config)

    weight_dtype = _weight_dtype(torch, accelerator.mixed_precision)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    if accelerator.mixed_precision == "fp16":
        deps.cast_training_params(unet, dtype=torch.float32)

    train_dataset = HearthstoneLoraDataset(
        rows=rows,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        torch=torch,
        numpy=deps.numpy,
        Image=deps.Image,
        ImageOps=deps.ImageOps,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=_build_collate_fn(tokenizer, torch),
    )

    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    lr_scheduler = deps.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Starting LoRA training: "
            f"epochs={num_train_epochs} max_steps={max_train_steps} "
            f"batch={args.train_batch_size} grad_accum={args.gradient_accumulation_steps}"
        )

    global_step = 0
    running_loss = 0.0
    best_loss = float("inf")
    best_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = _training_target(noise_scheduler, latents, noise, timesteps)
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().float().item()
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(args.log_every, 1)
                    running_loss = 0.0
                    avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
                    avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
                    lr = lr_scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        print(f"step={global_step} epoch={epoch + 1} loss={avg_loss:.6f} lr={lr:.6g}")
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_step = global_step
                        best_dir = args.output_dir / args.best_checkpoint_name
                        _save_lora_weights(deps, accelerator, unet, best_dir)
                        if accelerator.is_main_process:
                            _write_training_state(
                                best_dir,
                                {
                                    "best_step": best_step,
                                    "best_loss": best_loss,
                                    "selection_metric": "train_loss",
                                    "selection_note": "Lowest logged training loss window.",
                                },
                            )
                            print(f"Saved best checkpoint: {best_dir} loss={best_loss:.6f}")

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    _save_lora_weights(deps, accelerator, unet, checkpoint_dir)
                    if accelerator.is_main_process:
                        print(f"Saved checkpoint: {checkpoint_dir}")

                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    if best_step == 0:
        best_step = global_step
        best_loss = running_loss / max(global_step, 1)
        best_dir = args.output_dir / args.best_checkpoint_name
        _save_lora_weights(deps, accelerator, unet, best_dir)
        if accelerator.is_main_process:
            _write_training_state(
                best_dir,
                {
                    "best_step": best_step,
                    "best_loss": best_loss,
                    "selection_metric": "train_loss",
                    "selection_note": "Final training loss; run ended before the first log window.",
                },
            )
            print(f"Saved best checkpoint: {best_dir} loss={best_loss:.6f}")

    _save_lora_weights(deps, accelerator, unet, args.output_dir)
    accelerator.end_training()
    if accelerator.is_main_process:
        print(f"Saved final LoRA adapter: {args.output_dir}")
        if best_step:
            print(f"Best checkpoint: {args.output_dir / args.best_checkpoint_name} step={best_step} loss={best_loss:.6f}")


class HearthstoneLoraDataset:
    def __init__(
        self,
        *,
        rows: list[dict[str, str]],
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        torch,
        numpy,
        Image,
        ImageOps,
    ) -> None:
        self.rows = rows
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.torch = torch
        self.numpy = numpy
        self.Image = Image
        self.ImageOps = ImageOps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = self.Image.open(row["image_path"])
        image = _flatten_to_rgb(image, self.Image)
        if self.center_crop:
            image = self.ImageOps.fit(
                image,
                (self.resolution, self.resolution),
                method=self.Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize((self.resolution, self.resolution), self.Image.Resampling.BICUBIC)
        if self.random_flip and random.random() < 0.5:
            image = self.ImageOps.mirror(image)

        array = self.numpy.asarray(image).astype("float32") / 127.5 - 1.0
        pixel_values = self.torch.from_numpy(array).permute(2, 0, 1)
        return {"pixel_values": pixel_values, "caption": row["caption"]}


def _build_collate_fn(tokenizer, torch):
    def collate(examples: list[dict[str, object]]) -> dict[str, object]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples]).contiguous()
        captions = [str(example["caption"]) for example in examples]
        input_ids = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    return collate


def _flatten_to_rgb(image, Image):
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


def _training_target(noise_scheduler, latents, noise, timesteps):
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction_type={prediction_type}")


def _weight_dtype(torch, mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_lora_weights(deps: SimpleNamespace, accelerator, unet, output_dir: Path) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state_dict = deps.convert_state_dict_to_diffusers(deps.get_peft_model_state_dict(unwrapped_unet))
    deps.StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


def _write_training_state(output_dir: Path, state: dict[str, object]) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _require_training_deps() -> SimpleNamespace:
    try:
        import numpy
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import cast_training_params
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import LoraConfig, get_peft_model_state_dict
        from PIL import Image, ImageOps
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing diffusion training dependencies. Install them with `uv sync` after updating "
            "pyproject.toml, or install torch, diffusers, transformers, accelerate, peft, safetensors, "
            "numpy, and pillow."
        ) from exc

    return SimpleNamespace(
        Accelerator=Accelerator,
        AutoencoderKL=AutoencoderKL,
        CLIPTextModel=CLIPTextModel,
        CLIPTokenizer=CLIPTokenizer,
        DDPMScheduler=DDPMScheduler,
        Image=Image,
        ImageOps=ImageOps,
        LoraConfig=LoraConfig,
        StableDiffusionPipeline=StableDiffusionPipeline,
        UNet2DConditionModel=UNet2DConditionModel,
        cast_training_params=cast_training_params,
        convert_state_dict_to_diffusers=convert_state_dict_to_diffusers,
        get_peft_model_state_dict=get_peft_model_state_dict,
        get_scheduler=get_scheduler,
        numpy=numpy,
        set_seed=set_seed,
        torch=torch,
    )


if __name__ == "__main__":
    main()
