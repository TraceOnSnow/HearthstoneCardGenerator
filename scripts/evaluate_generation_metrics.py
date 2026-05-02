#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.metrics import (  # noqa: E402
    collect_style_reference_paths,
    read_jsonl,
    write_jsonl,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated images with automatic model-based metrics.")
    parser.add_argument("--plan", type=Path, default=Path("results/generation_eval/generation_plan.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/generation_eval/generation_metrics.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/generation_eval/generation_metrics_summary.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--style-metadata", type=Path, default=Path("data/hf_hearthstone_art_512/metadata.jsonl"))
    parser.add_argument("--style-reference-limit", type=int, default=512)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--style-model", choices=["clip", "dinov2"], default="clip")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--quality-positive", default="a high quality detailed fantasy illustration")
    parser.add_argument("--quality-negative", default="a low quality blurry distorted image with artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.plan)
    generated = [row for row in rows if Path(str(row.get("output_image", ""))).exists()]
    if not generated:
        raise SystemExit(f"No generated images found in plan: {args.plan}")

    evaluator = AutoGenerationEvaluator(
        clip_model_name=args.clip_model,
        style_model=args.style_model,
        dinov2_model_name=args.dinov2_model,
        quality_positive=args.quality_positive,
        quality_negative=args.quality_negative,
        device=args.device,
    )
    style_paths = collect_style_reference_paths(
        metadata_path=args.style_metadata,
        image_root=args.image_root,
        limit=args.style_reference_limit,
    )
    if not style_paths:
        raise SystemExit(
            "No Hearthstone style reference images found. Fetch the HF art dataset or pass --style-metadata/--image-root."
        )
    evaluator.fit_style_reference(style_paths, batch_size=args.batch_size)

    metric_rows = [evaluator.evaluate_row(row, batch_size=args.batch_size) for row in generated]
    write_jsonl(args.out, metric_rows)
    write_summary_csv(args.summary_out, metric_rows)
    print(f"rows={len(metric_rows)}")
    print(f"out={args.out}")
    print(f"summary={args.summary_out}")


class AutoGenerationEvaluator:
    def __init__(
        self,
        *,
        clip_model_name: str,
        style_model: str,
        dinov2_model_name: str,
        quality_positive: str,
        quality_negative: str,
        device: str | None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.style_model_name = style_model
        if style_model == "clip":
            self.style_model = self.clip_model
            self.style_processor = self.clip_processor
        else:
            self.style_model = AutoModel.from_pretrained(dinov2_model_name).to(self.device)
            self.style_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        self.style_centroid = None
        self.quality_text_features = self._encode_texts([quality_positive, quality_negative])

    def fit_style_reference(self, paths: list[Path], *, batch_size: int) -> None:
        features = self._encode_images(paths, model_kind=self.style_model_name, batch_size=batch_size)
        if features.numel() == 0:
            raise ValueError("No valid style reference images could be encoded.")
        centroid = features.mean(dim=0, keepdim=True)
        self.style_centroid = centroid / centroid.norm(dim=-1, keepdim=True)

    def evaluate_row(self, row: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
        output_path = Path(str(row["output_image"]))
        image_feature_clip = self._encode_images([output_path], model_kind="clip", batch_size=batch_size)
        text_feature = self._encode_text(str(row.get("prompt", "")))
        style_feature = self._encode_images([output_path], model_kind=self.style_model_name, batch_size=batch_size)

        result = {
            "prompt_id": row.get("prompt_id", ""),
            "method": row.get("method", ""),
            "prompt": row.get("prompt", ""),
            "output_image": output_path.as_posix(),
            "image_quality_score": self._quality_score(image_feature_clip),
            "clip_prompt_alignment": _cosine(text_feature, image_feature_clip),
            "style_model": self.style_model_name,
            "style_similarity": _cosine(self.style_centroid, style_feature),
            "reference_image": row.get("reference_path", ""),
            "reference_similarity": None,
        }
        reference_path = Path(str(row.get("reference_path", "")))
        if reference_path.is_file():
            ref_feature = self._encode_images([reference_path], model_kind=self.style_model_name, batch_size=batch_size)
            result["reference_similarity"] = _cosine(ref_feature, style_feature)
        return result

    def _encode_text(self, text: str):
        return self._encode_texts([text])

    def _encode_texts(self, texts: list[str]):
        with self.torch.no_grad():
            inputs = self.clip_processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            features = _as_tensor(self.clip_model.get_text_features(**inputs))
            return features / features.norm(dim=-1, keepdim=True)

    def _quality_score(self, image_feature) -> float | None:
        if image_feature is None or image_feature.numel() == 0:
            return None
        logits = image_feature @ self.quality_text_features.T
        probs = self.torch.softmax(logits, dim=-1)
        return round(float(probs[0, 0].detach().cpu()), 6)

    def _encode_images(self, paths: list[Path], *, model_kind: str, batch_size: int):
        from PIL import Image

        encoded = []
        model = self.clip_model if model_kind == "clip" else self.style_model
        processor = self.clip_processor if model_kind == "clip" else self.style_processor
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                images = []
                for path in paths[start : start + batch_size]:
                    if path.exists():
                        images.append(Image.open(path).convert("RGB"))
                if not images:
                    continue
                inputs = processor(images=images, return_tensors="pt").to(self.device)
                if model_kind == "clip":
                    features = _as_tensor(model.get_image_features(**inputs))
                else:
                    output = model(**inputs)
                    features = _as_tensor(output)
                features = features / features.norm(dim=-1, keepdim=True)
                encoded.append(features.detach())
                for image in images:
                    image.close()
        if not encoded:
            return self.torch.zeros((0, 0), device=self.device)
        return self.torch.cat(encoded, dim=0)


def _cosine(left, right) -> float | None:
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return None
    return round(float((left @ right.T).mean().detach().cpu()), 6)


def _as_tensor(value):
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
        return value.last_hidden_state[:, 0]
    return value


if __name__ == "__main__":
    main()
