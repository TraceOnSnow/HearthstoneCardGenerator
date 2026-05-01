from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.retrieval.common import query_to_text, result_row


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def retrieve_clip(
    *,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    image_root: Path,
    cache_path: Path,
    top_k: int,
    model_name: str = DEFAULT_CLIP_MODEL,
    limit: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> list[dict[str, Any]]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if limit is not None:
        corpus = corpus[:limit]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    image_features, valid_corpus = _load_or_build_image_cache(
        corpus=corpus,
        image_root=image_root,
        cache_path=cache_path,
        model=model,
        processor=processor,
        device=device,
        batch_size=batch_size,
    )
    text_features = _encode_texts([query_to_text(query) for query in queries], model=model, processor=processor, device=device)

    rows: list[dict[str, Any]] = []
    scores = text_features @ image_features.T
    for query_idx, query in enumerate(queries):
        order = np.argsort(-scores[query_idx])[:top_k]
        for rank, corpus_idx in enumerate(order, start=1):
            card = valid_corpus[int(corpus_idx)]
            rows.append(
                result_row(
                    query=query,
                    method="clip_image_baseline",
                    rank=rank,
                    card=card,
                    score=float(scores[query_idx, corpus_idx]),
                    reasons=["clip_text_to_image_cosine"],
                )
            )
    return rows


def _load_or_build_image_cache(
    *,
    corpus: list[dict[str, Any]],
    image_root: Path,
    cache_path: Path,
    model,
    processor,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = [str(row.get("image", "")) for row in corpus]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["images"]) == signature:
            return cached["features"].astype("float32"), list(cached["corpus"])

    valid_rows: list[dict[str, Any]] = []
    feature_chunks: list[np.ndarray] = []
    image_batch: list[Image.Image] = []
    for row in corpus:
        path = image_root / str(row.get("image", ""))
        if not path.exists():
            continue
        valid_rows.append(row)
        with Image.open(path) as image:
            image_batch.append(image.convert("RGB"))
        if len(image_batch) >= batch_size:
            feature_chunks.append(_encode_images(image_batch, model=model, processor=processor, device=device))
            image_batch = []

    if image_batch:
        feature_chunks.append(_encode_images(image_batch, model=model, processor=processor, device=device))

    features = np.concatenate(feature_chunks, axis=0).astype("float32") if feature_chunks else np.zeros((0, 0), dtype="float32")
    np.savez_compressed(
        cache_path,
        images=np.array([row["image"] for row in valid_rows], dtype=object),
        features=features,
        corpus=np.array(valid_rows, dtype=object),
    )
    return features, valid_rows


def _encode_images(images: list[Image.Image], *, model, processor, device: str, batch_size: int = 32) -> np.ndarray:
    import torch

    chunks = []
    with torch.no_grad():
        for idx in range(0, len(images), batch_size):
            batch = images[idx : idx + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            features = _as_tensor(model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).astype("float32")


def _encode_texts(texts: list[str], *, model, processor, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
        features = _as_tensor(model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().float().numpy().astype("float32")


def _as_tensor(value):
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state[:, 0]
    return value
