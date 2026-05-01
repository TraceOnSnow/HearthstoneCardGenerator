#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diffusion.lora_data import build_caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and validate the Hearthstone art Hugging Face dataset.")
    parser.add_argument("--repo-id", default="comp646/hearthstone-art-512")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing a Hugging Face token.")
    parser.add_argument("--token-file", type=Path, default=None, help="Optional local file containing an hf_ token.")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch only metadata and README files.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--trigger-token", default="hsart")
    parser.add_argument("--no-ensure-text", action="store_true", help="Do not add/normalize metadata.jsonl text prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = snapshot_dataset(args)
    summary = ensure_lora_ready_metadata(
        Path(local_dir) / "metadata.jsonl",
        dataset_dir=Path(local_dir),
        trigger_token=args.trigger_token,
        ensure_text=not args.no_ensure_text,
    )
    print(f"Fetched dataset: {args.repo_id} -> {local_dir}")
    print(
        "metadata_rows={rows} images_present={images_present} missing_images={missing_images} "
        "text_rows={text_rows} text_rows_written={text_rows_written}".format(**summary)
    )
    if summary["missing_image_examples"]:
        print("First missing images: " + ", ".join(summary["missing_image_examples"]))


def snapshot_dataset(args: argparse.Namespace) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub. Install it with `pip install huggingface-hub`.") from exc

    allow_patterns = ["README.md", "metadata.jsonl", "download_manifest.jsonl"]
    if not args.metadata_only:
        allow_patterns.append("images/*")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output_dir),
        token=resolve_token(args.token_env, args.token_file),
        allow_patterns=allow_patterns,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )


def ensure_lora_ready_metadata(
    metadata_path: Path,
    *,
    dataset_dir: Path,
    trigger_token: str,
    ensure_text: bool,
) -> dict[str, Any]:
    rows = read_jsonl(metadata_path)
    missing_images: list[str] = []
    images_present = 0
    text_rows_written = 0

    for row in rows:
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            missing_images.append("<missing file_name>")
        elif (dataset_dir / file_name).exists():
            images_present += 1
        else:
            missing_images.append(file_name)

        if ensure_text:
            text = build_caption(row, caption_column="text", trigger_token=trigger_token)
            if text and row.get("text") != text:
                row["text"] = text
                text_rows_written += 1

    if ensure_text and text_rows_written:
        write_jsonl(metadata_path, rows)

    return {
        "rows": len(rows),
        "images_present": images_present,
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:5],
        "text_rows": sum(1 for row in rows if isinstance(row.get("text"), str) and row["text"].strip()),
        "text_rows_written": text_rows_written,
    }


def resolve_token(env_name: str, token_file: Path | None) -> str | None:
    token = os.environ.get(env_name) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    if token_file is None or not token_file.exists():
        return None
    match = re.search(r"hf_[A-Za-z0-9]+", token_file.read_text(encoding="utf-8", errors="ignore"))
    return match.group(0) if match else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
