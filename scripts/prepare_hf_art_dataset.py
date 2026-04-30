import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_dataset_card(output_dir: Path, dataset_name: str, image_count: int, failed_count: int) -> None:
    readme = f"""---
pretty_name: Hearthstone Art 512
size_categories:
- 1K<n<10K
task_categories:
- text-to-image
- image-to-image
---

# Hearthstone Art 512

Private dataset for project collaborators.

This dataset contains HearthstoneJSON 512x512 art-only card images plus metadata.
The artwork is owned by Blizzard Entertainment. Keep this dataset private unless
you have the rights to redistribute the images.

## Contents

- `images/`: downloaded 512x512 JPG art images.
- `metadata.jsonl`: one row per image, with `file_name`, card metadata, source URL, and file size.
- `splits/train.txt`: relative paths for the training split.
- `download_manifest.jsonl`: original downloader manifest, including failed rows.

## Summary

- Dataset repo: `{dataset_name}`
- Images included: {image_count}
- Failed/missing source images excluded from `metadata.jsonl`: {failed_count}

## Metadata Schema

Each `metadata.jsonl` row includes:

- `file_name`
- `card_id`
- `dbf_id`
- `name`
- `set`
- `type`
- `card_class`
- `artist`
- `source_url`
- `size_bytes`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset folder for HS art.")
    parser.add_argument("--art-dir", type=Path, default=Path("data/hs_art_512"))
    parser.add_argument("--manifest", type=Path, default=Path("data/hs_art_512/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hf_hearthstone_art_512"))
    parser.add_argument("--dataset-name", default="TraceOnSnow/hearthstone-art-512")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hardlinking when possible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    ok_rows = [row for row in rows if row.get("status") in {"ok", "exists"}]
    failed_rows = [row for row in rows if row.get("status") not in {"ok", "exists"}]

    metadata_rows: list[dict[str, Any]] = []
    train_paths: list[str] = []
    images_dir = args.output_dir / "images"

    for row in ok_rows:
        source_path = Path(str(row["path"]))
        if not source_path.exists():
            failed_rows.append({**row, "status": "failed: missing local file"})
            continue

        file_name = f"images/{source_path.name}"
        target_path = args.output_dir / file_name
        link_or_copy(source_path, target_path, copy=args.copy)

        metadata_rows.append(
            {
                "file_name": file_name,
                "card_id": row.get("id"),
                "dbf_id": row.get("dbfId"),
                "name": row.get("name"),
                "set": row.get("set"),
                "type": row.get("type"),
                "card_class": row.get("cardClass"),
                "artist": row.get("artist"),
                "source_url": row.get("url"),
                "size_bytes": row.get("size_bytes"),
            }
        )
        train_paths.append(file_name)

    metadata_rows.sort(key=lambda row: str(row.get("card_id") or ""))
    train_paths.sort()

    write_jsonl(args.output_dir / "metadata.jsonl", metadata_rows)
    write_jsonl(args.output_dir / "download_manifest.jsonl", rows)
    (args.output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits" / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    build_dataset_card(args.output_dir, args.dataset_name, len(metadata_rows), len(failed_rows))

    print(f"Prepared {args.output_dir}")
    print(f"images={len(list(images_dir.glob('*.jpg')))}")
    print(f"metadata_rows={len(metadata_rows)}")
    print(f"failed_or_missing={len(failed_rows)}")


if __name__ == "__main__":
    main()
