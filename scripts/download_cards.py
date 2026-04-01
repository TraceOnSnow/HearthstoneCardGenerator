import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

SUPPORTED_TYPES = {4, 5}  # Minion, Spell


def download_one(card_id: int, url: str, output_path: Path, timeout: int) -> str:
    tmp_path = output_path.with_suffix(".png.tmp")
    try:
        with urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        tmp_path.write_bytes(data)
        tmp_path.rename(output_path)
        return "ok"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        return f"failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hearthstone card images.")
    parser.add_argument("--jsonl", type=Path, default=Path("data/cards_collectible.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/card_images"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--all-types", action="store_true", help="Download all card types, not just Minion/Spell.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[int, str, Path]] = []
    skipped_type = 0
    skipped_exists = 0

    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            card_id = card.get("id")
            card_type = card.get("cardTypeId")
            url = card.get("image", "").strip()

            if not isinstance(card_id, int) or not url:
                continue
            
            # for now, support minion, spell only.
            if not args.all_types and card_type not in SUPPORTED_TYPES:
                skipped_type += 1
                continue

            output_path = args.output_dir / f"{card_id}.png"
            if output_path.exists() and not args.force:
                skipped_exists += 1
                continue

            tasks.append((card_id, url, output_path))

    total = len(tasks)
    print(f"To download: {total}, skipped (exists): {skipped_exists}, skipped (type): {skipped_type}")

    if not tasks:
        print("Nothing to download.")
        return

    downloaded = 0
    failed_ids: list[int] = []
    lock = threading.Lock()

    def worker(card_id: int, url: str, output_path: Path) -> tuple[int, str]:
        result = download_one(card_id, url, output_path, args.timeout)
        nonlocal downloaded
        with lock:
            if result == "ok":
                downloaded += 1
                print(f"[{downloaded + len(failed_ids)}/{total}] {card_id}.png")
            else:
                failed_ids.append(card_id)
                print(f"[{downloaded + len(failed_ids)}/{total}] {card_id}.png - {result}")
        return card_id, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, cid, url, path) for cid, url, path in tasks]
        for f in as_completed(futures):
            f.result()

    print(f"\nDone. Downloaded: {downloaded}, Failed: {len(failed_ids)}")
    if failed_ids:
        print(f"Failed IDs: {failed_ids[:20]}{'...' if len(failed_ids) > 20 else ''}")


if __name__ == "__main__":
    main()
