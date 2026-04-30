import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CARDS_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.collectible.json"
ART_URLS = {
    "512x": ("https://art.hearthstonejson.com/v1/512x/{card_id}.jpg", ".jpg"),
    "256x": ("https://art.hearthstonejson.com/v1/256x/{card_id}.jpg", ".jpg"),
    "orig": ("https://art.hearthstonejson.com/v1/orig/{card_id}.png", ".png"),
}


def fetch_json(url: str, timeout: int) -> Any:
    req = Request(url, headers={"User-Agent": "HearthstoneCardGenerator/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_one(url: str, output_path: Path, timeout: int, retries: int) -> tuple[str, int]:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "HearthstoneCardGenerator/0.1"})
    last_error = ""
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp_path.write_bytes(data)
            tmp_path.rename(output_path)
            return "ok", len(data)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            tmp_path.unlink(missing_ok=True)
            if attempt == retries:
                break
    return f"failed: {last_error}", 0


def build_manifest_row(card: dict[str, Any], url: str, path: Path, status: str, size_bytes: int) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "dbfId": card.get("dbfId"),
        "name": card.get("name"),
        "set": card.get("set"),
        "type": card.get("type"),
        "cardClass": card.get("cardClass"),
        "artist": card.get("artist"),
        "url": url,
        "path": str(path),
        "status": status,
        "size_bytes": size_bytes,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HearthstoneJSON art-only card images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/hs_art_512"))
    parser.add_argument("--manifest", type=Path, help="Defaults to <output-dir>/manifest.jsonl.")
    parser.add_argument("--variant", choices=sorted(ART_URLS), default="512x")
    parser.add_argument("--limit", type=int, help="Download the first N cards after filtering.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cards-url", default=CARDS_URL)
    parser.add_argument(
        "--type",
        action="append",
        dest="types",
        help="Optional HearthstoneJSON type filter, e.g. MINION. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.output_dir / "manifest.jsonl"
    template, ext = ART_URLS[args.variant]

    cards = fetch_json(args.cards_url, args.timeout)
    if args.types:
        allowed_types = {item.upper() for item in args.types}
        cards = [card for card in cards if str(card.get("type", "")).upper() in allowed_types]
    if args.limit is not None:
        cards = cards[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cards={len(cards)} variant={args.variant} output={args.output_dir}")

    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    done = 0

    def worker(card: dict[str, Any]) -> dict[str, Any]:
        card_id = card.get("id")
        if not card_id:
            return build_manifest_row(card, "", Path(""), "failed: missing id", 0)

        url = template.format(card_id=card_id)
        output_path = args.output_dir / f"{card_id}{ext}"
        if output_path.exists() and not args.force:
            status = "exists"
            size_bytes = output_path.stat().st_size
        else:
            status, size_bytes = download_one(url, output_path, args.timeout, args.retries)
        return build_manifest_row(card, url, output_path, status, size_bytes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, card) for card in cards]
        for future in as_completed(futures):
            row = future.result()
            with rows_lock:
                rows.append(row)
                done += 1
                print(f"[{done}/{len(cards)}] {row['id']} {row['status']}")

    rows.sort(key=lambda row: str(row.get("id") or ""))
    write_jsonl(manifest_path, rows)
    ok_count = sum(1 for row in rows if row["status"] in {"ok", "exists"})
    failed_count = len(rows) - ok_count
    print(f"Done. ok={ok_count} failed={failed_count} manifest={manifest_path}")


if __name__ == "__main__":
    main()
