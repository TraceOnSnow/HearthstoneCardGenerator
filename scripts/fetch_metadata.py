import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.kg.metadata import normalize_metadata
from scripts.fetch_cards import _api_get, get_or_refresh_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Hearthstone metadata ID-name maps.")
    parser.add_argument("--output", type=Path, default=Path("data/hearthstone_metadata.json"))
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional path for the unmodified Blizzard metadata payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = get_or_refresh_token()
    raw = _api_get("/hearthstone/metadata", token)
    normalized = normalize_metadata(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    maps = normalized["maps"]
    print(f"Saved metadata maps: {args.output}")
    for key in sorted(maps):
        print(f"{key}={len(maps[key])}")


if __name__ == "__main__":
    main()
