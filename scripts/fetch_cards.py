import argparse
import base64
import json
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# ---------------------------
# Global config (keep args simple)
# ---------------------------
REGION = "us"
LOCALE = "en_US"
API_NAMESPACE = "dynamic-us"

TOKEN_ENV_KEY = "BLIZZARD_ACCESS_TOKEN"
CLIENT_ID_ENV_KEY = "BLIZZARD_CLIENT_ID"
CLIENT_SECRET_ENV_KEY = "BLIZZARD_CLIENT_SECRET"

OAUTH_TOKEN_URL = "https://oauth.battle.net/token"
TOKEN_CHECK_URL = f"https://{REGION}.battle.net/oauth/check_token"
API_BASE_URL = f"https://{REGION}.api.blizzard.com"

OUTPUT_PATH = Path("data/cards_collectible.jsonl")
PAGE_SIZE = 500

load_dotenv()

def _http_request(
	url: str,
	*,
	method: str = "GET",
	headers: dict[str, str] | None = None,
	data: bytes | None = None,
	timeout: int = 20,
) -> dict[str, Any]:
	req = Request(url=url, method=method, headers=headers or {}, data=data)
	with urlopen(req, timeout=timeout) as response:
		body = response.read().decode("utf-8")
		if not body:
			return {}
		return json.loads(body)


def _build_basic_auth_header(client_id: str, client_secret: str) -> str:
	raw = f"{client_id}:{client_secret}".encode("utf-8")
	encoded = base64.b64encode(raw).decode("utf-8")
	return f"Basic {encoded}"


def _check_token_available(token: str, client_id: str, client_secret: str) -> bool:
	form_data = urlencode({"token": token}).encode("utf-8")
	headers = {
		"Authorization": _build_basic_auth_header(client_id, client_secret),
		"Content-Type": "application/x-www-form-urlencoded",
	}
	try:
		result = _http_request(
			TOKEN_CHECK_URL,
			method="POST",
			headers=headers,
			data=form_data,
		)
	except (HTTPError, URLError, TimeoutError):
		return False

	# Blizzard check_token returns metadata when token is valid.
	if not result:
		return False

	if "exp" in result and isinstance(result.get("exp"), int):
		return result["exp"] > int(time.time())

	return bool(result.get("client_id") or result.get("scope"))


def _create_access_token(client_id: str, client_secret: str) -> str:
	form_data = urlencode({"grant_type": "client_credentials"}).encode("utf-8")
	headers = {
		"Authorization": _build_basic_auth_header(client_id, client_secret),
		"Content-Type": "application/x-www-form-urlencoded",
	}
	response = _http_request(
		OAUTH_TOKEN_URL,
		method="POST",
		headers=headers,
		data=form_data,
	)
	token = response.get("access_token", "")
	if not token:
		raise RuntimeError(f"Token create failed, response={response}")
	return token


def get_or_refresh_token() -> str:
	client_id = os.getenv(CLIENT_ID_ENV_KEY, "").strip()
	client_secret = os.getenv(CLIENT_SECRET_ENV_KEY, "").strip()

	if not client_id or not client_secret:
		raise RuntimeError(
			f"Missing env: {CLIENT_ID_ENV_KEY}/{CLIENT_SECRET_ENV_KEY}."
		)

	token = os.getenv(TOKEN_ENV_KEY, "").strip()
	if token and _check_token_available(token, client_id, client_secret):
		return token

	new_token = _create_access_token(client_id, client_secret)
	os.environ[TOKEN_ENV_KEY] = new_token
	return new_token


def _api_get(path: str, token: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
	params = {
		"locale": LOCALE,
	}
	if query:
		params.update(query)

	url = f"{API_BASE_URL}{path}?{urlencode(params, doseq=True)}"
	headers = {
		"Authorization": f"Bearer {token}",
		"Battlenet-Namespace": API_NAMESPACE,
	}
	return _http_request(url, headers=headers)


def fetch_one_card_for_test(token: str) -> list[dict[str, Any]]:
	data = _api_get(
		"/hearthstone/cards",
		token,
		query={"page": 1, "pageSize": 1, "collectible": 1},
	)
	cards = data.get("cards", [])
	return cards[:1]


def fetch_all_cards(token: str) -> list[dict[str, Any]]:
	all_cards: list[dict[str, Any]] = []
	page = 1

	while True:
		payload = _api_get(
			"/hearthstone/cards",
			token,
			query={"page": page, "pageSize": PAGE_SIZE, "collectible": "1"},
		)
		cards = payload.get("cards", [])
		if not cards:
			break

		all_cards.extend(cards)

		page_count = payload.get("pageCount")
		if isinstance(page_count, int) and page >= page_count:
			break
		page += 1

	return all_cards


def write_jsonl(cards: list[dict[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		for card in cards:
			f.write(json.dumps(card, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fetch Hearthstone cards to JSONL.")
	parser.add_argument(
		"--full",
		action="store_true",
		help="Fetch all cards. Default mode fetches exactly one card for testing.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	token = get_or_refresh_token()

	if args.full:
		cards = fetch_all_cards(token)
		mode = "full"
	else:
		cards = fetch_one_card_for_test(token)
		mode = "test(1-card)"

	write_jsonl(cards, OUTPUT_PATH)
	print(f"Mode={mode}, cards={len(cards)}, output={OUTPUT_PATH}")


if __name__ == "__main__":
	main()
