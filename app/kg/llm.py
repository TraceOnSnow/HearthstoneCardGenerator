from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from app.kg.io import append_jsonl


GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        loaded = False
        env_path = ".env"
        if not os.path.exists(env_path):
            return False
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#") or "=" not in text:
                    continue
                key, value = text.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded = True
        return loaded


def google_generate_content(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    url = f"{base}?{urllib.parse.urlencode({'key': api_key})}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Empty candidates from model: {body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Missing content parts from model: {body}")

    text = parts[0].get("text", "")
    if not text:
        raise RuntimeError(f"Empty model text output: {body}")

    return text


def minimax_generate_content(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    url = "https://api.minimaxi.com/v1/chat/completions"
    max_tokens = int(os.getenv("MINIMAX_MAX_TOKENS", "4096"))
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices from MiniMax: {body}")

    message = choices[0].get("message", {})
    text = message.get("content", "")
    if not text:
        raise RuntimeError(f"Empty MiniMax message content: {body}")

    return text


def _api_key_for_provider(provider: str) -> str:
    if provider == "google":
        env_key = GOOGLE_API_KEY_ENV
    elif provider == "minimax":
        env_key = MINIMAX_API_KEY_ENV
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env: {env_key}. Set it in .env or shell.")
    return api_key


def _generate_content(
    prompt: str,
    *,
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    if provider == "google":
        return google_generate_content(
            prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    if provider == "minimax":
        return minimax_generate_content(
            prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def run_llm_batches(
    prompt_rows: list[dict[str, Any]],
    *,
    output_path,
    existing_outputs: list[dict[str, Any]],
    dry_run: bool,
    provider: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
    resume: bool,
    force: bool,
    concurrency: int = 1,
    max_retries: int = 0,
    retry_backoff_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    load_dotenv()
    completed_status = "dry_run" if dry_run else "ok"
    completed_by_batch = {
        row.get("batch_id"): row
        for row in existing_outputs
        if row.get("status") == completed_status
    }
    outputs: list[dict[str, Any]] = list(existing_outputs) if resume and not force else []

    api_key = "" if dry_run else _api_key_for_provider(provider)
    pending_rows = [
        row
        for row in prompt_rows
        if not (resume and not force and row["batch_id"] in completed_by_batch)
    ]

    if concurrency <= 1 or dry_run:
        for row in pending_rows:
            output_row = _run_one_llm_batch(
                row,
                dry_run=dry_run,
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            outputs.append(output_row)
            append_jsonl(output_path, output_row)
            print(f"batch={row['batch_id']} cards={row['card_count']} status={output_row['status']}")
        return outputs

    output_path = Path(output_path)
    write_lock = Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_one_llm_batch,
                row,
                dry_run=dry_run,
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            for row in pending_rows
        ]
        for future in as_completed(futures):
            output_row = future.result()
            with write_lock:
                outputs.append(output_row)
                append_jsonl(output_path, output_row)
            print(f"batch={output_row['batch_id']} cards={output_row['card_count']} status={output_row['status']}")

    return outputs


def _run_one_llm_batch(
    row: dict[str, Any],
    *,
    dry_run: bool,
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    batch_id = row["batch_id"]
    attempts = max(1, max_retries + 1)
    last_error = ""
    result_text = ""
    status = "failed"
    started_at = time.time()

    for attempt in range(1, attempts + 1):
        started_at = time.time()
        if dry_run:
            result_text = json.dumps({"cards": []}, ensure_ascii=False)
            status = "dry_run"
            error = ""
            break
        else:
            try:
                result_text = _generate_content(
                    row["prompt"],
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
                status = "ok"
                error = ""
                break
            except urllib.error.HTTPError as exc:
                result_text = ""
                status = "failed"
                error_body = exc.read().decode("utf-8", errors="replace")
                error = f"HTTP {exc.code}: {error_body}"
            except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                result_text = ""
                status = "failed"
                error = str(exc)
        last_error = error
        if attempt < attempts:
            time.sleep(retry_backoff_seconds * attempt)

    return {
        "batch_id": batch_id,
        "card_count": row["card_count"],
        "card_ids": row["card_ids"],
        "status": status,
        "error": last_error if status == "failed" else "",
        "raw_response": result_text,
        "attempts": attempts if status == "failed" else attempt,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
