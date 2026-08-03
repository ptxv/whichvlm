from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def cache_checksum(payload: dict) -> str:
    content = {key: value for key, value in payload.items() if key != "checksum"}
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_cache_payload(cache_file: Any) -> dict | None:
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "checksum" in payload and payload["checksum"] != cache_checksum(payload):
        return None
    return payload


def write_cache_payload(cache_file: Any, payload: dict) -> None:
    payload_with_checksum = {**payload, "checksum": cache_checksum(payload)}
    cache_file.write_text(
        json.dumps(payload_with_checksum, ensure_ascii=False), encoding="utf-8"
    )


def cache_expired(cached_at: float, ttl_seconds: int, *, allow_stale: bool) -> bool:
    return not allow_stale and time.time() - cached_at > ttl_seconds


def cache_snapshot_metadata(
    payload: dict,
    *,
    default_ttl_seconds: int,
    item_key: str,
    item_count_key: str,
    default_source: dict,
) -> dict | None:
    try:
        items = payload[item_key]
        if not isinstance(items, (dict, list)):
            return None
        cached_at = float(payload["cached_at"])
        ttl_seconds = int(payload.get("ttl_seconds", default_ttl_seconds))
        schema_version = int(payload.get("schema_version", 1))
    except (KeyError, TypeError, ValueError):
        return None

    age_seconds = max(0.0, time.time() - cached_at)
    return {
        "schema_version": schema_version,
        "cached_at": cached_at,
        "ttl_seconds": ttl_seconds,
        "expires_at": cached_at + ttl_seconds,
        "age_seconds": round(age_seconds, 1),
        "stale": age_seconds > ttl_seconds,
        item_count_key: len(items),
        "source": payload.get("source", default_source),
    }
