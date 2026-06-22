from __future__ import annotations

import json
import time
from typing import Any

from whichvlm.utils import cache_dir

CACHE_DIR = cache_dir()
CACHE_FILE = CACHE_DIR / "models.json"
DEFAULT_TTL_SECONDS = 6 * 3600
CACHE_SCHEMA_VERSION = 2


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(*, allow_stale: bool = False) -> list[dict] | None:
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        cached_at = data["cached_at"]
        if not allow_stale and time.time() - cached_at > DEFAULT_TTL_SECONDS:
            return None
        models = data["models"]
        return models if isinstance(models, list) else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def cache_snapshot() -> dict[str, Any] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        cached_at = float(data["cached_at"])
        ttl = int(data.get("ttl_seconds", DEFAULT_TTL_SECONDS))
        age = max(0.0, time.time() - cached_at)
        return {
            "schema_version": int(data.get("schema_version", 1)),
            "cached_at": cached_at,
            "ttl_seconds": ttl,
            "expires_at": cached_at + ttl,
            "age_seconds": round(age, 1),
            "stale": age > ttl,
            "model_count": len(data.get("models", [])),
            "source": data.get("source", {"name": "huggingface"}),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_cache(models: list[dict], *, source: dict[str, Any] | None = None) -> None:
    ensure_cache_dir()
    data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at": time.time(),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "source": source or {"name": "huggingface"},
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
