from __future__ import annotations

import io

import httpx

from whichvlm.models.http import get_with_retries

LEADERBOARD_PARQUET_URL = (
    "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    "/parquet/default/train/0.parquet"
)
LEADERBOARD_ROWS_URL = "https://datasets-server.huggingface.co/rows"
LEADERBOARD_DATASET = "open-llm-leaderboard/contents"

# --- Leaderboard normalization ---
# Legacy archive averages range ~5 to ~52. Capping at this value prevents older
# high-score historical entries from dominating newer releases now covered by
# recency-updated sources.
_LB_AVG_MAX = 52
_ARCHIVE_SOURCE_MAX_NORMALIZED = 78.0


async def _fetch_leaderboard_parquet(client: httpx.AsyncClient) -> dict[str, float]:
    """Download legacy parquet snapshot (requires pyarrow)."""
    import pyarrow.parquet as pq

    resp = await get_with_retries(
        client, LEADERBOARD_PARQUET_URL, follow_redirects=True
    )
    resp.raise_for_status()
    table = pq.read_table(
        io.BytesIO(resp.content),
        columns=["fullname", "Average ⬆️"],
    )
    d = table.to_pydict()
    scores: dict[str, float] = {}
    for i in range(len(d["fullname"])):
        name = d["fullname"][i]
        avg = d["Average ⬆️"][i]
        if name and avg and avg > 0:
            scores[name] = _normalize_leaderboard_avg(avg)
    return scores


async def _fetch_leaderboard_api(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch legacy rows via API (no pyarrow needed)."""
    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = await get_with_retries(
            client,
            LEADERBOARD_ROWS_URL,
            params={
                "dataset": LEADERBOARD_DATASET,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "100",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            name = row.get("fullname")
            avg = row.get("Average ⬆️")
            if name and avg and avg > 0:
                scores[name] = _normalize_leaderboard_avg(avg)

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores


def _normalize_leaderboard_avg(avg: float) -> float:
    """Normalize legacy-archive average to 0-archive-source scale."""
    score = avg / _LB_AVG_MAX * _ARCHIVE_SOURCE_MAX_NORMALIZED
    return max(0.0, min(_ARCHIVE_SOURCE_MAX_NORMALIZED, round(score, 1)))


async def fetch_leaderboard_with_fallback(
    client: httpx.AsyncClient,
) -> dict[str, float]:
    """Prefer the parquet path (one request, full table) and fall back to the
    paginated rows API when pyarrow is unavailable."""
    try:
        return await _fetch_leaderboard_parquet(client)
    except ImportError:
        return await _fetch_leaderboard_api(client)
