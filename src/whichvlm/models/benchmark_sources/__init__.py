"""External benchmark sources for score material.

Each module fetches an independent leaderboard or index, normalizes it to the
shared 0-100 scale, and returns a ``dict[str, float]`` keyed by HuggingFace
model ID (or a list of synonyms).

The functions are intentionally defensive: if a source is unreachable or
returns malformed data, they log a warning and return an empty dict so the
main benchmark merge pipeline does not abort.
"""

from whichvlm.models.benchmark_sources.aa_index import (
    fetch_aa_index_scores,
    get_aa_curated_fallback,
)
from whichvlm.models.benchmark_sources.aider import fetch_aider_polyglot_scores
from whichvlm.models.benchmark_sources.chatbot_arena import fetch_arena_scores
from whichvlm.models.benchmark_sources.livebench import (
    get_livebench_data,
)
from whichvlm.models.benchmark_sources.open_llm_leaderboard import (
    fetch_leaderboard_with_fallback,
)
from whichvlm.models.benchmark_sources.vision import fetch_vision_scores

# Newest curated-fallback date across all sources. Live scrapes are merged when
# reachable, but they can fail, so the user-visible ranking is anchored to this
# snapshot. Surface it in the CLI so a stale recommendation is visible.
BENCHMARK_SNAPSHOT = "2026-05"

__all__ = [
    "BENCHMARK_SNAPSHOT",
    "fetch_aa_index_scores",
    "fetch_aider_polyglot_scores",
    "fetch_arena_scores",
    "fetch_leaderboard_with_fallback",
    "fetch_vision_scores",
    "get_aa_curated_fallback",
    "get_livebench_data",
]
