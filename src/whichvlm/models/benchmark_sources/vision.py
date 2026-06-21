"""Vision-language benchmark source.

Text-oriented score sources do not adequately cover multimodal models, so vision
profiles use this dedicated source. This is a curated snapshot rather than a live
scrape because a single stable, machine-readable VLM leaderboard is not always
available.

Values are a blended capability index (0-100) and only affect models tagged
as vision-capable in profile filtering.
"""

from __future__ import annotations

import httpx

# Curated multimodal capability index (0-100), 2026-05 snapshot.
# Anchored so frontier vision releases outrank older two-generation releases.
VISION_FALLBACK_2026_05: dict[str, float] = {
    # Qwen3-VL (current frontier)
    "Qwen/Qwen3-VL-235B-A22B-Instruct": 62.0,
    "Qwen/Qwen3-VL-32B-Instruct": 57.0,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 53.0,
    "Qwen/Qwen3-VL-8B-Instruct": 45.0,
    "Qwen/Qwen3-VL-8B-Thinking": 46.0,
    "Qwen/Qwen3-VL-4B-Instruct": 37.0,
    "Qwen/Qwen3-VL-4B-Thinking": 38.0,
    # Qwen2.5-VL (previous generation, still strong)
    "Qwen/Qwen2.5-VL-72B-Instruct": 55.0,
    "Qwen/Qwen2.5-VL-32B-Instruct": 49.0,
    "Qwen/Qwen2.5-VL-7B-Instruct": 41.0,
    "Qwen/Qwen2.5-VL-3B-Instruct": 33.0,
    # Qwen2-VL (two generations old — must rank below Qwen3-VL)
    "Qwen/Qwen2-VL-72B-Instruct": 45.0,
    "Qwen/Qwen2-VL-7B-Instruct": 33.0,
    "Qwen/Qwen2-VL-2B-Instruct": 24.0,
    # Meta Llama Vision
    "meta-llama/Llama-3.2-90B-Vision-Instruct": 41.0,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": 29.0,
    # Microsoft Phi vision
    "microsoft/Phi-4-reasoning-vision-15B": 46.0,
    "microsoft/Phi-3.5-vision-instruct": 27.0,
    # Google Gemma 3 (natively multimodal)
    "google/gemma-3-27b-it": 42.0,
    "google/gemma-3-12b-it": 35.0,
    "google/gemma-3-4b-it": 27.0,
    # Mistral Pixtral
    "mistralai/Pixtral-12B-2409": 35.0,
    "mistral-community/pixtral-12b": 35.0,
    # OpenGVLab InternVL3 (frontier open VLM)
    "OpenGVLab/InternVL3-78B": 56.0,
    "OpenGVLab/InternVL3-38B": 52.0,
    "OpenGVLab/InternVL3-14B": 45.0,
    "OpenGVLab/InternVL3-8B": 40.0,
    "OpenGVLab/InternVL2_5-78B": 50.0,
    # DeepSeek-VL2
    "deepseek-ai/deepseek-vl2": 38.0,
    # zhipu / GLM vision
    "zai-org/GLM-4.5V": 50.0,
}


async def fetch_vision_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Return curated VLM capability scores.

    A snapshot is returned for determinism while preserving a uniform call shape
    with other sources.
    """
    return dict(VISION_FALLBACK_2026_05)
