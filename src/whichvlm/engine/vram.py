from __future__ import annotations

from whichvlm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.workload import Workload
from whichvlm.models.types import GGUFVariant, ModelInfo

KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024


MOE_ATTENTION_PARAM_MULTIPLIER = 4.0


def estimate_kv_cache(model: ModelInfo, context_length: int) -> int:
    if model.is_moe and model.parameter_count_active:
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def activation_bytes(model: ModelInfo, context_length: int) -> int:

    if model.is_moe and model.parameter_count_active:
        effective_p = model.parameter_count_active
    else:
        effective_p = model.parameter_count

    base = 400_000_000
    param_term = int(effective_p * 0.08)
    ctx_term = int((context_length / 4096) * 150_000_000)
    return base + param_term + ctx_term


def component_params(model: ModelInfo, roles: set[str]) -> int:
    total = 0
    for component in model.components:
        if component.role in roles and component.parameter_count:
            total += component.parameter_count
    return total


def effective_params(model: ModelInfo) -> int:
    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active
    return model.parameter_count


def is_vlm(model: ModelInfo) -> bool:
    if model.hf_pipeline_tag in {
        "image-text-to-text",
        "visual-question-answering",
        "image-to-text",
    }:
        return True
    return any(
        component.role in {"vision_encoder", "projector", "processor"}
        for component in model.components
    )


def estimate_vision_overhead(model: ModelInfo, workload: Workload | None) -> int:
    if workload is None:
        return 0
    wl = workload.normalized()
    visual_inputs = wl.image_count + wl.video_frames
    if visual_inputs == 0 and wl.audio_seconds == 0:
        return 0
    if not is_vlm(model):
        return 0

    effective_p = effective_params(model)
    vision_params = component_params(model, {"vision_encoder", "projector"})
    if vision_params <= 0:
        vision_params = int(
            min(max(model.parameter_count * 0.18, 300_000_000), 4_000_000_000)
        )

    batch_size = wl.batch_size
    image_scale = (wl.image_size / 448) ** 2
    vision_weights = int(vision_params * 2)
    projector_scratch = int(128 * 1024**2 + vision_params * 0.15)
    image_tokens = max(1, (wl.image_size // 14) ** 2) * visual_inputs
    image_token_scratch = int(image_tokens * max(effective_p / 1e9, 1.0) * 96 * 1024)
    prefill = int((192 * 1024**2 + effective_p * 0.008) * image_scale * visual_inputs)
    audio = int((64 * 1024**2 + wl.audio_seconds * 2 * 1024**2) * batch_size)
    return (
        vision_weights
        + projector_scratch
        + (image_token_scratch + prefill) * batch_size
        + audio
    )


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: Workload | None = None,
) -> int:
    workload = vision_workload.normalized() if vision_workload else None
    effective_context = workload.context_length if workload else context_length
    batch_size = workload.batch_size if workload else 1
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, effective_context) * batch_size
    activation = activation_bytes(model, effective_context) * batch_size
    vision = estimate_vision_overhead(model, workload)
    framework = FRAMEWORK_OVERHEAD_BYTES
    return weights + kv_cache + activation + vision + framework
