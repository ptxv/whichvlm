from __future__ import annotations

from dataclasses import dataclass

from whichvlm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.workload import VisionWorkload
from whichvlm.models.types import GGUFVariant, ModelInfo

# Memory model. Adds weights, cache, activations, and vision overhead.

KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024


MOE_ATTENTION_PARAM_MULTIPLIER = 4.0
DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
    "int8": 1,
}


@dataclass
class VramEstimate:
    total_bytes: int
    lower_bytes: int
    upper_bytes: int
    confidence: str
    weights_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    vision_bytes: int
    runtime_overhead_bytes: int


def bytes_per_value(dtype: str | None) -> int:
    return DTYPE_BYTES.get((dtype or "float16").lower(), 2)


def model_head_dim(model: ModelInfo) -> int | None:
    if model.head_dim:
        return model.head_dim
    if model.hidden_size and model.num_attention_heads:
        return model.hidden_size // model.num_attention_heads
    return None


def has_transformer_dims(model: ModelInfo) -> bool:
    head_dim = model_head_dim(model)
    return bool(
        model.num_layers
        and (model.num_key_value_heads or model.num_attention_heads)
        and head_dim
    )


def estimate_kv_cache(
    model: ModelInfo,
    context_length: int,
    *,
    batch_size: int = 1,
    cache_dtype: str | None = None,
) -> int:
    head_dim = model_head_dim(model)
    kv_heads = model.num_key_value_heads or model.num_attention_heads
    if model.num_layers and kv_heads and head_dim:
        kv_bytes = (
            model.num_layers
            * kv_heads
            * head_dim
            * context_length
            * batch_size
            * 2
            * bytes_per_value(cache_dtype or model.dtype)
        )
        return int(kv_bytes)

    # KV model. Scales cache by active size and requested context.
    if model.is_moe and model.parameter_count_active:
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def activation_bytes(model: ModelInfo, context_length: int) -> int:
    if model.num_layers and model.hidden_size:
        effective_p = effective_params(model)
        token_states = context_length * model.hidden_size * bytes_per_value(model.dtype)
        layer_scratch = int(token_states * min(model.num_layers, 48) * 0.5)
        param_scratch = int(effective_p * 0.02)
        return 256 * 1024**2 + layer_scratch + param_scratch

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


def estimate_vision_overhead(model: ModelInfo, workload: VisionWorkload | None) -> int:
    # Vision add-on. Prices encoder, projector, and image prefill cost.
    if workload is None:
        return 0
    wl = workload.normalized()
    if wl.image_count == 0:
        return 0
    if not is_vlm(model):
        return 0

    effective_p = effective_params(model)
    vision_params = component_params(model, {"vision_encoder", "projector"})
    if vision_params <= 0:
        if model.vision_num_layers and model.vision_hidden_size:
            vision_params = int(
                12 * model.vision_num_layers * model.vision_hidden_size**2
            )
        else:
            vision_params = int(
                min(max(model.parameter_count * 0.18, 300_000_000), 4_000_000_000)
            )

    image_scale = (wl.image_size / 448) ** 2
    vision_weights = int(vision_params * 2)
    projector_scratch = int(128 * 1024**2 + vision_params * 0.15)
    if model.projector_hidden_size and model.vision_hidden_size:
        projector_scratch += int(
            model.projector_hidden_size * model.vision_hidden_size * 2
        )
    patch_size = model.vision_patch_size or 14
    image_tokens = max(1, (wl.image_size // patch_size) ** 2) * wl.image_count
    if model.image_token_strategy and "cls" in model.image_token_strategy.lower():
        image_tokens += wl.image_count
    if model.vision_hidden_size and model.vision_num_layers:
        image_token_scratch = int(
            image_tokens
            * model.vision_hidden_size
            * model.vision_num_layers
            * bytes_per_value(model.dtype)
            * 0.5
        )
    else:
        image_token_scratch = int(
            image_tokens * max(effective_p / 1e9, 1.0) * 96 * 1024
        )
    prefill = int((192 * 1024**2 + effective_p * 0.008) * image_scale * wl.image_count)
    return vision_weights + projector_scratch + image_token_scratch + prefill


def estimate_vram_breakdown(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
) -> VramEstimate:
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, context_length)
    activation = activation_bytes(model, context_length)
    vision = estimate_vision_overhead(model, vision_workload)
    total = weights + kv_cache + activation + vision + FRAMEWORK_OVERHEAD_BYTES
    confidence = "high" if has_transformer_dims(model) else "medium"
    if vision_workload and is_vlm(model) and not model.vision_hidden_size:
        confidence = "low"
    spread = 0.08 if confidence == "high" else 0.18 if confidence == "medium" else 0.3
    return VramEstimate(
        total_bytes=total,
        lower_bytes=int(total * (1 - spread)),
        upper_bytes=int(total * (1 + spread)),
        confidence=confidence,
        weights_bytes=weights,
        kv_cache_bytes=kv_cache,
        activation_bytes=activation,
        vision_bytes=vision,
        runtime_overhead_bytes=FRAMEWORK_OVERHEAD_BYTES,
    )


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
) -> int:
    # Main memory pass. Returns total runtime bytes for one candidate.
    return estimate_vram_breakdown(
        model,
        variant,
        context_length,
        vision_workload,
    ).total_bytes
