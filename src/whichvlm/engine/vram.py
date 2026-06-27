from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from whichvlm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.workload import VisionWorkload
from whichvlm.models.types import GGUFVariant, ModelInfo

# Memory model. Adds weights, cache, activations, and vision overhead.

KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024
MOE_ATTENTION_PARAM_MULTIPLIER = 4.0
VramConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class VramComponents:
    weights: int
    kv_cache: int
    activations: int
    vision: int
    runtime_overhead: int


@dataclass(frozen=True)
class VramEstimate:
    required_bytes: int
    lower_bytes: int
    upper_bytes: int
    confidence: VramConfidence
    components: VramComponents
    notes: list[str]


@dataclass(frozen=True)
class RuntimeCalibration:
    architecture: str
    backend: str
    overhead_ratio: float


RUNTIME_CALIBRATIONS: tuple[RuntimeCalibration, ...] = (
    RuntimeCalibration("llama", "llama.cpp", 0.06),
    RuntimeCalibration("qwen2vl", "transformers", 0.12),
    RuntimeCalibration("qwen2_vl", "transformers", 0.12),
    RuntimeCalibration("qwen2", "llama.cpp", 0.07),
    RuntimeCalibration("qwen2", "transformers", 0.10),
    RuntimeCalibration("mixtral", "llama.cpp", 0.08),
    RuntimeCalibration("deepseek", "transformers", 0.12),
)

VRAM_RANGE_FACTORS: dict[VramConfidence, tuple[float, float]] = {
    "high": (0.92, 1.12),
    "medium": (0.82, 1.28),
    "low": (0.65, 1.60),
}


def dtype_bytes(dtype: str | None) -> float:
    if not dtype:
        return 2.0
    normalized = dtype.lower().replace("torch.", "")
    if normalized in {"float32", "fp32", "f32"}:
        return 4.0
    if normalized in {"float8", "fp8", "int8"}:
        return 1.0
    return 2.0


def resolved_head_dim(model: ModelInfo) -> int | None:
    if model.head_dim:
        return model.head_dim
    if model.hidden_size and model.attention_heads:
        return model.hidden_size // model.attention_heads
    return None


def model_format_key(model: ModelInfo, variant: GGUFVariant | None) -> str:
    if variant:
        return "gguf"
    return model.model_format or "unknown"


def backend_key(model: ModelInfo, variant: GGUFVariant | None) -> str:
    model_format = model_format_key(model, variant)
    if model_format == "gguf":
        return "llama.cpp"
    if model_format == "mlx":
        return "mlx"
    return "transformers"


def runtime_calibration(
    model: ModelInfo,
    variant: GGUFVariant | None,
    backend: str | None = None,
) -> RuntimeCalibration | None:
    architecture = model.architecture.lower()
    backend = backend or backend_key(model, variant)
    for calibration in RUNTIME_CALIBRATIONS:
        if architecture.startswith(calibration.architecture):
            if backend == calibration.backend:
                return calibration
    return None


def kv_cache_has_architecture(model: ModelInfo) -> bool:
    return bool(
        model.layer_count
        and (model.kv_heads or model.attention_heads)
        and resolved_head_dim(model)
    )


def estimate_kv_cache(
    model: ModelInfo,
    context_length: int,
    batch_size: int = 1,
) -> int:
    kv_heads = model.kv_heads or model.attention_heads
    head_dim = resolved_head_dim(model)
    if model.layer_count and kv_heads and head_dim:
        bytes_per_value = dtype_bytes(model.kv_cache_dtype or model.dtype)
        return int(
            model.layer_count
            * kv_heads
            * head_dim
            * 2
            * context_length
            * batch_size
            * bytes_per_value
        )

    if model.is_moe and model.parameter_count_active:
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def activation_has_architecture(model: ModelInfo) -> bool:
    return bool(model.layer_count and model.hidden_size)


def activation_bytes(
    model: ModelInfo,
    context_length: int,
    batch_size: int = 1,
) -> int:
    if model.layer_count and model.hidden_size:
        token_bytes = (
            context_length
            * batch_size
            * model.hidden_size
            * dtype_bytes(model.dtype)
        )
        layer_factor = max(4, min(12, model.layer_count // 4))
        expert_scratch = int((effective_params(model) / 1e9) * 64 * 1024**2)
        return int(token_bytes * layer_factor + expert_scratch)

    if model.is_moe and model.parameter_count_active:
        effective_p = model.parameter_count_active
    else:
        effective_p = model.parameter_count

    base = 400_000_000
    param_term = int(effective_p * 0.08)
    ctx_term = int((context_length / 4096) * 150_000_000)
    return base + param_term + ctx_term


def model_confidence(notes: list[str]) -> VramConfidence:
    if any("KV cache" in note or "vision parameters" in note for note in notes):
        return "low"
    if notes:
        return "medium"
    return "high"


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


def image_tokens(model: ModelInfo, workload: VisionWorkload) -> int:
    patch_size = model.patch_size or 14
    tokens_per_image = max(1, (workload.image_size // patch_size) ** 2)
    if model.image_token_strategy == "default":
        tokens_per_image = max(1, tokens_per_image - 1)
    return tokens_per_image * workload.image_count


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
        vision_params = int(
            min(max(model.parameter_count * 0.18, 300_000_000), 4_000_000_000)
        )

    image_scale = (wl.image_size / 448) ** 2
    vision_weights = int(vision_params * 2)
    projector_scratch = int(128 * 1024**2 + vision_params * 0.15)
    tokens = image_tokens(model, wl)
    image_token_scratch = int(tokens * max(effective_p / 1e9, 1.0) * 96 * 1024)
    if model.vision_layer_count and model.vision_hidden_size:
        vision_hidden = (
            tokens
            * model.vision_hidden_size
            * model.vision_layer_count
            * dtype_bytes(model.dtype)
        )
        projector_hidden = model.projector_hidden_size or model.hidden_size or 0
        projector_scratch += int(tokens * projector_hidden * dtype_bytes(model.dtype))
        prefill = int(vision_hidden + effective_p * 0.004 * image_scale)
    else:
        prefill = int(
            (192 * 1024**2 + effective_p * 0.008) * image_scale * wl.image_count
        )
    return vision_weights + projector_scratch + image_token_scratch + prefill


def vram_notes(
    model: ModelInfo,
    variant: GGUFVariant | None,
    vision_workload: VisionWorkload | None,
    backend: str | None = None,
) -> list[str]:
    notes = []
    if not kv_cache_has_architecture(model):
        notes.append("KV cache uses parameter-count fallback")
    if not activation_has_architecture(model):
        notes.append("activations use parameter-count fallback")
    if is_vlm(model) and vision_workload is not None:
        if component_params(model, {"vision_encoder", "projector"}) <= 0:
            notes.append("vision parameters use VLM size fallback")
        if not (model.vision_layer_count and model.vision_hidden_size):
            notes.append("vision activations use image-size fallback")
    if runtime_calibration(model, variant, backend) is None:
        notes.append("runtime overhead uses default calibration")
    return notes


def runtime_overhead_bytes(
    model: ModelInfo,
    variant: GGUFVariant | None,
    subtotal: int,
    backend: str | None = None,
) -> int:
    calibration = runtime_calibration(model, variant, backend)
    ratio = calibration.overhead_ratio if calibration else 0.08
    return FRAMEWORK_OVERHEAD_BYTES + int(subtotal * ratio)


def estimate_vram_details(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
    batch_size: int = 1,
    backend: str | None = None,
) -> VramEstimate:
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, context_length, batch_size)
    activation = activation_bytes(model, context_length, batch_size)
    vision = estimate_vision_overhead(model, vision_workload)
    subtotal = weights + kv_cache + activation + vision
    runtime = runtime_overhead_bytes(model, variant, subtotal, backend)
    required = subtotal + runtime
    notes = vram_notes(model, variant, vision_workload, backend)
    confidence = model_confidence(notes)
    low_factor, high_factor = VRAM_RANGE_FACTORS[confidence]
    return VramEstimate(
        required_bytes=required,
        lower_bytes=int(required * low_factor),
        upper_bytes=int(required * high_factor),
        confidence=confidence,
        components=VramComponents(
            weights=weights,
            kv_cache=kv_cache,
            activations=activation,
            vision=vision,
            runtime_overhead=runtime,
        ),
        notes=notes,
    )


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
    batch_size: int = 1,
    backend: str | None = None,
) -> int:
    return estimate_vram_details(
        model,
        variant,
        context_length,
        vision_workload,
        batch_size,
        backend,
    ).required_bytes
