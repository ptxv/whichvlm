from whichvlm.engine.vram import (
    estimate_kv_cache,
    estimate_vram,
    estimate_vram_breakdown,
)
from whichvlm.engine.workload import VisionWorkload
from whichvlm.models.types import GGUFVariant, ModelComponent, ModelInfo


def make_model(params: int, **kwargs) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=params,
        **kwargs,
    )


def test_estimate_vram_gguf_variant():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)

    assert vram > 4_000_000_000
    assert vram < 7_000_000_000


def test_estimate_vram_fp16_fallback():
    model = make_model(7_000_000_000)
    vram = estimate_vram(model, None, context_length=4096)

    assert vram > 14_000_000_000
    assert vram < 20_000_000_000


def test_estimate_vram_increases_with_context():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram_4k = estimate_vram(model, variant, context_length=4096)
    vram_32k = estimate_vram(model, variant, context_length=32768)
    assert vram_32k > vram_4k


def test_estimate_kv_cache_scales_with_params():
    small = make_model(1_000_000_000)
    large = make_model(70_000_000_000)
    kv_small = estimate_kv_cache(small, 4096)
    kv_large = estimate_kv_cache(large, 4096)
    assert kv_large > kv_small


def test_estimate_kv_cache_uses_transformer_dimensions():
    model = make_model(
        7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        dtype="bfloat16",
    )

    assert estimate_kv_cache(model, 4096) == 32 * 8 * 128 * 4096 * 2 * 2


def test_moe_kv_cache_uses_attention_dims_not_active_params():
    small_active = make_model(
        100_000_000_000,
        parameter_count_active=3_000_000_000,
        is_moe=True,
        num_layers=48,
        hidden_size=6144,
        num_attention_heads=48,
        num_key_value_heads=8,
        dtype="float16",
    )
    large_active = make_model(
        100_000_000_000,
        parameter_count_active=30_000_000_000,
        is_moe=True,
        num_layers=48,
        hidden_size=6144,
        num_attention_heads=48,
        num_key_value_heads=8,
        dtype="float16",
    )

    assert estimate_kv_cache(small_active, 4096) == estimate_kv_cache(
        large_active,
        4096,
    )


def test_estimate_vram_small_model():
    model = make_model(500_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=300_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)

    assert vram > 300_000_000
    assert vram < 3_000_000_000


def test_vision_workload_increases_vram_predictably():
    model = make_model(7_000_000_000, hf_pipeline_tag="image-text-to-text")
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    text_only = estimate_vram(model, variant, context_length=4096)
    one_image = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=1, image_size=448),
    )
    two_large_images = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=2, image_size=896),
    )

    assert one_image > text_only
    assert two_large_images > one_image


def test_vision_workload_does_not_change_text_model_vram():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    text_only = estimate_vram(model, variant, context_length=4096)
    with_image_workload = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=1, image_size=448),
    )

    assert with_image_workload == text_only


def test_vision_component_sizes_increase_vlm_overhead():
    small = make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        components=[
            ModelComponent(
                role="vision_encoder",
                repo_id="test/model",
                parameter_count=300_000_000,
            ),
            ModelComponent(
                role="projector",
                repo_id="test/model",
                parameter_count=50_000_000,
            ),
        ],
    )
    large = make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        components=[
            ModelComponent(
                role="vision_encoder",
                repo_id="test/model",
                parameter_count=1_000_000_000,
            ),
            ModelComponent(
                role="projector",
                repo_id="test/model",
                parameter_count=200_000_000,
            ),
        ],
    )
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(large, None, vision_workload=workload) > estimate_vram(
        small,
        None,
        vision_workload=workload,
    )


def test_vision_patch_size_changes_image_token_scratch():
    small_patches = make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        vision_num_layers=24,
        vision_hidden_size=1024,
        vision_patch_size=14,
    )
    large_patches = make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        vision_num_layers=24,
        vision_hidden_size=1024,
        vision_patch_size=28,
    )
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(small_patches, None, vision_workload=workload) > estimate_vram(
        large_patches,
        None,
        vision_workload=workload,
    )


def test_estimate_vram_breakdown_reports_range_and_confidence():
    model = make_model(
        7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )

    estimate = estimate_vram_breakdown(model, None)

    assert estimate.lower_bytes < estimate.total_bytes < estimate.upper_bytes
    assert estimate.confidence == "high"
    assert estimate.total_bytes == (
        estimate.weights_bytes
        + estimate.kv_cache_bytes
        + estimate.activation_bytes
        + estimate.vision_bytes
        + estimate.runtime_overhead_bytes
    )
