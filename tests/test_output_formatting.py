from engine.types import CompatibilityResult, SpeedConfidence, VramConfidence
from models.types import ModelInfo
from output.formatting import format_speed, format_vram


def speed_result(speed: float, confidence: SpeedConfidence) -> CompatibilityResult:
    return CompatibilityResult(
        model=ModelInfo(
            id="org/model",
            family_id="org/model",
            name="model",
            parameter_count=1_000_000_000,
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        estimated_tok_per_sec=speed,
        speed_confidence=confidence,
    )


def vram_result(confidence: VramConfidence) -> CompatibilityResult:
    return CompatibilityResult(
        model=ModelInfo(
            id="org/model",
            family_id="org/model",
            name="model",
            parameter_count=1_000_000_000,
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=5 * 1024**3,
        vram_available_bytes=8 * 1024**3,
        vram_confidence=confidence,
    )


def test_format_speed_colors_by_runtime_speed_not_confidence():
    assert format_speed(speed_result(2.5, "medium")) == "[red]2.5 tok/s ~[/red]"
    assert format_speed(speed_result(6.0, "low")) == "[yellow]6.0 tok/s ?[/yellow]"
    assert format_speed(speed_result(12.0, "medium")) == "[green]12.0 tok/s ~[/green]"
    assert (
        format_speed(speed_result(30.0, "low"))
        == "[bright_green]30.0 tok/s ?[/bright_green]"
    )


def test_format_vram_marks_estimate_confidence():
    assert format_vram(vram_result("high")) == "5.0 GB"
    assert format_vram(vram_result("medium")) == "5.0 GB ~"
    assert format_vram(vram_result("low")) == "5.0 GB ?"
