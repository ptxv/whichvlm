from engine.types import CompatibilityResult, SpeedConfidence, VramConfidence
from models.types import ModelInfo
from output.formatting import format_speed, format_vram


def result_for_formatting(
    *,
    speed: float = 0.0,
    speed_confidence: SpeedConfidence = "high",
    vram_confidence: VramConfidence = "high",
) -> CompatibilityResult:
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
        estimated_tok_per_sec=speed,
        speed_confidence=speed_confidence,
        vram_confidence=vram_confidence,
    )


def test_format_speed_colors_by_runtime_speed_not_confidence():
    cases = (
        (2.5, "medium", "[red]2.5 tok/s ~[/red]"),
        (6.0, "low", "[yellow]6.0 tok/s ?[/yellow]"),
        (12.0, "medium", "[green]12.0 tok/s ~[/green]"),
        (30.0, "low", "[bright_green]30.0 tok/s ?[/bright_green]"),
    )
    for speed, confidence, expected in cases:
        result = result_for_formatting(speed=speed, speed_confidence=confidence)
        assert format_speed(result) == expected


def test_format_vram_marks_estimate_confidence():
    cases = (
        ("high", "5.0 GB"),
        ("medium", "5.0 GB ~"),
        ("low", "5.0 GB ?"),
    )
    for confidence, expected in cases:
        assert (
            format_vram(result_for_formatting(vram_confidence=confidence)) == expected
        )
