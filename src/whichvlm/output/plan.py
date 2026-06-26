from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from whichvlm.constants import (
    QUANT_BYTES_PER_WEIGHT,
    QUANT_QUALITY_PENALTY,
    BYTES_PER_GIB,
)
from whichvlm.engine.compatibility import check_compatibility
from whichvlm.engine.performance import estimate_tok_per_sec
from whichvlm.engine.vram import estimate_vram
from whichvlm.engine.workload import VisionWorkload
from whichvlm.hardware.catalog import (
    HARDWARE_CATALOG,
    PLAN_SYSTEM_RAM_BYTES,
    PLAN_VRAM_HEADROOM_RATIO,
    HardwareCatalogEntry,
)
from whichvlm.hardware.types import GPUInfo, HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.output import console
from whichvlm.output.formatting import format_bytes, format_params

PLAN_QUANTS = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16")


def plan_variant_for_quant(model: ModelInfo, quant: str) -> GGUFVariant:
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant.upper(), 0.5625)
    return GGUFVariant(
        filename="",
        quant_type=quant,
        file_size_bytes=int(model.parameter_count * bpw),
    )


def plan_vision_workload(
    context_length: int,
    image_count: int,
    image_size: int,
    video_frames: int,
) -> VisionWorkload | None:
    visual_inputs = image_count + video_frames
    if visual_inputs <= 0:
        return None
    return VisionWorkload(
        image_count=visual_inputs,
        image_size=image_size,
        context_length=context_length,
    )


def plan_vram_by_quant(
    model: ModelInfo,
    context_length: int,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
) -> dict[str, dict]:
    rows = {}
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    for quant in PLAN_QUANTS:
        if quant not in QUANT_BYTES_PER_WEIGHT:
            continue
        vram_bytes = estimate_vram(
            model, plan_variant_for_quant(model, quant), context_length, vision_workload
        )
        rows[quant] = {
            "vram_bytes": vram_bytes,
            "quality_loss": QUANT_QUALITY_PENALTY.get(quant, 0.0),
        }
    return rows


def plan_target_vram(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    vram_by_quant: dict[str, dict] | None = None,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
) -> int:
    vram_by_quant = vram_by_quant or plan_vram_by_quant(model, context_length)
    existing = vram_by_quant.get(target_quant.upper())
    if existing:
        return int(existing["vram_bytes"])
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    return estimate_vram(
        model,
        plan_variant_for_quant(model, target_quant),
        context_length,
        vision_workload,
    )


def plan_binding_constraint(row: dict, min_speed: float | None) -> str:
    if not row["context_fits"]:
        return "context length"
    if min_speed is not None and not row["meets_speed"]:
        return "speed"
    if not row["can_run"]:
        return "memory"
    if row["fit_type"] == "partial_offload":
        return "VRAM"
    return "none"


def gpu_backends(gpu: GPUInfo) -> list[str]:
    return [capability.name for capability in gpu.backend_capabilities if capability.available]


def plan_row_for_hardware(
    model: ModelInfo,
    target_quant: str,
    hardware: HardwareInfo,
    label: str,
    context_length: int,
    vision_workload: VisionWorkload | None,
    min_speed: float | None,
) -> dict:
    variant = plan_variant_for_quant(model, target_quant)
    result = check_compatibility(model, variant, hardware, context_length, vision_workload)
    fit_type = result.fit_type if result.can_run else "too_small"
    gpu = hardware.gpus[0]
    speed = None
    if result.can_run and gpu.memory_bandwidth_gbps:
        speed = round(estimate_tok_per_sec(model, variant, gpu, result.fit_type), 1)
    row = {
        "name": label,
        "vram_gb": round(sum(gpu.vram_bytes for gpu in hardware.gpus) / BYTES_PER_GIB),
        "usable_vram_bytes": (
            result.multi_gpu_effective_vram_bytes or result.vram_available_bytes
        ),
        "reserved_headroom_bytes": sum(
            gpu.vram_bytes - (gpu.usable_vram_bytes or gpu.vram_bytes)
            for gpu in hardware.gpus
        ),
        "system_ram_bytes": hardware.ram_bytes,
        "required_memory_bytes": result.vram_required_bytes,
        "fit_type": fit_type,
        "can_run": result.can_run,
        "context_fits": result.context_fits,
        "uses_multi_gpu": result.uses_multi_gpu,
        "multi_gpu_effective_vram_bytes": result.multi_gpu_effective_vram_bytes,
        "estimated_tok_per_sec": speed,
        "meets_speed": min_speed is None or (speed is not None and speed >= min_speed),
        "supported_backends": gpu_backends(gpu),
        "warnings": result.warnings,
    }
    row["binding_constraint"] = plan_binding_constraint(row, min_speed)
    return row


def plan_gpu_compatibility(
    model: ModelInfo,
    target_quant: str,
    target_vram: int,
    context_length: int = 4096,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
    min_speed: float | None = None,
) -> list[dict]:
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    rows = []
    for entry in HARDWARE_CATALOG:
        hardware = entry.to_hardware(system_ram_bytes)
        rows.append(
            plan_row_for_hardware(
                model,
                target_quant,
                hardware,
                entry.name,
                context_length,
                vision_workload,
                min_speed,
            )
        )
    return rows


def multi_gpu_hardware(
    entry: HardwareCatalogEntry,
    count: int,
    system_ram_bytes: int,
) -> HardwareInfo:
    hardware = entry.to_hardware(system_ram_bytes)
    hardware.gpus = [entry.to_hardware(system_ram_bytes).gpus[0] for _ in range(count)]
    return hardware


def plan_multi_gpu_compatibility(
    model: ModelInfo,
    target_quant: str,
    context_length: int,
    image_count: int,
    image_size: int,
    video_frames: int,
    system_ram_bytes: int,
    min_speed: float | None,
) -> list[dict]:
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    rows = []
    for entry in HARDWARE_CATALOG:
        hardware = multi_gpu_hardware(entry, 2, system_ram_bytes)
        rows.append(
            plan_row_for_hardware(
                model,
                target_quant,
                hardware,
                f"2x {entry.name}",
                context_length,
                vision_workload,
                min_speed,
            )
        )
    return rows


def first_runnable(rows: list[dict], fit_type: str) -> dict | None:
    for row in rows:
        if row["fit_type"] == fit_type and row["can_run"] and row["meets_speed"]:
            return row
    return None


def plan_recommendations(
    single_gpu_rows: list[dict],
    multi_gpu_rows: list[dict],
) -> dict:
    return {
        "smallest_full_gpu": first_runnable(single_gpu_rows, "full_gpu"),
        "smallest_partial_offload": first_runnable(
            single_gpu_rows, "partial_offload"
        ),
        "multi_gpu_alternatives": [
            row
            for row in multi_gpu_rows
            if row["fit_type"] == "full_gpu" and row["can_run"] and row["meets_speed"]
        ][:3],
    }


def display_plan(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
    min_speed: float | None = None,
) -> None:
    params = format_params(model.parameter_count)
    active = ""
    if model.is_moe and model.parameter_count_active:
        active = f" ({format_params(model.parameter_count_active)} active)"
    ctx = str(model.context_length) if model.context_length else "unknown"

    lines = [
        f"[bold cyan]Model:[/]  {model.id}",
        f"[bold cyan]Params:[/] {params}{active} | Arch: {model.architecture} | Context: {ctx}",
    ]
    if model.license:
        lines.append(f"[bold cyan]License:[/] {model.license}")
    panel = Panel("\n".join(lines), title="[bold]Model Info[/]", border_style="cyan")
    console.console.print(panel)

    vram_table = Table(
        title=f"VRAM Required (context: {context_length})", show_lines=True
    )
    vram_table.add_column("Quant", style="bold", width=8)
    vram_table.add_column("VRAM", justify="right", width=10)
    vram_table.add_column("Quality Loss", justify="right", width=12)

    vram_by_quant = plan_vram_by_quant(
        model, context_length, image_count, image_size, video_frames
    )
    target_vram = plan_target_vram(
        model,
        context_length,
        target_quant,
        vram_by_quant,
        image_count,
        image_size,
        video_frames,
    )
    for qt, row in vram_by_quant.items():
        vram_bytes = row["vram_bytes"]
        penalty = row["quality_loss"]
        penalty_str = f"-{penalty * 100:.0f}%" if penalty > 0 else "0%"
        marker = " ★" if qt.upper() == target_quant.upper() else ""
        style = "bold green" if qt.upper() == target_quant.upper() else ""
        vram_table.add_row(
            f"{qt}{marker}", format_bytes(vram_bytes), penalty_str, style=style
        )

    console.console.print(vram_table)

    gpu_rows = plan_gpu_compatibility(
        model,
        target_quant,
        target_vram,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
    )
    multi_gpu_rows = plan_multi_gpu_compatibility(
        model,
        target_quant,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
    )
    recommendations = plan_recommendations(gpu_rows, multi_gpu_rows)

    gpu_table = Table(
        title=(
            f"GPU Compatibility ({target_quant}, {format_bytes(target_vram)} required, "
            f"{PLAN_VRAM_HEADROOM_RATIO:.0%} headroom reserved)"
        ),
        show_lines=True,
    )
    gpu_table.add_column("GPU", style="bold", min_width=14)
    gpu_table.add_column("VRAM", justify="right", width=8)
    gpu_table.add_column("Fit", justify="center", width=12)
    gpu_table.add_column("Backend", justify="left", width=12)
    gpu_table.add_column("Est. Speed", justify="right", width=10)
    gpu_table.add_column("Limit", justify="left", width=14)

    for row in gpu_rows:
        gpu_name = row["name"]
        vram_gb = row["vram_gb"]
        fit_type = row["fit_type"]
        if fit_type == "full_gpu":
            fit = "[green]✓ Full GPU[/]"
        elif fit_type == "partial_offload":
            fit = "[yellow]~ Partial[/]"
        else:
            fit = "[red]✗ Too small[/]"
        speed = row["estimated_tok_per_sec"]
        speed_str = f"{speed:.1f} tok/s" if speed is not None else "—"
        backends = ", ".join(row["supported_backends"]) or "unknown"

        gpu_table.add_row(
            gpu_name,
            f"{vram_gb} GB",
            fit,
            backends,
            speed_str,
            row["binding_constraint"],
        )

    console.console.print(gpu_table)

    display_plan_recommendations(recommendations)


def recommendation_line(title: str, row: dict | None) -> str:
    if row is None:
        return f"[bold]{title}:[/] none found"
    backends = ", ".join(row["supported_backends"]) or "unknown"
    return (
        f"[bold]{title}:[/] {row['name']} | {row['fit_type']} | "
        f"required {format_bytes(row['required_memory_bytes'])}, "
        f"usable VRAM {format_bytes(row['usable_vram_bytes'])}, "
        f"reserved {format_bytes(row['reserved_headroom_bytes'])}, "
        f"backend {backends}, limit {row['binding_constraint']}"
    )


def display_plan_recommendations(recommendations: dict) -> None:
    lines = [
        recommendation_line(
            "Smallest full-GPU option", recommendations["smallest_full_gpu"]
        ),
        recommendation_line(
            "Smallest partial-offload option",
            recommendations["smallest_partial_offload"],
        ),
    ]
    multi = recommendations["multi_gpu_alternatives"]
    if multi:
        lines.append("[bold]Practical multi-GPU alternatives:[/]")
        lines.extend(recommendation_line("  option", row) for row in multi)
    else:
        lines.append("[bold]Practical multi-GPU alternatives:[/] none found")

    console.console.print(Panel("\n".join(lines), title="[bold]Reverse Lookup[/]"))
