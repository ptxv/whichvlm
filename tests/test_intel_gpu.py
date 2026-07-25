from __future__ import annotations

import subprocess
from io import StringIO

import pytest
from rich.console import Console

from hardware import intel
from hardware.types import (
    GPUInfo,
    HardwareInfo,
    ensure_backend_capabilities,
    has_backend,
)


def test_detect_intel_gpu_from_lspci(monkeypatch):
    output = (
        '00:02.0 "VGA compatible controller" "Intel Corporation" '
        '"Alder Lake-P GT1 [UHD Graphics]" -r0c "Dell" "Device 0b19"\n'
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)

    gpus = intel.detect_intel_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].shared_memory is True
    assert "UHD Graphics" in gpus[0].name


@pytest.mark.parametrize(
    "detected_name",
    [
        "Intel(R) Arc(TM) Pro B70 Graphics",
        "Battlemage G31 [Intel Graphics]",
    ],
)
def test_detect_arc_pro_b70_from_lspci_alias(monkeypatch, detected_name):
    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [detected_name])

    gpu = intel.detect_intel_gpus()[0]
    ensure_backend_capabilities(gpu, "linux")

    assert gpu.name == "Intel Arc Pro B70 Graphics"
    assert gpu.vram_bytes == 32 * 1024**3
    assert gpu.memory_bandwidth_gbps == 608.0
    assert gpu.shared_memory is False
    assert has_backend(gpu, "vulkan")


def test_detect_intel_gpu_ignores_non_display_lspci(monkeypatch):
    output = '00:00.0 "Host bridge" "Intel Corporation" "Device 4621"\n'

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)
    monkeypatch.setattr(intel, "detect_from_sysfs", lambda: [])

    assert intel.detect_intel_gpus() == []


def test_detect_intel_gpu_from_sysfs_when_lspci_missing(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "uevent").write_text("PCI_SLOT_NAME=0000:00:02.0\n")

    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [])
    original_sysfs = intel.detect_from_sysfs
    monkeypatch.setattr(intel, "detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpus = intel.detect_intel_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].name == "Intel Integrated Graphics"


def test_detect_arc_pro_b70_from_sysfs_alias(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "product_name").write_text("Battlemage G31 [Intel Graphics]\n")

    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [])
    original_sysfs = intel.detect_from_sysfs
    monkeypatch.setattr(intel, "detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpu = intel.detect_intel_gpus()[0]

    assert gpu.name == "Intel Arc Pro B70 Graphics"
    assert gpu.vram_bytes == 32 * 1024**3
    assert gpu.memory_bandwidth_gbps == 608.0
    assert gpu.shared_memory is False


def test_display_intel_shared_memory_without_zero_kb(monkeypatch):
    from output import console as console_mod
    from output import display as display_mod

    buf = StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, force_terminal=False))

    display_mod.display_hardware(
        HardwareInfo(
            gpus=[
                GPUInfo(
                    name="Alder Lake-P GT1 [UHD Graphics]",
                    vendor="intel",
                    vram_bytes=0,
                    shared_memory=True,
                )
            ],
            cpu_name="CPU",
            cpu_cores=8,
            ram_bytes=16 * 1024**3,
            disk_free_bytes=100 * 1024**3,
            os="linux",
        )
    )

    output = buf.getvalue()
    assert "shared memory" in output
    assert "0 KB" not in output
