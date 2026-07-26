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
        '00:02.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" '
        '"Alder Lake-P GT1 [UHD Graphics] [46a6]" -r0c "Dell [1028]" '
        '"Device 0b19 [0b19]"\n'
    )

    def fake_run(*args, **kwargs):
        assert args[0] == ["lspci", "-nnmm"]
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)

    gpus = intel.detect_intel_gpus()

    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_bytes == 0
    assert gpus[0].shared_memory is True
    assert "UHD Graphics" in gpus[0].name


def test_detect_arc_pro_b70_marketing_name(monkeypatch):
    detected_name = "Intel(R) Arc(TM) Pro B70 Graphics"
    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [(detected_name, None)])

    gpu = intel.detect_intel_gpus()[0]

    assert gpu.name == detected_name
    assert gpu.vram_bytes == 32 * 1024**3
    assert gpu.memory_bandwidth_gbps == 608.0
    assert gpu.shared_memory is False


def test_detect_arc_pro_b70_from_lspci_device_id(monkeypatch):
    output = (
        '03:00.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" '
        '"Battlemage G31 [Intel Graphics] [e223]" -r00 "Intel Corporation [8086]" '
        '"Device 1701 [1701]"\n'
    )

    def fake_run(*args, **kwargs):
        assert args[0] == ["lspci", "-nnmm"]
        return subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(intel.subprocess, "run", fake_run)

    gpu = intel.detect_intel_gpus()[0]
    ensure_backend_capabilities(gpu, "linux")

    assert gpu.name == "Battlemage G31 [Intel Graphics]"
    assert gpu.vram_bytes == 32 * 1024**3
    assert gpu.memory_bandwidth_gbps == 608.0
    assert gpu.shared_memory is False
    assert has_backend(gpu, "vulkan")


@pytest.mark.parametrize(
    ("detected_name", "pci_id"),
    [
        ("Battlemage G31 [Intel Graphics]", "8086:e220"),
        ("Battlemage G31 [Intel Graphics]", "8086:e221"),
        ("Battlemage G31 [Arc Pro B65]", "8086:e222"),
        ("Intel(R) Arc(TM) Pro B65 Graphics", "8086:e222"),
    ],
)
def test_other_battlemage_devices_do_not_use_b70_profile(
    monkeypatch, detected_name, pci_id
):
    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [(detected_name, pci_id)])

    gpu = intel.detect_intel_gpus()[0]

    assert gpu.name == detected_name
    assert gpu.vram_bytes == 0
    assert gpu.memory_bandwidth_gbps is None
    assert gpu.shared_memory is True


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


def test_detect_arc_pro_b70_from_sysfs_device_id(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "product_name").write_text("Battlemage G31 [Intel Graphics]\n")
    (card / "device").write_text("0xe223\n")

    monkeypatch.setattr(intel, "detect_from_lspci", lambda: [])
    original_sysfs = intel.detect_from_sysfs
    monkeypatch.setattr(intel, "detect_from_sysfs", lambda: original_sysfs(tmp_path))

    gpu = intel.detect_intel_gpus()[0]

    assert gpu.name == "Battlemage G31 [Intel Graphics]"
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
