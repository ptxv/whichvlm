from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from data.gpu import BYTES_PER_GIB
from hardware.gpu_db import normalize_detected_gpu_name, resolve_detected_bandwidth
from hardware.types import GPUInfo

logger = logging.getLogger(__name__)

ARC_PRO_B70_KEY = "Arc Pro B70"
ARC_PRO_B70_PCI_ID = "8086:e223"
PCI_ID_RE = re.compile(r"\[([0-9a-f]{4})\]$", re.IGNORECASE)

DetectedIntelGPU = tuple[str, str | None]

DISPLAY_CLASSES = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)


def parse_lspci_device(line: str) -> DetectedIntelGPU:
    parts = [p.strip() for p in line.split('"') if p.strip() and p.strip() != "\t"]
    for i, part in enumerate(parts):
        if part.lower().startswith("intel corporation") and i + 1 < len(parts):
            vendor_id = PCI_ID_RE.search(part)
            device = parts[i + 1]
            device_id = PCI_ID_RE.search(device)
            name = PCI_ID_RE.sub("", device).strip()
            pci_id = (
                f"{vendor_id.group(1).lower()}:{device_id.group(1).lower()}"
                if vendor_id and device_id
                else None
            )
            return name, pci_id
    return "Intel Integrated Graphics", None


def detect_from_lspci() -> list[DetectedIntelGPU]:
    try:
        result = subprocess.run(
            ["lspci", "-nnmm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("lspci not available or timed out")
        return []

    if result.returncode != 0:
        return []

    devices: list[DetectedIntelGPU] = []
    seen: set[DetectedIntelGPU] = set()
    for line in result.stdout.splitlines():
        line_lower = line.lower()
        if "intel" not in line_lower or not any(
            display_class in line_lower for display_class in DISPLAY_CLASSES
        ):
            continue
        device = parse_lspci_device(line)
        if device not in seen:
            devices.append(device)
            seen.add(device)
    return devices


def detect_from_sysfs(
    drm_path: Path = Path("/sys/class/drm"),
) -> list[DetectedIntelGPU]:
    devices: list[DetectedIntelGPU] = []
    seen: set[DetectedIntelGPU] = set()
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []

    for card in cards:
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x8086":
            continue

        name = "Intel Integrated Graphics"
        try:
            product_name = (device / "product_name").read_text().strip()
            if product_name:
                name = product_name
        except OSError:
            pass

        pci_id = None
        try:
            device_id = (device / "device").read_text().strip().lower()
            pci_id = f"{vendor.removeprefix('0x')}:{device_id.removeprefix('0x')}"
        except OSError:
            pass

        detected = (name, pci_id)
        if detected not in seen:
            devices.append(detected)
            seen.add(detected)
    return devices


def detect_intel_gpus() -> list[GPUInfo]:
    devices = detect_from_lspci() or detect_from_sysfs()

    gpus: list[GPUInfo] = []
    for detected_name, pci_id in devices:
        is_arc_pro_b70 = (
            normalize_detected_gpu_name(detected_name) == ARC_PRO_B70_KEY
            or pci_id == ARC_PRO_B70_PCI_ID
        )
        vram_bytes = 32 * BYTES_PER_GIB if is_arc_pro_b70 else 0
        gpus.append(
            GPUInfo(
                name=detected_name,
                vendor="intel",
                vram_bytes=vram_bytes,
                memory_bandwidth_gbps=(
                    resolve_detected_bandwidth(ARC_PRO_B70_KEY)
                    if is_arc_pro_b70
                    else None
                ),
                shared_memory=not is_arc_pro_b70,
            )
        )
    return gpus
