from __future__ import annotations

from dataclasses import dataclass

from whichvlm.constants import BYTES_PER_GIB, GPU_BANDWIDTH, NVIDIA_COMPUTE_CAPABILITY
from whichvlm.hardware.types import BackendCapability, GPUInfo, HardwareInfo


PLAN_VRAM_HEADROOM_RATIO = 0.10
PLAN_SYSTEM_RAM_BYTES = 64 * BYTES_PER_GIB


@dataclass(frozen=True)
class HardwareCatalogEntry:
    name: str
    vendor: str
    vram_gb: int
    supported_backends: tuple[str, ...]
    os_names: tuple[str, ...]
    shared_memory: bool = False
    price_usd: int | None = None

    def to_hardware(
        self, system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES
    ) -> HardwareInfo:
        vram_bytes = self.vram_gb * BYTES_PER_GIB
        gpu = GPUInfo(
            name=self.name,
            vendor=self.vendor,
            vram_bytes=vram_bytes,
            usable_vram_bytes=int(vram_bytes * (1.0 - PLAN_VRAM_HEADROOM_RATIO)),
            compute_capability=nvidia_compute_capability(self.name),
            memory_bandwidth_gbps=GPU_BANDWIDTH.get(self.name),
            shared_memory=self.shared_memory,
            backend_capabilities=[
                BackendCapability(backend, True) for backend in self.supported_backends
            ],
        )
        return HardwareInfo(
            gpus=[gpu],
            ram_bytes=system_ram_bytes,
            disk_free_bytes=1_000 * BYTES_PER_GIB,
            os=self.os_names[0],
        )


def nvidia_compute_capability(name: str) -> tuple[int, int] | None:
    if name in NVIDIA_COMPUTE_CAPABILITY:
        return NVIDIA_COMPUTE_CAPABILITY[name]
    return next(
        (
            capability
            for key, capability in NVIDIA_COMPUTE_CAPABILITY.items()
            if key in name
        ),
        None,
    )


HARDWARE_CATALOG: tuple[HardwareCatalogEntry, ...] = (
    HardwareCatalogEntry(
        "RTX 4060", "nvidia", 8, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RTX 3060", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RTX 4070", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RTX 4080", "nvidia", 16, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RTX 4090", "nvidia", 24, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RX 7900 XTX", "amd", 24, ("rocm", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry(
        "RTX 5090", "nvidia", 32, ("cuda", "vulkan"), ("linux", "windows")
    ),
    HardwareCatalogEntry("A100 40GB", "nvidia", 40, ("cuda",), ("linux",)),
    HardwareCatalogEntry("L40S", "nvidia", 48, ("cuda",), ("linux",)),
    HardwareCatalogEntry("A100 80GB", "nvidia", 80, ("cuda",), ("linux",)),
    HardwareCatalogEntry("H100", "nvidia", 80, ("cuda",), ("linux",)),
    HardwareCatalogEntry("H200", "nvidia", 141, ("cuda",), ("linux",)),
)
