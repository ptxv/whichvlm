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
    memory_bandwidth_gbps: float | None
    compute_capability: tuple[int, int] | None
    supported_backends: tuple[str, ...]
    os_names: tuple[str, ...]
    shared_memory: bool = False
    price_usd: int | None = None

    def to_hardware(
        self,
        system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
        os_name: str | None = None,
    ) -> HardwareInfo:
        os_name = os_name or self.os_names[0]
        vram_bytes = self.vram_gb * BYTES_PER_GIB
        gpu = GPUInfo(
            name=self.name,
            vendor=self.vendor,
            vram_bytes=vram_bytes,
            usable_vram_bytes=int(vram_bytes * (1.0 - PLAN_VRAM_HEADROOM_RATIO)),
            compute_capability=self.compute_capability,
            memory_bandwidth_gbps=self.memory_bandwidth_gbps,
            shared_memory=self.shared_memory,
            backend_capabilities=[
                BackendCapability(
                    backend, backend_supported_on_os(self, backend, os_name)
                )
                for backend in self.supported_backends
            ],
        )
        return HardwareInfo(
            gpus=[gpu],
            ram_bytes=system_ram_bytes,
            disk_free_bytes=1_000 * BYTES_PER_GIB,
            os=os_name,
        )


def catalog_entry(
    name: str,
    vendor: str,
    vram_gb: int,
    supported_backends: tuple[str, ...],
    os_names: tuple[str, ...],
    shared_memory: bool = False,
    price_usd: int | None = None,
) -> HardwareCatalogEntry:
    return HardwareCatalogEntry(
        name=name,
        vendor=vendor,
        vram_gb=vram_gb,
        memory_bandwidth_gbps=GPU_BANDWIDTH.get(name),
        compute_capability=(
            nvidia_compute_capability(name) if vendor == "nvidia" else None
        ),
        supported_backends=supported_backends,
        os_names=os_names,
        shared_memory=shared_memory,
        price_usd=price_usd,
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


def backend_supported_on_os(
    entry: HardwareCatalogEntry, backend: str, os_name: str
) -> bool:
    if os_name not in entry.os_names:
        return False
    return backend != "rocm" or os_name == "linux"


def catalog_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def lookup_catalog_entry(name: str) -> HardwareCatalogEntry | None:
    target = catalog_key(name.removesuffix("(simulated)").strip())
    for entry in HARDWARE_CATALOG:
        entry_key = catalog_key(entry.name)
        if target == entry_key:
            return entry
        if entry_key in {"a100", "h100", "h200"} and target.startswith(entry_key):
            return entry
    return None


HARDWARE_CATALOG: tuple[HardwareCatalogEntry, ...] = (
    catalog_entry(
        "RTX 4060", "nvidia", 8, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RTX 3060", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RTX 4070", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RTX 4080", "nvidia", 16, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RTX 4090", "nvidia", 24, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RX 7900 XTX", "amd", 24, ("rocm", "vulkan"), ("linux", "windows")
    ),
    catalog_entry(
        "RTX 5090", "nvidia", 32, ("cuda", "vulkan"), ("linux", "windows")
    ),
    catalog_entry("A100 40GB", "nvidia", 40, ("cuda",), ("linux",)),
    catalog_entry("L40S", "nvidia", 48, ("cuda",), ("linux",)),
    catalog_entry("A100 80GB", "nvidia", 80, ("cuda",), ("linux",)),
    catalog_entry("H100", "nvidia", 80, ("cuda",), ("linux",)),
    catalog_entry("H200", "nvidia", 141, ("cuda",), ("linux",)),
)
