from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_gpu_pct: int
    notes: str = ""

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


def query_gpus() -> list[GPUInfo]:
    if platform.system() == "Darwin":
        return _query_macos_gpus()
    return _query_nvidia_smi()


def _query_nvidia_smi() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return []
    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        gpus.append(
            GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                memory_used_mib=int(parts[3]),
                utilization_gpu_pct=int(parts[4]),
            )
        )
    return gpus


def _query_macos_gpus() -> list[GPUInfo]:
    name = _macos_gpu_name()
    if name is None:
        return []
    total_mib, used_mib = _macos_memory_mib()
    return [
        GPUInfo(
            index=0,
            name=name,
            memory_total_mib=total_mib,
            memory_used_mib=used_mib,
            utilization_gpu_pct=0,
            notes="unified memory (CPU+GPU shared); GPU utilization not measurable on macOS",
        )
    ]


def _macos_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        for gpu in displays:
            name = (gpu.get("sppci_model") or "").strip()
            if name:
                return name
    except Exception:
        pass
    return None


def _macos_memory_mib() -> tuple[int, int]:
    try:
        import re

        total_bytes = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        vm = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, timeout=5)
        page_size = 16384
        free_pages = 0
        used_pages = 0
        for line in vm.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("page size of"):
                match = re.search(r"(\d+)", stripped)
                if match:
                    page_size = int(match.group(1))
                continue
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            key = key.strip()
            raw = value.strip().rstrip(".")
            try:
                count = int(raw)
            except ValueError:
                continue
            if key == "Pages free":
                free_pages = count
            elif key in ("Pages active", "Pages wired down", "Pages occupied by compressor"):
                used_pages += count
        total_mib = total_bytes // (1024 * 1024)
        used_mib = used_pages * page_size // (1024 * 1024)
        free_mib = free_pages * page_size // (1024 * 1024)
        # clamp used to [0, total]
        return total_mib, max(0, min(used_mib, total_mib - free_mib))
    except Exception:
        return 0, 0


def _nvidia_smi_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    except Exception:
        return None


def gpu_backend_info() -> tuple[str, str | None]:
    if platform.system() == "Darwin":
        return ("apple-silicon", None)
    version = _nvidia_smi_version()
    if version is not None:
        return ("nvidia-smi", version)
    return ("none", None)


def parse_gpu_set(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def classify_gpus(
    gpus: Iterable[GPUInfo],
    *,
    max_gpu_util: int,
    max_used_mib: int,
    min_free_mib: int,
    include: set[int] | None,
    exclude: set[int] | None,
) -> tuple[list[GPUInfo], list[dict[str, object]]]:
    selected: list[GPUInfo] = []
    rows: list[dict[str, object]] = []
    for gpu in gpus:
        reasons: list[str] = []
        if include is not None and gpu.index not in include:
            reasons.append("not in include list")
        if exclude is not None and gpu.index in exclude:
            reasons.append("excluded")
        if gpu.utilization_gpu_pct > max_gpu_util:
            reasons.append(f"utilization {gpu.utilization_gpu_pct}% > {max_gpu_util}%")
        if gpu.memory_used_mib > max_used_mib:
            reasons.append(f"used memory {gpu.memory_used_mib} MiB > {max_used_mib} MiB")
        if gpu.memory_free_mib < min_free_mib:
            reasons.append(f"free memory {gpu.memory_free_mib} MiB < {min_free_mib} MiB")
        is_selected = not reasons
        if is_selected:
            selected.append(gpu)
        rows.append(
            {
                "gpu": gpu.index,
                "name": gpu.name,
                "total_mib": gpu.memory_total_mib,
                "used_mib": gpu.memory_used_mib,
                "free_mib": gpu.memory_free_mib,
                "util_pct": gpu.utilization_gpu_pct,
                "selected": "yes" if is_selected else "no",
                "reason": "selected" if is_selected else "; ".join(reasons),
                "notes": gpu.notes,
            }
        )
    return selected, rows
