from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sae_tools.workflow.artifacts import normalize_n, normalize_split, safe_path_part


DEFAULT_EXPERIMENT_ID = "standalone"


def experiment_id_from_path(path: str | Path | None) -> str:
    if path is None:
        return DEFAULT_EXPERIMENT_ID
    return safe_path_part(Path(path).stem, field="experiment")


def normalize_dims(values: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        key = safe_path_part(key, field="dimension")
        if key == "split":
            normalized[key] = normalize_split(value)
        elif key in {"n", "max_samples"}:
            normalized["n"] = normalize_n(value)
        elif key == "layer":
            normalized[key] = str(int(value))
        else:
            normalized[key] = safe_path_part(value, field=key)
    return normalized


def artifact_id(kind: str, **dimensions: Any) -> str:
    return _record_id(kind, normalize_dims(dimensions))


def job_id(stage: str, **dimensions: Any) -> str:
    return _record_id(stage, normalize_dims(dimensions))


def _record_id(prefix: str, dimensions: Mapping[str, str]) -> str:
    prefix = safe_path_part(prefix, field="record prefix")
    parts = [f"{key}={dimensions[key]}" for key in sorted(dimensions)]
    return prefix if not parts else f"{prefix}:{'.'.join(parts)}"
