from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def safe_path_part(value: Any, *, field: str = "value") -> str:
    part = str(value)
    if not part or not _SAFE_PART_RE.match(part):
        raise ValueError(
            f"Invalid {field} for artifact path: {part!r}. "
            "Use only letters, numbers, '.', '_' and '-'."
        )
    return part


def normalize_split(split: Any) -> str:
    if split is None:
        return "default"
    text = str(split).strip()
    if not text or text.lower() in {"none", "null"}:
        return "default"
    return safe_path_part(text, field="split")


def normalize_n(max_samples: Any) -> str:
    if max_samples is None:
        return "all"
    if isinstance(max_samples, str):
        text = max_samples.strip()
        if not text or text.lower() in {"all", "none", "null", "-1"}:
            return "all"
        max_samples = int(text)
    max_samples = int(max_samples)
    if max_samples <= 0:
        return "all"
    return str(max_samples)


def _base(root: str | os.PathLike[str]) -> Path:
    return Path(root)


def activation_dir(
    *,
    root: str | os.PathLike[str] = "artifacts",
    model: str,
    sae: str,
    layer: int,
    dataset: str,
    split: Any = None,
    max_samples: Any = None,
) -> Path:
    return (
        _base(root)
        / "activations"
        / f"model={safe_path_part(model, field='model')}"
        / f"sae={safe_path_part(sae, field='sae')}"
        / f"layer={int(layer)}"
        / f"dataset={safe_path_part(dataset, field='dataset')}"
        / f"split={normalize_split(split)}"
        / f"n={normalize_n(max_samples)}"
    )


def activation_path(
    *,
    root: str | os.PathLike[str] = "artifacts",
    model: str,
    sae: str,
    layer: int,
    dataset: str,
    split: Any = None,
    max_samples: Any = None,
) -> Path:
    return activation_dir(
        root=root,
        model=model,
        sae=sae,
        layer=layer,
        dataset=dataset,
        split=split,
        max_samples=max_samples,
    ) / "acts.pt"


def stat_metrics_path(
    *,
    root: str | os.PathLike[str] = "artifacts",
    model: str,
    sae: str,
    layer: int,
    dataset: str,
    agg: str,
    metric: str,
) -> Path:
    return (
        _base(root)
        / "analyses"
        / "stat"
        / f"model={safe_path_part(model, field='model')}"
        / f"sae={safe_path_part(sae, field='sae')}"
        / f"layer={int(layer)}"
        / f"dataset={safe_path_part(dataset, field='dataset')}"
        / f"agg={safe_path_part(agg, field='agg')}"
        / f"metric={safe_path_part(metric, field='metric')}"
        / "metrics.json"
    )


def geometric_path(
    *,
    root: str | os.PathLike[str] = "artifacts",
    sae: str,
    layer: int,
    method: str,
) -> Path:
    filename = "neighbors.json" if method == "topk_cosine" else "metrics.json"
    return (
        _base(root)
        / "analyses"
        / "geometric"
        / f"sae={safe_path_part(sae, field='sae')}"
        / f"layer={int(layer)}"
        / f"method={safe_path_part(method, field='method')}"
        / filename
    )


def done_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).with_name("DONE")


def analysis_done_path(path: str | os.PathLike[str]) -> Path:
    return done_path(path)


def artifact_meta_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).with_name("meta.json")


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(target)
    return target


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def mark_done(path: str | os.PathLike[str]) -> Path:
    target = done_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(".DONE.tmp")
    tmp.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def git_commit(cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact_meta(
    *,
    script: str,
    params: dict[str, Any],
    repo_dir: str | os.PathLike[str] | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "script": script,
        "params": params,
        "inputs": inputs or {},
    }
    return {
        **payload,
        "git_commit": git_commit(repo_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": stable_hash(payload),
    }
