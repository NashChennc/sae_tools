from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentManifest(_Model):
    schema_version: int = SCHEMA_VERSION
    experiment_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_config: str | None = None
    registry_dir: str | None = None
    registry_hash: str | None = None
    default_page: str = "dashboard"
    index: dict[str, str] = Field(
        default_factory=lambda: {
            "jobs": "index/jobs.parquet",
            "artifacts": "index/artifacts.parquet",
            "features": "index/features.parquet",
            "summaries": "index/summaries.parquet",
            "reports": "index/reports.parquet",
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobRecord(_Model):
    job_id: str
    experiment_id: str
    stage: Literal["activations", "stat", "geometric", "report"]
    model: str | None = None
    sae: str | None = None
    layer: int | None = None
    dataset: str | None = None
    split: str | None = None
    n: str | None = None
    agg: str | None = None
    metric: str | None = None
    method: str | None = None
    status: str = "missing"
    artifact_id: str | None = None
    target_path: str | None = None
    log_path: str | None = None
    reason: str = ""


class ArtifactRecord(_Model):
    artifact_id: str
    experiment_id: str
    kind: str
    role: str
    path: str
    mime: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    status: str = "missing"
    producer: str | None = None
    created_at: str | None = None


class ReportPageRecord(_Model):
    page_id: str
    experiment_id: str
    path: str
    title: str
    layout_path: str = "reports/layout.json"
    created_at: str | None = None


def path_to_str(path: str | Path | None) -> str | None:
    return None if path is None else str(path)
