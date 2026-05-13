from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from sae_tools.workflow.registry import ExperimentSpec, Registry

from .ids import DEFAULT_EXPERIMENT_ID, artifact_id as build_artifact_id, experiment_id_from_path
from .ids import job_id as build_job_id
from .manifest import ArtifactRecord, ExperimentManifest, JobRecord, ReportPageRecord


INDEX_TABLES = ("jobs", "artifacts", "features", "summaries", "reports")

JOBS_COLUMNS = tuple(JobRecord.model_fields)
ARTIFACT_COLUMNS = tuple(ArtifactRecord.model_fields)
REPORT_COLUMNS = tuple(ReportPageRecord.model_fields)
FEATURE_COLUMNS = (
    "experiment_id",
    "stat_id",
    "model",
    "sae",
    "layer",
    "dataset",
    "agg",
    "feature",
    "precision",
    "recall",
    "f1",
    "activation_ratio",
    "diff",
    "pearson",
    "auroc",
)
SUMMARY_COLUMNS = (
    "experiment_id",
    "summary_id",
    "kind",
    "model",
    "sae",
    "layer",
    "dataset",
    "agg",
    "metric",
    "method",
    "name",
    "value",
)


class ExperimentStore:
    """File-backed experiment bundle store.

    The store owns a single experiment directory under
    ``<artifact_root>/experiments/<experiment_id>``. Structured indexes are
    Parquet tables; large artifacts remain native files under ``objects/``.
    """

    def __init__(self, root: str | os.PathLike[str], experiment_id: str = DEFAULT_EXPERIMENT_ID) -> None:
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.path = self.root / "experiments" / experiment_id
        self.manifest_path = self.path / "manifest.json"

    @classmethod
    def open(cls, root: str | os.PathLike[str], experiment_id: str = DEFAULT_EXPERIMENT_ID) -> "ExperimentStore":
        store = cls(root, experiment_id)
        if not store.manifest_path.exists():
            raise FileNotFoundError(f"Experiment store manifest not found: {store.manifest_path}")
        return store

    @classmethod
    def from_artifact_path(cls, path: str | os.PathLike[str]) -> "ExperimentStore | None":
        parts = Path(path).parts
        try:
            idx = parts.index("experiments")
        except ValueError:
            return None
        if idx + 1 >= len(parts):
            return None
        root_parts = parts[:idx]
        root = Path(*root_parts) if root_parts else Path(".")
        return cls(root, parts[idx + 1])

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str],
        *,
        experiment: ExperimentSpec,
        registry: Registry,
        experiment_id: str | None = None,
        overwrite: bool = False,
    ) -> "ExperimentStore":
        experiment_id = experiment_id or experiment_id_from_path(experiment.path)
        store = cls(root, experiment_id)
        if overwrite and store.path.exists():
            shutil.rmtree(store.path)
        store.path.mkdir(parents=True, exist_ok=True)
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            source_config=str(experiment.path),
            registry_dir=str(registry.directory),
            registry_hash=_directory_hash(registry.directory),
        )
        store.write_manifest(manifest)
        store._write_text("config.resolved.yaml", Path(experiment.path).read_text(encoding="utf-8") if Path(experiment.path).exists() else "")
        store._write_text("registry.snapshot.json", _registry_snapshot(registry))
        store.initialize_indices()
        store.record_workflow_plan(experiment=experiment, registry=registry)
        return store

    def ensure_initialized(self) -> None:
        if not self.manifest_path.exists():
            self.write_manifest(ExperimentManifest(experiment_id=self.experiment_id))
        self.initialize_indices()

    def read_manifest(self) -> ExperimentManifest:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return ExperimentManifest.model_validate(json.load(handle))

    def write_manifest(self, manifest: ExperimentManifest) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump(mode="json")
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.manifest_path

    def initialize_indices(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "jobs": JOBS_COLUMNS,
            "artifacts": ARTIFACT_COLUMNS,
            "features": FEATURE_COLUMNS,
            "summaries": SUMMARY_COLUMNS,
            "reports": REPORT_COLUMNS,
        }
        for name, columns in defaults.items():
            path = self.index_path(name)
            if not path.exists():
                pd.DataFrame(columns=columns).to_parquet(path, index=False)

    @property
    def index_dir(self) -> Path:
        return self.path / "index"

    @property
    def objects_dir(self) -> Path:
        return self.path / "objects"

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    def index_path(self, name: str) -> Path:
        if name not in INDEX_TABLES:
            raise KeyError(f"Unknown store index table: {name}")
        return self.index_dir / f"{name}.parquet"

    def artifact_path(self, kind: str, **dimensions: Any) -> Path:
        from sae_tools.workflow.artifacts import (
            activation_path,
            geometric_path,
            stat_analysis_dir,
            stat_metrics_path,
            stat_plot_path,
        )

        if kind == "activation":
            return activation_path(root=self.root, experiment=self.experiment_id, **dimensions)
        if kind == "stat_dir":
            return stat_analysis_dir(root=self.root, experiment=self.experiment_id, **dimensions)
        if kind == "stat_metric":
            return stat_metrics_path(root=self.root, experiment=self.experiment_id, **dimensions)
        if kind == "stat_plot":
            return stat_plot_path(root=self.root, experiment=self.experiment_id, **dimensions)
        if kind == "geometric":
            return geometric_path(root=self.root, experiment=self.experiment_id, **dimensions)
        raise KeyError(f"Unknown artifact path kind: {kind}")

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.index_path(name)
        if not path.exists():
            self.initialize_indices()
        return pd.read_parquet(path)

    def write_table(self, name: str, table: pd.DataFrame) -> Path:
        path = self.index_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    def upsert_rows(self, name: str, rows: Iterable[Mapping[str, Any]], key: str) -> Path:
        existing = self.read_table(name)
        incoming = pd.DataFrame(list(rows))
        if incoming.empty:
            return self.index_path(name)
        if not existing.empty:
            for column in existing.columns:
                if column not in incoming.columns:
                    incoming[column] = pd.NA
            incoming = incoming.loc[:, existing.columns]
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
        combined = combined.drop_duplicates(subset=[key], keep="last", ignore_index=True)
        return self.write_table(name, combined)

    def record_job(self, record: JobRecord | Mapping[str, Any]) -> Path:
        item = record if isinstance(record, JobRecord) else JobRecord.model_validate(record)
        return self.upsert_rows("jobs", [item.model_dump(mode="json")], key="job_id")

    def record_artifact(self, record: ArtifactRecord | Mapping[str, Any]) -> Path:
        item = record if isinstance(record, ArtifactRecord) else ArtifactRecord.model_validate(record)
        return self.upsert_rows("artifacts", [item.model_dump(mode="json")], key="artifact_id")

    def record_report(self, record: ReportPageRecord | Mapping[str, Any]) -> Path:
        item = record if isinstance(record, ReportPageRecord) else ReportPageRecord.model_validate(record)
        return self.upsert_rows("reports", [item.model_dump(mode="json")], key="page_id")

    def record_artifact_path(
        self,
        *,
        kind: str,
        role: str,
        path: str | Path,
        dimensions: Mapping[str, Any] | None = None,
        producer: str | None = None,
        mime: str | None = None,
    ) -> Path:
        record = artifact_record_for_path(
            root=self.root,
            experiment_id=self.experiment_id,
            kind=kind,
            role=role,
            path=path,
            producer=producer,
            dimensions=dimensions,
            mime=mime,
        )
        return self.record_artifact(record)

    def record_workflow_plan(self, *, experiment: ExperimentSpec, registry: Registry) -> None:
        from sae_tools.workflow.runtime import target_records_for_experiment

        self.ensure_initialized()
        records = target_records_for_experiment(experiment=experiment, registry=registry, artifact_root=self.root)
        job_rows: list[JobRecord] = []
        artifact_rows: list[ArtifactRecord] = []
        for stage, targets in records.items():
            for target in targets:
                dims = _job_dimensions(stage, target.job)
                target_path = target.path
                status = "done" if target_path.exists() else "missing"
                artifact_id = build_artifact_id(stage, **dims)
                job_rows.append(
                    JobRecord(
                        job_id=build_job_id(stage, **dims),
                        experiment_id=self.experiment_id,
                        stage=stage,  # type: ignore[arg-type]
                        model=dims.get("model"),
                        sae=dims.get("sae"),
                        layer=None if dims.get("layer") is None else int(dims["layer"]),
                        dataset=dims.get("dataset"),
                        split=dims.get("split"),
                        n=dims.get("n"),
                        agg=dims.get("agg"),
                        metric=dims.get("metric"),
                        method=dims.get("method"),
                        status=status,
                        artifact_id=artifact_id,
                        target_path=str(_relative_to(target_path, self.root)),
                        log_path=None if target.log_path is None else str(target.log_path),
                    )
                )
                artifact_rows.append(
                    artifact_record_for_path(
                        root=self.root,
                        experiment_id=self.experiment_id,
                        kind=stage,
                        role="target",
                        path=target_path,
                        producer="snakemake",
                        dimensions=dims,
                    )
                )
        self.upsert_rows("jobs", [row.model_dump(mode="json") for row in job_rows], key="job_id")
        self.upsert_rows("artifacts", [row.model_dump(mode="json") for row in artifact_rows], key="artifact_id")

    def replace_features(self, rows: pd.DataFrame, *, stat_id: str) -> Path:
        existing = self.read_table("features")
        if not existing.empty and "stat_id" in existing.columns:
            existing = existing[existing["stat_id"] != stat_id]
        combined = rows if existing.empty else pd.concat([existing, rows], ignore_index=True)
        return self.write_table("features", combined)

    def replace_stat_features(
        self,
        table: pd.DataFrame,
        *,
        model: str,
        sae: str,
        layer: int,
        dataset: str,
        agg: str,
    ) -> Path:
        stat_id = build_artifact_id("stat", model=model, sae=sae, layer=layer, dataset=dataset, agg=agg)
        rows = table.copy()
        rows["experiment_id"] = self.experiment_id
        rows["stat_id"] = stat_id
        rows["model"] = model
        rows["sae"] = sae
        rows["layer"] = int(layer)
        rows["dataset"] = dataset
        rows["agg"] = agg
        for column in FEATURE_COLUMNS:
            if column not in rows.columns:
                rows[column] = pd.NA
        return self.replace_features(rows.loc[:, FEATURE_COLUMNS], stat_id=stat_id)

    def query_jobs(self, **filters: Any) -> pd.DataFrame:
        return _filter(self.read_table("jobs"), filters)

    def query_artifacts(self, **filters: Any) -> pd.DataFrame:
        return _filter(self.read_table("artifacts"), filters)

    def query_features(self, **filters: Any) -> pd.DataFrame:
        return _filter(self.read_table("features"), filters)

    def query_summaries(self, **filters: Any) -> pd.DataFrame:
        return _filter(self.read_table("summaries"), filters)

    def _write_text(self, relative: str, payload: str) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path


def _filter(table: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    result = table
    for key, value in filters.items():
        if value is None or key not in result.columns:
            continue
        result = result[result[key] == value]
    return result.reset_index(drop=True)


def _directory_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.yaml")):
        digest.update(item.name.encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _registry_snapshot(registry: Registry) -> str:
    payload = {
        "directory": str(registry.directory),
        "models": {key: value.__dict__ for key, value in registry.models.items()},
        "saes": {key: value.__dict__ for key, value in registry.saes.items()},
        "datasets": {key: value.__dict__ for key, value in registry.datasets.items()},
        "analyses": {key: value.__dict__ for key, value in registry.analyses.items()},
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def artifact_record_for_path(
    *,
    root: str | os.PathLike[str],
    experiment_id: str,
    kind: str,
    role: str,
    path: str | Path,
    producer: str | None = None,
    dimensions: Mapping[str, Any] | None = None,
    mime: str | None = None,
) -> ArtifactRecord:
    target = Path(path)
    try:
        relative = target.relative_to(Path(root))
    except ValueError:
        relative = target
    return ArtifactRecord(
        artifact_id=build_artifact_id(kind, **(dimensions or {})),
        experiment_id=experiment_id,
        kind=kind,
        role=role,
        path=str(relative),
        mime=mime,
        size_bytes=target.stat().st_size if target.exists() else None,
        status="done" if target.exists() else "missing",
        producer=producer,
        created_at=_mtime_iso(target) if target.exists() else None,
    )


def _job_dimensions(stage: str, job: Mapping[str, Any]) -> dict[str, Any]:
    keys_by_stage = {
        "activations": ("model", "sae", "layer", "dataset", "split", "n"),
        "stat": ("model", "sae", "layer", "dataset", "agg"),
        "geometric": ("sae", "layer", "method"),
    }
    return {key: job[key] for key in keys_by_stage.get(stage, ()) if key in job and job[key] is not None}


def _relative_to(path: str | Path, root: str | Path) -> Path:
    target = Path(path)
    try:
        return target.relative_to(Path(root))
    except ValueError:
        return target


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
