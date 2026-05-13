from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Sequence

from sae_tools.workflow.registry import ExperimentSpec, Registry
from sae_tools.workflow.runtime import (
    ArtifactRecord,
    EnvironmentCheck,
    ExperimentSummary,
    ResourceRecord,
    build_idle_runner_command,
    build_snakemake_dry_run_command,
    check_environment,
    classify_gpus,
    default_repo_root,
    environment_ok,
    query_gpus,
    requested_stages,
    scan_artifacts,
    scan_experiments,
    scan_resources,
    workflow_target_records,
)


@dataclass(frozen=True)
class ExperimentDetails:
    registry: Registry
    experiment: ExperimentSpec
    artifacts: dict[str, list[ArtifactRecord]]
    resources: list[ResourceRecord]
    checks: list[EnvironmentCheck]

    @property
    def runnable(self) -> bool:
        return environment_ok(self.checks)


@dataclass(frozen=True)
class ArtifactFileRecord:
    kind: str
    role: str
    path: Path
    size_bytes: int
    modified_at: str


@dataclass(frozen=True)
class CommandEvent:
    kind: str
    text: str
    returncode: int | None = None


class TUIBackend:
    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        registry_dir: str | Path | None = None,
        experiments_dir: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.repo_root = default_repo_root() if repo_root is None else Path(repo_root)
        self.registry_dir = self.repo_root / "configs/registry" if registry_dir is None else Path(registry_dir)
        self.experiments_dir = (
            self.repo_root / "configs/experiments" if experiments_dir is None else Path(experiments_dir)
        )
        self.env = os.environ.copy() if env is None else dict(env)

    def list_experiments(self) -> list[ExperimentSummary]:
        return scan_experiments(self.experiments_dir, self.registry_dir, repo_root=self.repo_root)

    def load_details(self, config_path: str | Path) -> ExperimentDetails:
        registry = Registry.load(self.registry_dir)
        experiment = ExperimentSpec.load(config_path, registry)
        return ExperimentDetails(
            registry=registry,
            experiment=experiment,
            artifacts=scan_artifacts(config_path, self.registry_dir, repo_root=self.repo_root),
            resources=scan_resources(config_path, self.registry_dir, env=self.env),
            checks=check_environment(config_path=config_path, registry_dir=self.registry_dir, env=self.env),
        )

    def environment_checks(self, config_path: str | Path | None) -> list[EnvironmentCheck]:
        return check_environment(config_path=config_path, registry_dir=self.registry_dir, env=self.env)

    def artifact_records(self, config_path: str | Path) -> dict[str, list[ArtifactRecord]]:
        return scan_artifacts(config_path, self.registry_dir, repo_root=self.repo_root)

    def artifact_files(self, config_path: str | Path, *, limit: int = 5000) -> list[ArtifactFileRecord]:
        experiment_id = Path(config_path).stem
        bundle = self.repo_root / "artifacts" / "experiments" / experiment_id
        if not bundle.exists():
            return []
        records: list[ArtifactFileRecord] = []
        for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
            stat = path.stat()
            records.append(
                ArtifactFileRecord(
                    kind=_artifact_file_kind(bundle, path),
                    role=_artifact_file_role(path),
                    path=path.relative_to(self.repo_root),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            if len(records) >= limit:
                break
        return records

    def artifact_file_detail(self, path: str | Path) -> str:
        target = _resolve_repo_path(self.repo_root, path)
        if not _is_relative_to(target.resolve(), self.repo_root.resolve()):
            raise PermissionError(f"Artifact path is outside repo root: {path}")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        stat = target.stat()
        lines = [
            f"path: {target.relative_to(self.repo_root)}",
            f"size: {_format_bytes(stat.st_size)}",
            f"modified: {datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}",
            f"type: {_artifact_file_role(target)}",
        ]
        preview = _artifact_preview(target)
        if preview:
            lines.extend(["", preview])
        return "\n".join(lines)

    def gpu_rows(
        self,
        *,
        max_gpu_util: int = 0,
        max_used_mib: int = 512,
        min_free_mib: int = 30000,
    ) -> tuple[list[object], list[dict[str, object]], str | None]:
        try:
            gpus = query_gpus()
        except Exception as exc:
            return [], [], str(exc)
        selected, rows = classify_gpus(
            gpus,
            max_gpu_util=max_gpu_util,
            max_used_mib=max_used_mib,
            min_free_mib=min_free_mib,
            include=None,
            exclude=None,
        )
        return selected, rows, None

    def target_counts(self, config_path: str | Path) -> dict[str, int]:
        records = workflow_target_records(config_path, self.registry_dir)
        return {stage: len(items) for stage, items in records.items()}

    def dry_run_command(self, config_path: str | Path, stages: Sequence[str] | str = "all") -> list[str]:
        return build_snakemake_dry_run_command(
            config_path=config_path,
            repo_root=self.repo_root,
            registry_dir=self.registry_dir,
            stages=requested_stages(stages),
        )

    def run_missing_command(self, config_path: str | Path, stages: Sequence[str] | str = "all") -> list[str]:
        return build_idle_runner_command(
            config_path=config_path,
            repo_root=self.repo_root,
            registry_dir=self.registry_dir,
            stages=requested_stages(stages),
            current_environment=True,
        )

    async def stream_command(self, command: Sequence[str]) -> AsyncIterator[CommandEvent]:
        yield CommandEvent("start", "$ " + " ".join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.repo_root,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdout is not None
        async for raw_line in process.stdout:
            yield CommandEvent("output", raw_line.decode(errors="replace").rstrip())
        returncode = await process.wait()
        yield CommandEvent("exit", f"process exited with {returncode}", returncode=returncode)


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else repo_root / target


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _artifact_file_kind(bundle: Path, path: Path) -> str:
    try:
        parts = path.relative_to(bundle).parts
    except ValueError:
        return "external"
    if len(parts) >= 2 and parts[0] == "objects":
        return parts[1]
    if parts:
        return parts[0]
    return "bundle"


def _artifact_file_role(path: Path) -> str:
    name = path.name
    if name == "DONE":
        return "done-marker"
    if name == "acts.pt":
        return "activation"
    if name == "feature_table.parquet":
        return "feature-table"
    if name in {"summary.json", "top_features.json", "pareto_front.json", "metrics.json", "neighbors.json", "seeds.json"}:
        return name.removesuffix(".json")
    if name == "meta.json":
        return "metadata"
    if path.suffix:
        return path.suffix.lstrip(".")
    return "file"


def _artifact_preview(path: Path, *, max_bytes: int = 64_000) -> str:
    if path.suffix == ".json":
        return _json_preview(path, max_bytes=max_bytes)
    if path.suffix == ".parquet":
        return _parquet_preview(path)
    if path.suffix in {".txt", ".md", ".csv", ".yaml", ".yml", ".log"} or path.name == "DONE":
        return _text_preview(path, max_bytes=max_bytes)
    if path.suffix in {".pt", ".pth", ".pkl", ".pickle", ".safetensors"}:
        return "preview: binary model/data artifact; not loaded by the TUI"
    if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".html"}:
        return "preview: renderable file; open from the report server or filesystem for full view"
    return ""


def _json_preview(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return _text_preview(path, max_bytes=max_bytes)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"json preview failed: {exc}"
    return "json:\n" + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)[:max_bytes]


def _parquet_preview(path: Path) -> str:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        columns = ", ".join(schema.names)
        return (
            "parquet:\n"
            f"  rows: {parquet.metadata.num_rows}\n"
            f"  row_groups: {parquet.metadata.num_row_groups}\n"
            f"  columns: {len(schema.names)}\n"
            f"  names: {columns[:1000]}"
        )
    except Exception as exc:
        return f"parquet preview failed: {exc}"


def _text_preview(path: Path, *, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    suffix = "\n..." if path.stat().st_size > max_bytes else ""
    return "preview:\n" + text + suffix


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
