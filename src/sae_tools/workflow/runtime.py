from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .artifacts import activation_path, done_path, geometric_path, stat_analysis_dir
from .gpu import GPUInfo, classify_gpus, gpu_backend_info, parse_gpu_set, query_gpus
from .registry import ExperimentSpec, Registry


STAGES = ("activations", "stat", "geometric")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]



@dataclass(frozen=True)
class WorkflowTarget:
    stage: str
    path: Path
    log_path: Path | None
    job: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactRecord:
    stage: str
    target: Path
    status: str
    done_marker: Path
    log_path: Path | None
    reason: str


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    path: Path
    models: int = 0
    saes: int = 0
    layers: int = 0
    datasets: int = 0
    activation_targets: int = 0
    stat_targets: int = 0
    geometric_targets: int = 0
    done: int = 0
    missing: int = 0
    incomplete: int = 0
    failed: int = 0
    error: str = ""


@dataclass(frozen=True)
class ResourceRecord:
    kind: str
    key: str
    path: Path | None
    available: bool
    detail: str


@dataclass(frozen=True)
class EnvironmentCheck:
    name: str
    ok: bool
    detail: str


def requested_stages(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return list(STAGES)
        stages = [item.strip() for item in value.split(",") if item.strip()]
    else:
        stages = [str(item).strip() for item in value if str(item).strip()]
    unknown = sorted(set(stages).difference(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {list(STAGES)}")
    return stages


def target_log_name(stage: str, target: str) -> str:
    import hashlib

    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:10]
    compact = re.sub(r"[^A-Za-z0-9_.=-]+", "_", target)
    compact = compact.strip("_")[-100:]
    return f"{stage}.{compact}.{digest}.log"


def load_experiment_bundle(config_path: str | Path, registry_dir: str | Path) -> tuple[Registry, ExperimentSpec]:
    registry = Registry.load(registry_dir)
    experiment = ExperimentSpec.load(config_path, registry)
    return registry, experiment


def workflow_target_records(
    config_path: str | Path,
    registry_dir: str | Path,
    *,
    artifact_root: str | os.PathLike[str] = "artifacts",
) -> dict[str, list[WorkflowTarget]]:
    registry, experiment = load_experiment_bundle(config_path, registry_dir)
    return target_records_for_experiment(
        experiment=experiment,
        registry=registry,
        artifact_root=artifact_root,
    )


def target_records_for_experiment(
    *,
    experiment: ExperimentSpec,
    registry: Registry,
    artifact_root: str | os.PathLike[str] = "artifacts",
) -> dict[str, list[WorkflowTarget]]:
    experiment_name = experiment.path.stem
    records: dict[str, list[WorkflowTarget]] = {stage: [] for stage in STAGES}
    for job in experiment.activation_jobs(registry):
        dataset = registry.dataset(str(job["dataset"]))
        path = activation_path(
            root=artifact_root,
            experiment=experiment_name,
            model=str(job["model"]),
            sae=str(job["sae"]),
            layer=int(job["layer"]),
            dataset=str(job["dataset"]),
            split=job["split"],
            max_samples=job["max_samples"],
        )
        log_path = Path(
            "logs/activations/"
            f"experiment={experiment_name}.model={job['model']}.sae={job['sae']}.layer={job['layer']}."
            f"dataset={job['dataset']}.split={job['split']}.n={job['n']}.log"
        )
        records["activations"].append(WorkflowTarget("activations", path, log_path, job))

    for job in experiment.stat_batch_jobs(registry):
        path = (
            stat_analysis_dir(
                root=artifact_root,
                experiment=experiment_name,
                model=str(job["model"]),
                sae=str(job["sae"]),
                layer=int(job["layer"]),
                dataset=str(job["dataset"]),
                agg=str(job["agg"]),
            )
            / "DONE"
        )
        log_path = Path(
            "logs/stat_batch/"
            f"experiment={experiment_name}.model={job['model']}.sae={job['sae']}.layer={job['layer']}."
            f"dataset={job['dataset']}.agg={job['agg']}.log"
        )
        records["stat"].append(WorkflowTarget("stat", path, log_path, job))

    for job in experiment.geometric_jobs(registry):
        method = str(job["method"])
        path = geometric_path(
            root=artifact_root,
            experiment=experiment_name,
            sae=str(job["sae"]),
            layer=int(job["layer"]),
            method=method,
        )
        if method == "seed_topk_cosine":
            log_path = Path(
                "logs/geometric/"
                f"experiment={experiment_name}.sae={job['sae']}.layer={job['layer']}."
                "method=seed_topk_cosine.neighbors.log"
            )
        else:
            log_path = Path(
                "logs/geometric/"
                f"experiment={experiment_name}.sae={job['sae']}.layer={job['layer']}.method={method}.filename={path.name}.log"
            )
        records["geometric"].append(WorkflowTarget("geometric", path, log_path, job))
    return records


def workflow_targets(config_path: str | Path, registry_dir: str | Path) -> dict[str, list[str]]:
    return {
        stage: [str(record.path) for record in records]
        for stage, records in workflow_target_records(config_path, registry_dir).items()
    }


def flatten_targets(records: Mapping[str, Sequence[WorkflowTarget]], stages: Sequence[str] | str = "all") -> list[str]:
    return [str(record.path) for stage in requested_stages(stages) for record in records[stage]]


def scan_artifacts(
    config_path: str | Path,
    registry_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    artifact_root: str | os.PathLike[str] = "artifacts",
) -> dict[str, list[ArtifactRecord]]:
    root = default_repo_root() if repo_root is None else Path(repo_root)
    records = workflow_target_records(config_path, registry_dir, artifact_root=artifact_root)
    return {
        stage: [artifact_status(record, repo_root=root) for record in stage_records]
        for stage, stage_records in records.items()
    }


def artifact_status(record: WorkflowTarget, *, repo_root: str | Path) -> ArtifactRecord:
    root = Path(repo_root)
    target_abs = _resolve_repo_path(root, record.path)
    marker_abs = _resolve_repo_path(root, done_path(record.path))
    log_abs = None if record.log_path is None else _resolve_repo_path(root, record.log_path)
    target_exists = target_abs.exists()
    marker_exists = marker_abs.exists()
    log_exists = bool(log_abs and log_abs.exists() and log_abs.stat().st_size > 0)
    if target_exists and marker_exists:
        status = "done"
        reason = "target and DONE marker exist"
    elif target_exists or marker_exists:
        status = "incomplete"
        reason = "target exists without DONE marker" if target_exists else "DONE marker exists without target"
    elif log_exists:
        status = "failed"
        reason = f"log exists: {log_abs.relative_to(root)}"
    else:
        status = "missing"
        reason = "target has not been produced"
    return ArtifactRecord(
        stage=record.stage,
        target=record.path,
        status=status,
        done_marker=done_path(record.path),
        log_path=record.log_path,
        reason=reason,
    )


def scan_experiments(
    experiments_dir: str | Path,
    registry_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    artifact_root: str | os.PathLike[str] = "artifacts",
) -> list[ExperimentSummary]:
    root = default_repo_root() if repo_root is None else Path(repo_root)
    summaries: list[ExperimentSummary] = []
    for config_path in sorted(Path(experiments_dir).glob("*.yaml")):
        summaries.append(experiment_summary(config_path, registry_dir, repo_root=root, artifact_root=artifact_root))
    return summaries


def experiment_summary(
    config_path: str | Path,
    registry_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    artifact_root: str | os.PathLike[str] = "artifacts",
) -> ExperimentSummary:
    path = Path(config_path)
    root = default_repo_root() if repo_root is None else Path(repo_root)
    try:
        registry, experiment = load_experiment_bundle(path, registry_dir)
        records = target_records_for_experiment(experiment=experiment, registry=registry, artifact_root=artifact_root)
        artifact_records = {
            stage: [artifact_status(record, repo_root=root) for record in stage_records]
            for stage, stage_records in records.items()
        }
        status_counts = _status_counts(record for stage_records in artifact_records.values() for record in stage_records)
        return ExperimentSummary(
            name=path.stem,
            path=path,
            models=len(experiment.models),
            saes=len(experiment.saes),
            layers=sum(len(experiment.layers_for_sae(registry, sae_key)) for sae_key in experiment.saes),
            datasets=len(experiment.datasets),
            activation_targets=len(records["activations"]),
            stat_targets=len(records["stat"]),
            geometric_targets=len(records["geometric"]),
            done=status_counts["done"],
            missing=status_counts["missing"],
            incomplete=status_counts["incomplete"],
            failed=status_counts["failed"],
        )
    except Exception as exc:
        return ExperimentSummary(name=path.stem, path=path, error=str(exc))


def scan_resources(
    config_path: str | Path,
    registry_dir: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> list[ResourceRecord]:
    environment = os.environ if env is None else env
    registry, experiment = load_experiment_bundle(config_path, registry_dir)
    records: list[ResourceRecord] = []

    model_root = environment.get("MODEL_ROOT")
    for model_key in experiment.models:
        model = registry.model(model_key)
        path, detail, available = _rooted_resource(model_root, model.local_path, "MODEL_ROOT")
        records.append(ResourceRecord("model", model_key, path, available, detail))

    sae_root = environment.get("SAE_ROOT")
    for sae_key in experiment.saes:
        sae = registry.sae(sae_key)
        for layer in experiment.layers_for_sae(registry, sae_key):
            layer_path = Path(sae.local_dir) / sae.layer_filename(layer)
            path, detail, available = _rooted_resource(sae_root, str(layer_path), "SAE_ROOT")
            records.append(ResourceRecord("sae", f"{sae_key}:L{layer}", path, available, detail))

    dataset_root = environment.get("DATASET_ROOT")
    for item in experiment.datasets:
        dataset = registry.dataset(item.key)
        path, detail, available = _rooted_resource(dataset_root, dataset.folder, "DATASET_ROOT")
        records.append(ResourceRecord("dataset", dataset.key, path, available, detail))

    return records


def check_environment(
    *,
    config_path: str | Path | None,
    registry_dir: str | Path,
    env: Mapping[str, str] | None = None,
) -> list[EnvironmentCheck]:
    environment = os.environ if env is None else env
    checks: list[EnvironmentCheck] = [
        EnvironmentCheck("python", True, sys.executable),
        _python_import_check("sae_tools"),
        _command_version_check("snakemake", ["snakemake", "--version"]),
        _gpu_check(),
    ]
    try:
        registry = Registry.load(registry_dir)
        checks.append(EnvironmentCheck("registry", True, f"loaded {Path(registry_dir)}"))
    except Exception as exc:
        registry = None
        checks.append(EnvironmentCheck("registry", False, str(exc)))

    if config_path is not None and registry is not None:
        try:
            experiment = ExperimentSpec.load(config_path, registry)
            layer_pairs = sum(len(experiment.layers_for_sae(registry, sae_key)) for sae_key in experiment.saes)
            checks.append(EnvironmentCheck("experiment", True, f"loaded {Path(config_path)}"))
            checks.append(EnvironmentCheck("sae layers", True, f"validated {layer_pairs} requested SAE layer entries"))
            checks.append(EnvironmentCheck("datasets", True, f"validated {len(experiment.datasets)} dataset registry entries"))
        except Exception as exc:
            checks.append(EnvironmentCheck("experiment", False, str(exc)))
    elif config_path is None:
        checks.append(EnvironmentCheck("experiment", False, "no experiment selected"))

    if config_path is not None:
        try:
            missing = [record for record in scan_resources(config_path, registry_dir, env=environment) if not record.available]
            if missing:
                checks.append(EnvironmentCheck("resources", False, f"{len(missing)} resources missing or not configured"))
            else:
                checks.append(EnvironmentCheck("resources", True, "all requested local resources exist"))
        except Exception as exc:
            checks.append(EnvironmentCheck("resources", False, str(exc)))

    return checks


def environment_ok(checks: Iterable[EnvironmentCheck]) -> bool:
    return all(check.ok for check in checks)


def build_snakemake_dry_run_command(
    *,
    config_path: str | Path,
    repo_root: str | Path | None = None,
    snakefile: str | Path | None = None,
    registry_dir: str | Path | None = None,
    stages: Sequence[str] | str = "all",
    cores: int = 1,
    snakemake_cmd: str = "snakemake",
    extra_args: Sequence[str] = (),
) -> list[str]:
    root = default_repo_root() if repo_root is None else Path(repo_root)
    registry = root / "configs/registry" if registry_dir is None else Path(registry_dir)
    records = workflow_target_records(config_path, registry)
    targets = flatten_targets(records, stages)
    return [
        *shlex.split(snakemake_cmd),
        "--snakefile",
        str(root / "Snakefile" if snakefile is None else snakefile),
        "--directory",
        str(root),
        "--nolock",
        "--rerun-incomplete",
        "--config",
        f"experiment_config={config_path}",
        "-j",
        str(cores),
        "--dry-run",
        *targets,
        *extra_args,
    ]


def build_idle_runner_command(
    *,
    config_path: str | Path,
    repo_root: str | Path | None = None,
    registry_dir: str | Path | None = None,
    snakefile: str | Path | None = None,
    stages: Sequence[str] | str = "all",
    python_executable: str | Path | None = None,
    run_root: str | Path | None = None,
    max_gpu_util: int = 0,
    max_used_mib: int = 512,
    min_free_mib: int = 30000,
    snakemake_cmd: str = "snakemake",
    current_environment: bool = True,
) -> list[str]:
    root = default_repo_root() if repo_root is None else Path(repo_root)
    command = [
        str(python_executable or sys.executable),
        str(root / "scripts/run_idle_gpu_workflow.py"),
        "--config",
        str(config_path),
        "--registry-dir",
        str(root / "configs/registry" if registry_dir is None else registry_dir),
        "--snakefile",
        str(root / "Snakefile" if snakefile is None else snakefile),
        "--stages",
        ",".join(requested_stages(stages)),
        "--max-gpu-util",
        str(max_gpu_util),
        "--max-used-mib",
        str(max_used_mib),
        "--min-free-mib",
        str(min_free_mib),
        "--snakemake-cmd",
        snakemake_cmd,
    ]
    if run_root is not None:
        command.extend(["--run-root", str(run_root)])
    if current_environment:
        command.append("--no-conda-run")
    return command


def build_runner_target_command(
    *,
    target: str,
    config_path: str | Path,
    snakefile: str | Path,
    repo_root: str | Path | None = None,
    snakemake_cmd: str = "snakemake",
    extra_snakemake_args: Sequence[str] = (),
    conda_bin: str | Path | None = None,
    conda_env: str | None = None,
    no_conda_run: bool = False,
) -> list[str]:
    root = default_repo_root() if repo_root is None else Path(repo_root)
    snakemake_args = [
        "--snakefile",
        str(snakefile),
        "--directory",
        str(root),
        "--nolock",
        "--rerun-incomplete",
        "--config",
        f"experiment_config={config_path}",
        "-j",
        "1",
        target,
        *extra_snakemake_args,
    ]
    if no_conda_run:
        return shlex.split(snakemake_cmd) + snakemake_args
    return [
        str(conda_bin),
        "run",
        "--live-stream",
        "-n",
        str(conda_env),
        *shlex.split(snakemake_cmd),
        *snakemake_args,
    ]


def _status_counts(records: Iterable[ArtifactRecord]) -> dict[str, int]:
    counts = {"done": 0, "missing": 0, "incomplete": 0, "failed": 0}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _rooted_resource(root: str | None, relative_path: str, root_name: str) -> tuple[Path | None, str, bool]:
    path = Path(relative_path).expanduser()
    if not path.is_absolute():
        if not root:
            return None, f"{root_name} is not set", False
        path = Path(root).expanduser() / path
    if path.exists():
        return path, "found", True
    return path, "not found", False


def _python_import_check(module: str) -> EnvironmentCheck:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return EnvironmentCheck(f"import {module}", False, str(exc))
    detail = result.stderr.strip() or result.stdout.strip() or "ok"
    return EnvironmentCheck(f"import {module}", result.returncode == 0, detail)


def _gpu_check() -> EnvironmentCheck:
    backend, version = gpu_backend_info()
    if backend == "nvidia-smi":
        detail = version or "ok"
        return EnvironmentCheck("gpu", True, f"nvidia-smi: {detail}")
    if backend == "apple-silicon":
        return EnvironmentCheck("gpu", True,
            "apple-silicon: development-only (no NVIDIA GPU). "
            "Activation generation and GPU runner require CUDA.")
    return EnvironmentCheck("gpu", False, "no GPU backend detected (nvidia-smi not found)")


def _command_version_check(name: str, command: Sequence[str]) -> EnvironmentCheck:
    if shutil.which(command[0]) is None:
        return EnvironmentCheck(name, False, f"{command[0]} not found on PATH")
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return EnvironmentCheck(name, False, str(exc))
    detail = result.stdout.strip().splitlines()[0] if result.stdout.strip() else result.stderr.strip()
    return EnvironmentCheck(name, result.returncode == 0, detail or "ok")
