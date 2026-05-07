from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
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
