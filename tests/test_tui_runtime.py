from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sae_tools.workflow import runtime


def test_tui_experiment_scan_discovers_default_configs():
    summaries = runtime.scan_experiments("configs/experiments", "configs/registry", repo_root=Path.cwd())
    by_name = {summary.name: summary for summary in summaries}

    assert {"response_grid", "safety_grid"}.issubset(by_name)
    assert by_name["response_grid"].activation_targets == 42
    assert by_name["response_grid"].stat_targets == 84
    assert by_name["response_grid"].geometric_targets == 28


def test_tui_sae_resource_scan_checks_all_requested_layers(tmp_path):
    env = {
        "MODEL_ROOT": str(tmp_path / "models"),
        "SAE_ROOT": str(tmp_path / "saes"),
        "DATASET_ROOT": str(tmp_path / "datasets"),
    }
    records = runtime.scan_resources("configs/experiments/response_grid.yaml", "configs/registry", env=env)
    sae_records = [record for record in records if record.kind == "sae"]

    assert len(sae_records) == 14
    assert {record.key.split(":L")[0] for record in sae_records} == {
        "qwen-scope-qwen3-8b-l0-50",
        "qwen-scope-qwen3-8b-l0-100",
    }
    assert {int(record.key.split(":L")[1]) for record in sae_records} == {15, 18, 21, 24, 27, 30, 33}
    assert all(record.path is not None and f"layer{record.key.split(':L')[1]}.sae.pt" in str(record.path) for record in sae_records)


def test_environment_checks_report_missing_runtime_tools(monkeypatch):
    def fake_which(command: str) -> str | None:
        if command in {"snakemake", "nvidia-smi"}:
            return None
        return f"/usr/bin/{command}"

    monkeypatch.setattr(runtime.shutil, "which", fake_which)
    checks = runtime.check_environment(
        config_path="configs/experiments/safety_grid.yaml",
        registry_dir="configs/registry",
        env={},
    )

    failures = {check.name: check.detail for check in checks if not check.ok}
    assert "snakemake" in failures
    assert "nvidia-smi" in failures
    assert "resources" in failures


def test_tui_backend_builds_expected_commands():
    config = Path("configs/experiments/response_grid.yaml")
    dry_run = runtime.build_snakemake_dry_run_command(
        config_path=config,
        repo_root=Path.cwd(),
        registry_dir="configs/registry",
        stages=["activations"],
    )
    assert dry_run[:2] == ["snakemake", "--snakefile"]
    assert "--dry-run" in dry_run
    assert dry_run.count("--config") == 1
    assert dry_run[dry_run.index("--config") + 1] == f"experiment_config={config}"
    assert len(dry_run[dry_run.index("--dry-run") + 1 :]) == 42

    run_missing = runtime.build_idle_runner_command(
        config_path=config,
        repo_root=Path.cwd(),
        registry_dir="configs/registry",
        stages=["stat", "geometric"],
    )
    assert run_missing[0] == sys.executable
    assert run_missing[1].endswith("scripts/run_idle_gpu_workflow.py")
    assert run_missing[run_missing.index("--stages") + 1] == "stat,geometric"
    assert "--no-conda-run" in run_missing


def test_python_module_help_smoke():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    result = subprocess.run(
        [sys.executable, "-m", "sae_tools_tui", "--help"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "sae-tools-tui" in result.stdout
