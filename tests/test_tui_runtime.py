from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sae_tools_tui.app import CreateExperimentScreen, SAEWorkflowTUI
from sae_tools_tui.backend import TUIBackend
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
        if command == "snakemake":
            return None
        return f"/usr/bin/{command}"

    monkeypatch.setattr(runtime.shutil, "which", fake_which)
    monkeypatch.setattr(runtime, "gpu_backend_info", lambda: ("none", None))
    checks = runtime.check_environment(
        config_path="configs/experiments/safety_grid.yaml",
        registry_dir="configs/registry",
        env={},
    )

    failures = {check.name: check.detail for check in checks if not check.ok}
    assert "snakemake" in failures
    assert "gpu" in failures
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


def test_tui_backend_scans_artifact_files_and_details(tmp_path):
    bundle = tmp_path / "artifacts/experiments/response_grid/objects/stat/model=qwen3-8b"
    summary = bundle / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status": "ok", "count": 3}', encoding="utf-8")
    table_path = bundle / "feature_table.parquet"
    pd.DataFrame({"feature": [1, 2], "f1": [0.5, 0.6]}).to_parquet(table_path, index=False)
    activation = tmp_path / "artifacts/experiments/response_grid/objects/activations/model=qwen3-8b/acts.pt"
    activation.parent.mkdir(parents=True)
    activation.write_bytes(b"placeholder")

    backend = TUIBackend(repo_root=tmp_path)
    records = backend.artifact_files("configs/experiments/response_grid.yaml")
    by_name = {record.path.name: record for record in records}

    assert by_name["summary.json"].kind == "stat"
    assert by_name["summary.json"].role == "summary"
    assert by_name["feature_table.parquet"].role == "feature-table"
    assert by_name["acts.pt"].role == "activation"

    json_detail = backend.artifact_file_detail(by_name["summary.json"].path)
    parquet_detail = backend.artifact_file_detail(by_name["feature_table.parquet"].path)
    binary_detail = backend.artifact_file_detail(by_name["acts.pt"].path)

    assert '"status": "ok"' in json_detail
    assert "rows: 2" in parquet_detail
    assert "not loaded by the TUI" in binary_detail


def test_tui_backend_saves_valid_experiment_config(tmp_path):
    backend = TUIBackend(repo_root=Path.cwd(), registry_dir="configs/registry", experiments_dir=tmp_path)
    data = {
        "models": ["qwen3-8b"],
        "saes": ["qwen-scope-qwen3-8b-l0-50"],
        "layers": [18],
        "datasets": [{"id": "ToxicChat_prompt", "max_samples": 10}],
        "activation": {"overwrite": False, "batch_size": 2},
        "analyses": ["stat_basic", "geo_basic"],
    }

    path = backend.save_experiment(data, "tui-created-smoke")

    assert path == tmp_path / "tui-created-smoke.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == data
    with pytest.raises(FileExistsError):
        backend.save_experiment(data, "tui-created-smoke")


def test_tui_backend_rejects_unsafe_experiment_names(tmp_path):
    backend = TUIBackend(repo_root=Path.cwd(), registry_dir="configs/registry", experiments_dir=tmp_path)

    with pytest.raises(ValueError):
        backend.experiment_path("../bad")
    with pytest.raises(ValueError):
        backend.experiment_path("bad value")
    assert backend.experiment_path("ok.yaml") == tmp_path / "ok.yaml"


def test_tui_create_screen_saves_config_from_independent_screen(tmp_path):
    async def run_flow() -> None:
        app = SAEWorkflowTUI(repo_root=Path.cwd())
        app.backend = TUIBackend(repo_root=Path.cwd(), registry_dir="configs/registry", experiments_dir=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.open_create_experiment()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CreateExperimentScreen)
            screen.draft.name = "ui-smoke"
            screen.draft.models = {"qwen3-8b"}
            screen.draft.saes = {"qwen-scope-qwen3-8b-l0-50"}
            screen.draft.layers = {18}
            screen.draft.datasets = {"ToxicChat_prompt"}
            screen.draft.dataset_max_samples = {"ToxicChat_prompt": "10"}
            screen.draft.analyses = {"stat_basic", "geo_basic"}
            screen._save()
            await pilot.pause()
            assert app.selected_config == tmp_path / "ui-smoke.yaml"

    asyncio.run(run_flow())

    path = tmp_path / "ui-smoke.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["activation"] == {"overwrite": False, "batch_size": 2}
    assert data["layers"] == [18]


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
