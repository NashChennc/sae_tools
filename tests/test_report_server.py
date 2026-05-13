from __future__ import annotations

import json
from pathlib import Path

import pytest

from sae_tools.reporting.__main__ import build_parser
from sae_tools.reporting.server import ReportData, ReportServerConfig, Selection, _safe_join, render_dashboard, render_index
from sae_tools.workflow.artifacts import stat_analysis_dir


def test_report_cli_parser_defaults():
    args = build_parser().parse_args(["serve"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.config == Path("configs/experiments/response_grid.yaml")
    assert args.report_root is None


def test_report_server_routes_index_dashboard_and_layer_report(tmp_path):
    report_dir = tmp_path / "artifacts/experiments/response_grid/reports/pages/layer_trends"
    report_dir.mkdir(parents=True)
    (report_dir / "layer_trends.html").write_text("<html><body>layer report</body></html>", encoding="utf-8")

    config = ReportServerConfig.from_paths(
        repo_root=Path.cwd(),
        config_path="configs/experiments/response_grid.yaml",
        artifact_root=tmp_path / "artifacts",
    )
    data = ReportData(config)
    index = render_index(config, data)
    assert "SAE Tools Reports" in index
    assert "/dashboard" in index

    dashboard = render_dashboard(config, data)
    assert "Feature Explorer" in dashboard
    assert "Artifact Status" in dashboard

    assert (report_dir / "layer_trends.html").read_text(encoding="utf-8") == "<html><body>layer report</body></html>"
    with pytest.raises(PermissionError):
        _safe_join(report_dir, Path("../secret.txt"))


def test_report_features_api_reads_top_features(tmp_path):
    stat_dir = stat_analysis_dir(
        root=tmp_path / "artifacts",
        experiment="response_grid",
        model="qwen3-8b",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_response",
        agg="max",
    )
    stat_dir.mkdir(parents=True)
    (stat_dir / "top_features.json").write_text(
        json.dumps({"f1": [{"feature": 42, "score": 0.9, "f1": 0.9, "precision": 1.0, "recall": 0.82}]}),
        encoding="utf-8",
    )
    config = ReportServerConfig.from_paths(
        repo_root=Path.cwd(),
        config_path="configs/experiments/response_grid.yaml",
        artifact_root=tmp_path / "artifacts",
    )

    payload = ReportData(config).features(
        Selection(
            model="qwen3-8b",
            sae="qwen-scope-qwen3-8b-l0-50",
            layer=18,
            dataset="ToxicChat_response",
            agg="max",
        ),
        metric="f1",
        top_k=5,
    )

    assert payload["features"][0]["feature"] == 42
    assert payload["features"][0]["score"] == 0.9
