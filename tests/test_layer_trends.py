from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sae_tools.workflow import stat_analysis_dir


def _load_layer_trends():
    scripts_dir = Path("scripts").resolve()
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "analyze_layer_trends.py"
    spec = importlib.util.spec_from_file_location("analyze_layer_trends", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_feature_table(module, artifact_root: Path, *, layer: int, f1_values: list[float]) -> Path:
    out_dir = stat_analysis_dir(
        root=artifact_root,
        model="qwen3-8b",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=layer,
        dataset="ToxicChat_response",
        agg="max",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table = module.pd.DataFrame(
        {
            "feature": [0, 1, 2, 3],
            "precision": [0.2, 0.7, 0.5, 0.1],
            "recall": [0.3, 0.6, 0.4, 0.2],
            "f1": f1_values,
            "activation_ratio": [0.05, 0.2, 0.1, 0.01],
            "diff": [0.1, 0.8, 0.4, 0.05],
            "pearson": [0.1, 0.3, 0.2, 0.0],
            "auroc": [0.55, 0.75, 0.65, 0.5],
        }
    )
    path = out_dir / "feature_table.parquet"
    table.to_parquet(path, index=False)
    (out_dir / "DONE").write_text("done\n", encoding="utf-8")
    return path


def test_layer_trends_generates_csv_plots_and_markdown(tmp_path):
    module = _load_layer_trends()
    artifact_root = tmp_path / "artifacts"
    _write_feature_table(module, artifact_root, layer=15, f1_values=[0.1, 0.4, 0.3, 0.2])
    _write_feature_table(module, artifact_root, layer=18, f1_values=[0.5, 0.6, 0.4, 0.1])

    result = module.generate_layer_trends(
        config_path="configs/experiments/response_grid.yaml",
        registry_dir="configs/registry",
        artifact_root=artifact_root,
        out_root=tmp_path / "report",
        metrics=("f1",),
        top_k=2,
        saes={"qwen-scope-qwen3-8b-l0-50"},
        datasets={"ToxicChat_response"},
        aggs={"max"},
        layers={15, 18},
    )

    assert result.markdown_path == tmp_path / "report/layer_trends/experiment=response_grid/layer_trends.md"
    assert result.trend_csv.exists()
    assert result.missing_csv.exists()
    assert result.summary_json.exists()
    assert result.plot_paths == (tmp_path / "report/layer_trends/experiment=response_grid/plots/f1.png",)
    assert result.plot_paths[0].exists()
    assert result.missing.empty
    assert len(result.trends) == 2

    row15 = result.trends[result.trends["layer"] == 15].iloc[0]
    assert row15["all_mean"] == pytest.approx(0.25)
    assert row15["all_median"] == pytest.approx(0.25)
    assert row15["topk_mean"] == pytest.approx(0.35)
    assert row15["topk_max"] == pytest.approx(0.4)

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "![f1](plots/f1.png)" in markdown
    assert "| f1 | 18 | 0.55 | qwen3-8b | qwen-scope-qwen3-8b-l0-50 | ToxicChat_response | max |" in markdown
    assert "[Layer trends CSV](layer_trends.csv)" in markdown


def test_layer_trends_records_missing_and_strict_fails(tmp_path):
    module = _load_layer_trends()
    artifact_root = tmp_path / "artifacts"
    _write_feature_table(module, artifact_root, layer=15, f1_values=[0.1, 0.4, 0.3, 0.2])

    result = module.generate_layer_trends(
        config_path="configs/experiments/response_grid.yaml",
        registry_dir="configs/registry",
        artifact_root=artifact_root,
        out_root=tmp_path / "report",
        metrics=("f1",),
        top_k=2,
        saes={"qwen-scope-qwen3-8b-l0-50"},
        datasets={"ToxicChat_response"},
        aggs={"max"},
        layers={15, 18},
    )

    assert len(result.trends) == 1
    assert len(result.missing) == 1
    missing = result.missing.iloc[0]
    assert missing["layer"] == 18
    assert missing["status"] == "missing"
    assert "feature_table.parquet" in missing["path"]
    assert "missing: 1" in result.markdown_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        module.generate_layer_trends(
            config_path="configs/experiments/response_grid.yaml",
            registry_dir="configs/registry",
            artifact_root=artifact_root,
            out_root=tmp_path / "strict-report",
            metrics=("f1",),
            top_k=2,
            saes={"qwen-scope-qwen3-8b-l0-50"},
            datasets={"ToxicChat_response"},
            aggs={"max"},
            layers={15, 18},
            strict=True,
        )
    assert (tmp_path / "strict-report/layer_trends/experiment=response_grid/missing_artifacts.csv").exists()


def test_layer_trends_filters_and_help_smoke(tmp_path):
    module = _load_layer_trends()
    artifact_root = tmp_path / "artifacts"
    _write_feature_table(module, artifact_root, layer=15, f1_values=[0.1, 0.4, 0.3, 0.2])
    _write_feature_table(module, artifact_root, layer=18, f1_values=[0.5, 0.6, 0.4, 0.1])

    result = module.generate_layer_trends(
        config_path="configs/experiments/response_grid.yaml",
        registry_dir="configs/registry",
        artifact_root=artifact_root,
        out_root=tmp_path / "report",
        metrics=("f1", "auroc"),
        top_k=1,
        models={"qwen3-8b"},
        saes={"qwen-scope-qwen3-8b-l0-50"},
        datasets={"ToxicChat_response"},
        aggs={"max"},
        layers={15},
    )

    assert set(result.trends["metric"]) == {"f1", "auroc"}
    assert set(result.trends["layer"]) == {15}
    assert len(result.trends) == 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_layer_trends.py", "--help"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Analyze statistical metric trends across SAE layers" in completed.stdout
