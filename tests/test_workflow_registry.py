from pathlib import Path

import pytest

from sae_tools.workflow import (
    activation_path,
    done_path,
    geometric_path,
    stat_analysis_dir,
    stat_feature_table_path,
    stat_plot_path,
    load_registry,
    normalize_n,
    normalize_split,
    safe_path_part,
    stat_metrics_path,
)
from sae_tools.workflow.registry import ExperimentSpec


def test_registry_loads_default_yaml_and_validates_backends():
    registry = load_registry("configs/registry")

    assert "qwen3-8b-guard" in registry.models
    assert "qwen3-8b" in registry.models
    assert "qwen-scope-qwen3-8b-l0-50" in registry.saes
    assert registry.sae("qwen-scope-qwen3-8b-l0-100").layers == (15, 18, 21, 24, 27, 30, 33)
    assert registry.sae("qwen-scope-qwen3-8b-l0-100").top_k == 100
    assert registry.dataset("ToxicChat_prompt").adapter == "ToxicChat"
    assert registry.dataset("ToxicChat_response").label_field == "prompt_label"
    assert registry.dataset("Aegis2.0_response").label_field == "response_label"
    assert registry.dataset("ToxicChat_prompt").max_samples == 1000
    assert registry.analysis("stat_basic").metrics == ("pearson", "auroc", "f1")
    assert registry.validate_backends() == []


def test_experiment_expands_activation_stat_and_geometric_jobs():
    registry = load_registry("configs/registry")
    experiment = ExperimentSpec.load("configs/experiments/safety_grid.yaml", registry)

    activation_jobs = experiment.activation_jobs(registry)
    stat_jobs = experiment.stat_jobs(registry)
    stat_batch_jobs = experiment.stat_batch_jobs(registry)
    geo_jobs = experiment.geometric_jobs(registry)

    assert experiment.activation_batch_size == 2
    assert len(activation_jobs) == 2
    assert activation_jobs[0]["model"] == "qwen3-8b"
    assert activation_jobs[0]["layer"] == 18
    assert activation_jobs[0]["n"] == "1000"
    assert activation_jobs[0]["batch_size"] == 2
    assert {job["model"] for job in activation_jobs} == {"qwen3-8b"}
    assert len(stat_jobs) == 12
    assert len(stat_batch_jobs) == 4
    assert {job["metric"] for job in stat_jobs} == {"pearson", "auroc", "f1"}
    assert {job["method"] for job in geo_jobs} == {"norm", "seed_topk_cosine"}


def test_response_grid_expands_requested_layers_only():
    registry = load_registry("configs/registry")
    experiment = ExperimentSpec.load("configs/experiments/response_grid.yaml", registry)

    activation_jobs = experiment.activation_jobs(registry)
    stat_batch_jobs = experiment.stat_batch_jobs(registry)
    geo_jobs = experiment.geometric_jobs(registry)

    assert experiment.layers == (15, 18, 21, 24, 27, 30, 33)
    assert len(activation_jobs) == 42
    assert len(stat_batch_jobs) == 84
    assert len(geo_jobs) == 28
    assert {job["sae"] for job in activation_jobs} == {
        "qwen-scope-qwen3-8b-l0-50",
        "qwen-scope-qwen3-8b-l0-100",
    }
    assert {job["layer"] for job in activation_jobs} == {15, 18, 21, 24, 27, 30, 33}


def test_deterministic_artifact_paths_are_normalized():
    acts = activation_path(
        root="artifacts",
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        split=None,
        max_samples=1000,
    )
    assert acts == Path(
        "artifacts/experiments/standalone/objects/activations/model=qwen3-8b-guard/"
        "sae=qwen-scope-qwen3-8b-l0-50/layer=18/dataset=ToxicChat_prompt/split=default/n=1000/acts.pt"
    )
    assert stat_metrics_path(
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        agg="max",
        metric="auroc",
    ) == Path(
        "artifacts/experiments/standalone/objects/stat/model=qwen3-8b-guard/"
        "sae=qwen-scope-qwen3-8b-l0-50/layer=18/dataset=ToxicChat_prompt/agg=max/metric=auroc/metrics.json"
    )
    stat_dir = stat_analysis_dir(
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        agg="max",
    )
    assert stat_feature_table_path(
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        agg="max",
    ) == stat_dir / "feature_table.parquet"
    assert stat_plot_path(
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        agg="max",
        color="diff",
    ) == stat_dir / "plots" / "pr_space.color=diff.png"
    assert geometric_path(sae="qwen-scope-qwen3-8b-l0-50", layer=18, method="seed_topk_cosine").name == "neighbors.json"
    assert geometric_path(
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        method="seed_topk_cosine",
        experiment="response_grid",
    ) == Path(
        "artifacts/experiments/response_grid/objects/geometric/"
        "sae=qwen-scope-qwen3-8b-l0-50/layer=18/method=seed_topk_cosine/neighbors.json"
    )
    assert done_path(acts) == acts.with_name("DONE")
    assert normalize_split(None) == "default"
    assert normalize_n(-1) == "all"


def test_artifact_path_parts_reject_nested_or_shell_like_values():
    with pytest.raises(ValueError):
        safe_path_part("../bad", field="dataset")
    with pytest.raises(ValueError):
        safe_path_part("bad value", field="dataset")
