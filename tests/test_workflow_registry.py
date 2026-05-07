from pathlib import Path

import pytest

from sae_tools.workflow import (
    activation_path,
    done_path,
    geometric_path,
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
    assert "qwen-scope-qwen3-8b-l0-50" in registry.saes
    assert registry.dataset("ToxicChat_prompt").adapter == "ToxicChat"
    assert registry.dataset("ToxicChat_prompt").max_samples == 1000
    assert registry.analysis("stat_basic").metrics == ("pearson", "auroc", "f1")
    assert registry.validate_backends() == []


def test_experiment_expands_activation_stat_and_geometric_jobs():
    registry = load_registry("configs/registry")
    experiment = ExperimentSpec.load("configs/experiments/safety_grid.yaml", registry)

    activation_jobs = experiment.activation_jobs(registry)
    stat_jobs = experiment.stat_jobs(registry)
    geo_jobs = experiment.geometric_jobs(registry)

    assert len(activation_jobs) == 2
    assert activation_jobs[0]["model"] == "qwen3-8b-guard"
    assert activation_jobs[0]["layer"] == 18
    assert activation_jobs[0]["n"] == "1000"
    assert len(stat_jobs) == 12
    assert {job["metric"] for job in stat_jobs} == {"pearson", "auroc", "f1"}
    assert {job["method"] for job in geo_jobs} == {"norm", "topk_cosine"}


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
        "artifacts/activations/model=qwen3-8b-guard/sae=qwen-scope-qwen3-8b-l0-50/"
        "layer=18/dataset=ToxicChat_prompt/split=default/n=1000/acts.pt"
    )
    assert stat_metrics_path(
        model="qwen3-8b-guard",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_prompt",
        agg="max",
        metric="auroc",
    ) == Path(
        "artifacts/analyses/stat/model=qwen3-8b-guard/sae=qwen-scope-qwen3-8b-l0-50/"
        "layer=18/dataset=ToxicChat_prompt/agg=max/metric=auroc/metrics.json"
    )
    assert geometric_path(sae="qwen-scope-qwen3-8b-l0-50", layer=18, method="topk_cosine").name == "neighbors.json"
    assert done_path(acts) == acts.with_name("DONE")
    assert normalize_split(None) == "default"
    assert normalize_n(-1) == "all"


def test_artifact_path_parts_reject_nested_or_shell_like_values():
    with pytest.raises(ValueError):
        safe_path_part("../bad", field="dataset")
    with pytest.raises(ValueError):
        safe_path_part("bad value", field="dataset")
