from __future__ import annotations

import pandas as pd

from sae_tools.experiment_store import ExperimentStore
from sae_tools.workflow import load_registry
from sae_tools.workflow.registry import ExperimentSpec


def test_experiment_store_initializes_manifest_and_workflow_indices(tmp_path):
    registry = load_registry("configs/registry")
    experiment = ExperimentSpec.load("configs/experiments/safety_grid.yaml", registry)

    store = ExperimentStore.create(
        tmp_path / "artifacts",
        experiment=experiment,
        registry=registry,
        experiment_id="safety_grid",
    )

    assert store.manifest_path.exists()
    assert (store.path / "config.resolved.yaml").exists()
    assert (store.path / "registry.snapshot.json").exists()
    assert set(store.read_manifest().index) == {"jobs", "artifacts", "features", "summaries", "reports"}
    assert len(store.read_table("jobs")) == 8
    assert len(store.read_table("artifacts")) == 8


def test_experiment_store_records_stat_feature_index(tmp_path):
    path = (
        tmp_path
        / "artifacts/experiments/response_grid/objects/stat/model=qwen3-8b/sae=qwen-scope-qwen3-8b-l0-50/"
        "layer=18/dataset=ToxicChat_response/agg=max/feature_table.parquet"
    )
    store = ExperimentStore.from_artifact_path(path)
    assert store is not None
    store.ensure_initialized()

    table = pd.DataFrame(
        {
            "feature": [1],
            "precision": [0.8],
            "recall": [0.7],
            "f1": [0.746],
            "activation_ratio": [0.2],
            "diff": [0.5],
            "pearson": [0.3],
            "auroc": [0.9],
        }
    )
    store.replace_stat_features(
        table,
        model="qwen3-8b",
        sae="qwen-scope-qwen3-8b-l0-50",
        layer=18,
        dataset="ToxicChat_response",
        agg="max",
    )

    indexed = store.query_features(model="qwen3-8b", layer=18)
    assert indexed.iloc[0]["experiment_id"] == "response_grid"
    assert indexed.iloc[0]["feature"] == 1
