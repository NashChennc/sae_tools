import importlib

import pytest

from sae_tools.analysis.artifacts import (
    build_generate_activations_command,
    find_latest_activation_file,
    require_activation_file,
    require_activation_keys,
)
from sae_tools.analysis.category import build_category_labels, parse_category_field


def test_category_analysis_parses_supported_formats():
    assert parse_category_field('{"harm:A": 1.0, "meta:x": 0.5}') == ["harm:A", "meta:x"]
    assert parse_category_field({"harm:A": 1.0}) == ["harm:A"]
    assert parse_category_field("harm:B, meta:y") == ["harm:B", "meta:y"]
    assert parse_category_field("{}") == []


def test_category_analysis_builds_binary_labels():
    rows = [
        {"category": '{"harm:A": 1.0, "meta:x": 0.5}'},
        {"category": {"harm:A": 1.0}},
        {"category": "harm:B, meta:y"},
        {"category": "{}"},
    ]

    labels, counts = build_category_labels(rows, min_samples=2, verbose=False)

    assert counts["harm:A"] == 2
    assert counts["harm:B"] == 1
    assert set(labels) == {"harm:A"}
    assert labels["harm:A"].tolist() == [1, 1, 0, 0]


def test_analysis_modules_are_exposed_from_canonical_namespace():
    assert importlib.import_module("sae_tools.analysis.statistical")
    assert importlib.import_module("sae_tools.analysis.geometric.norm")
    assert importlib.import_module("sae_tools.analysis.geometric.umap")
    assert importlib.import_module("sae_tools.analysis.dashboard.viewer")


def test_activation_artifact_check_points_to_generation_step(tmp_path):
    command = build_generate_activations_command(
        dataset_config="configs/datasets/datasets_prompt.yaml",
        dataset_name="ToxicChat",
        output_dir=tmp_path,
        model_profile="qwen3-8b-guard",
        sae_profile="qwen-scope-qwen3-8b-l0-50",
        layer=18,
    )

    with pytest.raises(FileNotFoundError, match="Generate activations before analysis"):
        find_latest_activation_file(
            results_dir=tmp_path,
            dataset_name="ToxicChat",
            model_profile="qwen3-8b-guard",
            sae_profile="qwen-scope-qwen3-8b-l0-50",
            layer=18,
            generate_command=command,
        )

    activation_file = (
        tmp_path
        / "SAE_qwen3-8b-guard_qwen-scope-qwen3-8b-l0-50_L18_20260506_1200"
        / "predictions"
        / "ToxicChat.pt"
    )
    activation_file.parent.mkdir(parents=True)
    activation_file.write_bytes(b"placeholder")

    assert require_activation_file(activation_file) == activation_file
    assert (
        find_latest_activation_file(
            results_dir=tmp_path,
            dataset_name="ToxicChat",
            model_profile="qwen3-8b-guard",
            sae_profile="qwen-scope-qwen3-8b-l0-50",
            layer=18,
        )
        == activation_file
    )


def test_activation_schema_check_requires_generate_activations_keys():
    require_activation_keys({"sparse_acts": object(), "valid_token_idx": object(), "seq_lens": object()})
    with pytest.raises(ValueError, match="missing required keys"):
        require_activation_keys({"sparse_acts": object()})


def test_old_analysis_entrypoints_are_removed():
    for module_name in [
        "sae_tools.statistical",
        "sae_tools.geometric",
        "sae_tools.dashboard",
        "sae_tools.data_loader.category",
    ]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
