import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _notebook_text(name: str) -> str:
    notebook = json.loads((ROOT / name).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_notebooks_use_canonical_module_paths():
    text = "\n".join(
        _notebook_text(name)
        for name in ["1_statistical.ipynb", "2_dashboard.ipynb", "3_geometric.ipynb"]
    )

    for old_path in [
        "sae_tools.data_loader",
        "sae_tools.statistical",
        "sae_tools.dashboard",
        "sae_tools.geometric",
    ]:
        assert old_path not in text

    assert "sae_tools.adapters.datasets" in text
    assert "sae_tools.analysis.statistical" in text
    assert "sae_tools.analysis.dashboard.viewer" in text
    assert "sae_tools.analysis.geometric.norm" in text


def test_activation_notebooks_require_generated_artifacts():
    for name in ["1_statistical.ipynb", "2_dashboard.ipynb"]:
        text = _notebook_text(name)
        assert "build_generate_activations_command" in text
        assert "find_latest_activation_file" in text
        assert "require_activation_keys(sparse_data)" in text
        assert "load_sae_predictions_pt(PT_FILE)" in text


def test_geometric_notebook_uses_sae_profile_location():
    text = _notebook_text("3_geometric.ipynb")

    assert "get_sae_profile" in text
    assert "qwen-scope-qwen3-8b-l0-50" in text
    assert "sae_profile.layer_path(SAE_ROOT, LAYER)" in text
