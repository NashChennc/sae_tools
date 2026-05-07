from pathlib import Path
import importlib

import pytest
import torch

from sae_tools.adapters.datasets import ADAPTER_REGISTRY, get_adapter
from sae_tools.adapters.models import get_model_profile, model_path_from_profile
from sae_tools.adapters.saes import get_sae_profile
from sae_tools.adapters.saes.adamkarvonen import load_adamkarvonen_sae
from sae_tools.adapters.saes.qwen_scope import QwenScopeTopKSAE
from sae_tools.adapters.saes.base import TopKSAEConfig
from sae_tools.model import residual_post_hook_name
from sae_tools.model.load_model import load_decoder_matrix


def test_profiles_resolve_stable_paths():
    model_profile = get_model_profile("qwen3-8b-guard")
    assert model_profile.local_path == "Qwen/Qwen3Guard-Gen-8B"
    assert model_path_from_profile("/models", model_profile) == Path("/models/Qwen/Qwen3Guard-Gen-8B")
    assert get_model_profile("qwen3-8b").local_path == "Qwen/Qwen3-8B"
    assert get_model_profile("qwen3-8b-base").name == "qwen3-8b"

    qwen_sae = get_sae_profile("qwen-scope-qwen3-8b-l0-50")
    assert qwen_sae.layer_filename(18) == "layer18.sae.pt"
    assert qwen_sae.layer_path("/root", 18) == Path(
        "/root/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50/layer18.sae.pt"
    )
    qwen_l0_100 = get_sae_profile("qwen-scope-qwen3-8b-l0-100")
    assert qwen_l0_100.repo_id == "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"
    assert qwen_l0_100.top_k == 100
    assert qwen_l0_100.layer_path("/root", 24) == Path(
        "/root/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100/layer24.sae.pt"
    )

    adam = get_sae_profile("adamkarvonen")
    assert adam.adapter == "adamkarvonen"
    assert adam.layer_filename(18) == "ae.pt"


def test_dataset_adapters_are_exposed_from_canonical_module():
    assert "ToxicChat" in ADAPTER_REGISTRY
    assert get_adapter("ToxicChat").__class__.__name__ == "ToxicChatAdapter"


def test_old_model_adapter_modules_are_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sae_tools.model.profiles")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sae_tools.model.sae_adapters")

    import sae_tools.model as model_module

    assert not hasattr(model_module, "get_model_profile")
    assert not hasattr(model_module, "get_sae_profile")
    assert not hasattr(model_module, "load_sae_adapter")


def test_old_dataset_adapter_entrypoint_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sae_tools.data_loader")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sae_tools.data_loader.adapters.registry")


def test_residual_post_hook_name():
    assert residual_post_hook_name(18) == "blocks.18.hook_out"
    with pytest.raises(ValueError):
        residual_post_hook_name(-1)


def test_qwen_scope_topk_encode_shape_and_sparsity():
    d_model = 4
    d_sae = 8
    top_k = 3
    state_dict = {
        "W_enc": torch.eye(d_sae, d_model),
        "W_dec": torch.zeros(d_model, d_sae),
        "b_enc": torch.zeros(d_sae),
        "b_dec": torch.zeros(d_model),
    }
    cfg = TopKSAEConfig(
        source="fake",
        hook_layer=0,
        hook_name="blocks.0.hook_out",
        d_in=d_model,
        d_sae=d_sae,
        top_k=top_k,
        device="cpu",
        dtype="float32",
    )
    sae = QwenScopeTopKSAE(state_dict, cfg)

    acts = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    encoded = sae.encode(acts)

    assert encoded.shape == (1, 1, d_sae)
    assert torch.count_nonzero(encoded).item() == top_k
    assert encoded[0, 0, 0].item() == 4.0
    assert encoded[0, 0, 1].item() == 3.0
    assert encoded[0, 0, 2].item() == 2.0
    assert encoded[0, 0, 3].item() == 0.0


def test_decoder_matrix_loader_normalizes_qwen_scope_layout(tmp_path):
    checkpoint = tmp_path / "layer18.sae.pt"
    torch.save({"W_dec": torch.zeros(4, 8)}, checkpoint)

    w_dec = load_decoder_matrix(checkpoint)

    assert w_dec.shape == (8, 4)


def test_adamkarvonen_adapter_calls_original_loader(monkeypatch, tmp_path):
    profile = get_sae_profile("adamkarvonen")
    calls = {}

    def fake_loader(**kwargs):
        calls.update(kwargs)
        return "loaded"

    import sae_tools.model.load_model as load_model

    monkeypatch.setattr(load_model, "load_custom_batch_topk_as_jumprelu", fake_loader)
    loaded = load_adamkarvonen_sae(
        profile=profile,
        model_name="Qwen/Qwen3-8B",
        sae_path=tmp_path,
        layer=18,
        device="cpu",
        dtype="float32",
    )

    assert loaded == "loaded"
    assert calls["sae_id"] == "adamkarvonen/qwen3-8b-saes"
    assert calls["sae_path"] == str(tmp_path / "ae.pt")
    assert calls["layer"] == 18
