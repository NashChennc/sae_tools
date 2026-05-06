from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .base import TopKSAEConfig
from .profiles import SAEProfile


class QwenScopeTopKSAE:
    """Adapter for Qwen-Scope TopK SAE checkpoints."""

    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        cfg: TopKSAEConfig,
    ) -> None:
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        self.W_enc = state_dict["W_enc"].to(device=device, dtype=dtype)
        self.b_enc = state_dict["b_enc"].to(device=device, dtype=dtype)
        self.b_dec = state_dict["b_dec"].to(device=device, dtype=dtype)

        raw_w_dec = state_dict["W_dec"].to(device=device, dtype=dtype)
        self.W_dec = raw_w_dec.T.contiguous()

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        x = acts.to(device=self.W_enc.device, dtype=self.W_enc.dtype)
        if self.cfg.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_acts = torch.matmul(x, self.W_enc.T) + self.b_enc
        values, indices = torch.topk(pre_acts, k=self.cfg.top_k, dim=-1)
        values = torch.relu(values)
        encoded = torch.zeros_like(pre_acts)
        encoded.scatter_(-1, indices, values)
        return encoded


def torch_load_weights(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected a checkpoint dict in {path}, got {type(state_dict)!r}")
    return state_dict


def load_qwen_scope_topk_sae(
    sae_path: str | Path,
    profile: SAEProfile,
    layer: int,
    hook_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> QwenScopeTopKSAE:
    path = Path(sae_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Qwen-Scope SAE file not found: {path}. "
            "Run download_saes.py before activation generation."
        )

    state_dict = torch_load_weights(path)
    missing = QwenScopeTopKSAE.required_keys.difference(state_dict)
    if missing:
        raise ValueError(f"{path} is missing required Qwen-Scope keys: {sorted(missing)}")

    expected = {
        "W_enc": (profile.d_sae, profile.d_model),
        "W_dec": (profile.d_model, profile.d_sae),
        "b_enc": (profile.d_sae,),
        "b_dec": (profile.d_model,),
    }
    actual = {key: tuple(state_dict[key].shape) for key in expected}
    bad_shapes = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
    if bad_shapes:
        raise ValueError(f"Unexpected Qwen-Scope SAE shapes in {path}: {bad_shapes}")
    if not 0 <= layer < 36:
        raise ValueError(f"Qwen-Scope Qwen3-8B SAE layer must be in [0, 35], got {layer}")

    cfg = TopKSAEConfig(
        source=str(path),
        hook_layer=layer,
        hook_name=hook_name,
        d_in=profile.d_model,
        d_sae=profile.d_sae,
        top_k=profile.top_k,
        device=device,
        dtype=dtype,
    )
    return QwenScopeTopKSAE(state_dict, cfg)
