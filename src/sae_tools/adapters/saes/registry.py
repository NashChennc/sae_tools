from __future__ import annotations

from pathlib import Path

from .adamkarvonen import load_adamkarvonen_sae
from .base import SAEAdapter
from .profiles import SAEProfile
from .qwen_scope import load_qwen_scope_topk_sae


def load_sae_adapter(
    profile: SAEProfile,
    sae_path: str | Path,
    model_name: str,
    layer: int,
    hook_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> SAEAdapter:
    if profile.adapter == "qwen_scope_topk":
        return load_qwen_scope_topk_sae(
            sae_path=sae_path,
            profile=profile,
            layer=layer,
            hook_name=hook_name,
            device=device,
            dtype=dtype,
        )
    if profile.adapter == "adamkarvonen":
        return load_adamkarvonen_sae(
            profile=profile,
            model_name=model_name,
            sae_path=sae_path,
            layer=layer,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported SAE adapter '{profile.adapter}' for profile '{profile.name}'")
