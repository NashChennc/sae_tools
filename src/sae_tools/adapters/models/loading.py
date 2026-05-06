from __future__ import annotations

import os
from pathlib import Path

from .profiles import ModelProfile, get_model_profile
from sae_tools.model.load_model import load_transformer_bridge_offline


def model_path_from_profile(model_root: str | os.PathLike[str], profile: ModelProfile) -> Path:
    return Path(model_root) / profile.local_path


def load_model_from_profile(
    model_root: str | os.PathLike[str],
    profile_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    profile = get_model_profile(profile_name)
    tokenizer, model = load_transformer_bridge_offline(
        model_name=profile.hf_name,
        model_path=str(model_path_from_profile(model_root, profile)),
        device=device,
        dtype=dtype,
    )
    return tokenizer, model, profile
