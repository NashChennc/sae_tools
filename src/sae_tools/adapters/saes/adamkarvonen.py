from __future__ import annotations

from pathlib import Path

from .profiles import SAEProfile


def load_adamkarvonen_sae(
    profile: SAEProfile,
    model_name: str,
    sae_path: str | Path,
    layer: int,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    path = Path(sae_path)
    if path.is_dir():
        path = path / profile.layer_filename(layer)

    # The Adam Karvonen path intentionally keeps the original conversion
    # function as the source of truth. This adapter only normalizes routing.
    from sae_tools.model.load_model import load_custom_batch_topk_as_jumprelu

    return load_custom_batch_topk_as_jumprelu(
        model_name=model_name,
        sae_id=profile.repo_id,
        sae_path=str(path),
        layer=layer,
        device=device,
        dtype=dtype,
    )
