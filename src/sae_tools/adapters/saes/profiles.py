from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SAEProfile:
    name: str
    repo_id: str
    local_dir: str
    adapter: str
    default_layer: int
    top_k: int
    d_model: int
    d_sae: int
    description: str
    file_template: str = "layer{layer}.sae.pt"

    def layer_filename(self, layer: int) -> str:
        return self.file_template.format(layer=layer)

    def layer_path(self, sae_root: str | os.PathLike[str], layer: int | None = None) -> Path:
        layer = self.default_layer if layer is None else layer
        return Path(sae_root) / self.local_dir / self.layer_filename(layer)


ADAMKARVONEN_PROFILE = SAEProfile(
    name="adamkarvonen",
    repo_id="adamkarvonen/qwen3-8b-saes",
    local_dir="adamkarvonen/qwen3-8b-saes/saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2",
    adapter="adamkarvonen",
    default_layer=18,
    top_k=80,
    d_model=4096,
    d_sae=65536,
    description="Legacy Adam Karvonen BatchTopK checkpoint loaded through the original JumpReLU conversion logic.",
    file_template="ae.pt",
)


SAE_PROFILES = {
    "qwen-scope-qwen3-8b-l0-50": SAEProfile(
        name="qwen-scope-qwen3-8b-l0-50",
        repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        local_dir="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        adapter="qwen_scope_topk",
        default_layer=18,
        top_k=50,
        d_model=4096,
        d_sae=65536,
        description="Qwen-Scope residual-stream TopK SAE for Qwen3-8B Base.",
    ),
    "qwen-scope-qwen3-8b-l0-100": SAEProfile(
        name="qwen-scope-qwen3-8b-l0-100",
        repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        local_dir="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        adapter="qwen_scope_topk",
        default_layer=18,
        top_k=100,
        d_model=4096,
        d_sae=65536,
        description="Qwen-Scope residual-stream TopK SAE L0_100 for Qwen3-8B Base.",
    ),
    "adamkarvonen": ADAMKARVONEN_PROFILE,
    "adamkarvonen-qwen3-8b-batch-topk": ADAMKARVONEN_PROFILE,
}


def get_sae_profile(name: str) -> SAEProfile:
    try:
        return SAE_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SAE_PROFILES))
        raise ValueError(f"Unknown SAE profile '{name}'. Known profiles: {known}") from exc
