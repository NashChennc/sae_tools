from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hf_name: str
    local_path: str
    description: str


MODEL_PROFILES = {
    "qwen3-8b-guard": ModelProfile(
        name="qwen3-8b-guard",
        hf_name="Qwen/Qwen3-8B",
        local_path="Qwen/Qwen3Guard-Gen-8B",
        description="Local Qwen3Guard-Gen-8B checkpoint analyzed with base Qwen3 SAE features.",
    ),
    "qwen3-8b": ModelProfile(
        name="qwen3-8b",
        hf_name="Qwen/Qwen3-8B",
        local_path="Qwen/Qwen3-8B",
        description="Qwen3-8B checkpoint matching the Qwen-Scope SAE training target.",
    ),
}

def get_model_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown model profile '{name}'. Known profiles: {known}") from exc
