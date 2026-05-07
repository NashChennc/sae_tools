from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable

from sae_tools.workflow.artifacts import activation_path


def activation_run_pattern(model_profile: str, sae_profile: str, layer: int) -> str:
    """Return the run directory glob used by 0_generate_activations.py."""
    return f"SAE_{model_profile}_{sae_profile}_L{layer}_*"


def build_generate_activations_command(
    *,
    dataset_config: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    model_profile: str,
    sae_profile: str,
    layer: int,
    max_samples: int | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> str:
    """Build a copy-pasteable single-artifact activation generation command."""
    parts = [
        "python",
        "scripts/gen_activations_one.py",
        "--model",
        model_profile,
        "--sae",
        sae_profile,
        "--layer",
        str(layer),
        "--dataset",
        dataset_name,
        "--out",
        str(
            activation_path(
                root=output_dir,
                model=model_profile,
                sae=sae_profile,
                layer=layer,
                dataset=dataset_name,
                max_samples=max_samples,
            )
        ),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if max_samples is not None:
        parts.extend(["--max-samples", str(max_samples)])
    del dataset_config
    return " ".join(shlex.quote(part) for part in parts)


def require_activation_file(
    path: str | Path,
    *,
    generate_command: str | None = None,
) -> Path:
    """Return an activation file path or raise a generation-oriented error."""
    activation_path = Path(path).expanduser()
    if activation_path.exists():
        return activation_path

    message = f"Activation file not found: {activation_path}"
    if generate_command:
        message += f"\nGenerate activations before analysis:\n  {generate_command}"
    raise FileNotFoundError(message)


def find_latest_activation_file(
    *,
    results_dir: str | Path,
    dataset_name: str,
    model_profile: str,
    sae_profile: str,
    layer: int,
    predictions_dir: str = "predictions",
    generate_command: str | None = None,
) -> Path:
    """Find a deterministic activation file, with legacy timestamp fallback."""
    base = Path(results_dir).expanduser()
    deterministic = activation_path(
        root=base,
        model=model_profile,
        sae=sae_profile,
        layer=layer,
        dataset=dataset_name,
        max_samples=None,
    )
    if deterministic.exists():
        return deterministic

    deterministic_candidates = list(
        (
            base
            / "activations"
            / f"model={model_profile}"
            / f"sae={sae_profile}"
            / f"layer={layer}"
            / f"dataset={dataset_name}"
        ).glob("split=*/n=*/acts.pt")
    )
    if deterministic_candidates:
        return max(deterministic_candidates, key=lambda path: path.stat().st_mtime)

    pattern = activation_run_pattern(model_profile, sae_profile, layer)
    candidates = list(base.glob(f"{pattern}/{predictions_dir}/{dataset_name}.pt"))
    if not candidates:
        expected = base / pattern / predictions_dir / f"{dataset_name}.pt"
        message = f"Activation file not found for {dataset_name}: {expected}"
        if generate_command:
            message += f"\nGenerate activations before analysis:\n  {generate_command}"
        raise FileNotFoundError(message)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def require_activation_keys(sparse_data: dict, required_keys: Iterable[str] | None = None) -> None:
    """Validate that loaded activation data has the expected generate_activations schema."""
    keys = tuple(required_keys or ("sparse_acts", "valid_token_idx", "seq_lens"))
    missing = [key for key in keys if key not in sparse_data]
    if missing:
        raise ValueError(f"Activation data is missing required keys: {missing}")
