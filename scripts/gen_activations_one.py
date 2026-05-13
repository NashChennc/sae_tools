from __future__ import annotations

import argparse
from pathlib import Path

from workflow_common import REPO_ROOT, add_registry_args, load_repo_env, require_env, resolve_max_samples, resolve_registry

from sae_tools.adapters.datasets import get_adapter
from sae_tools.adapters.models import load_model_from_profile
from sae_tools.adapters.saes import get_sae_profile, load_sae_adapter
from sae_tools.experiment_store import ExperimentStore
from sae_tools.model import residual_post_hook_name
from sae_tools.model.run import generate_activations
from sae_tools.workflow.artifacts import (
    activation_path,
    artifact_meta_path,
    atomic_torch_save,
    build_artifact_meta,
    done_path,
    mark_done,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one deterministic SAE activation artifact.")
    add_registry_args(parser)
    parser.add_argument("--model", required=True, help="Model registry key.")
    parser.add_argument("--sae", required=True, help="SAE registry key.")
    parser.add_argument("--dataset", required=True, help="Dataset registry key.")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--experiment", default=None, help="Experiment id for default artifact paths.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_repo_env()
    registry = resolve_registry(args)
    model_spec = registry.model(args.model)
    sae_spec = registry.sae(args.sae)
    dataset_spec = registry.dataset(args.dataset)

    if sae_spec.compatible_models and args.model not in sae_spec.compatible_models:
        raise ValueError(f"Model '{args.model}' is not compatible with SAE '{args.sae}'.")

    sae_profile = get_sae_profile(args.sae)
    layer = sae_profile.default_layer if args.layer is None else args.layer
    max_samples = resolve_max_samples(args.max_samples, dataset_spec)
    output = args.out or activation_path(
        root=args.artifact_root,
        experiment=args.experiment,
        model=args.model,
        sae=args.sae,
        layer=layer,
        dataset=args.dataset,
        split=dataset_spec.split,
        max_samples=max_samples,
    )

    if output.exists() and done_path(output).exists() and not args.overwrite:
        print(f"READY {output}")
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Artifact exists without --overwrite: {output}")

    model_root = require_env("MODEL_ROOT")
    sae_root = require_env("SAE_ROOT")
    dataset_root = require_env("DATASET_ROOT")
    device = args.device or model_spec.device
    dtype = args.dtype or model_spec.dtype

    sae_path = sae_profile.layer_path(sae_root, layer)
    hook_name = residual_post_hook_name(layer)
    tokenizer, model, model_profile = load_model_from_profile(
        model_root=model_root,
        profile_name=args.model,
        device=device,
        dtype=dtype,
    )
    sae = load_sae_adapter(
        profile=sae_profile,
        sae_path=sae_path,
        model_name=model_profile.hf_name,
        layer=layer,
        hook_name=hook_name,
        device=device,
        dtype=dtype,
    )

    adapter = get_adapter(dataset_spec.adapter)
    dataset_path = Path(dataset_spec.folder)
    if not dataset_path.is_absolute():
        dataset_path = Path(dataset_root) / dataset_path
    dataset = adapter.load(
        str(dataset_path),
        max_samples,
        split=dataset_spec.split,
        subset=dataset_spec.subset,
    )

    results = generate_activations(
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        layer=layer,
        dataset=dataset,
        data_type=dataset_spec.data_type,
        batch_size=args.batch_size,
    )
    atomic_torch_save(results, output)
    meta = build_artifact_meta(
        script="gen_activations_one.py",
        repo_dir=REPO_ROOT,
        params={
            "model": args.model,
            "sae": args.sae,
            "layer": layer,
            "dataset": args.dataset,
            "adapter": dataset_spec.adapter,
            "data_type": dataset_spec.data_type,
            "split": dataset_spec.split,
            "subset": dataset_spec.subset,
            "max_samples": max_samples,
            "batch_size": args.batch_size,
            "device": device,
            "dtype": dtype,
            "output": str(output),
        },
        inputs={
            "model_path": model_spec.local_path,
            "sae_path": str(sae_path),
            "dataset_folder": dataset_spec.folder,
        },
    )
    write_json_atomic(artifact_meta_path(output), meta)
    mark_done(output)
    _record_store_output(
        output=output,
        model=args.model,
        sae=args.sae,
        layer=layer,
        dataset=args.dataset,
        split=dataset_spec.split,
        max_samples=max_samples,
    )
    print(f"DONE {output}")


def _record_store_output(
    *,
    output: Path,
    model: str,
    sae: str,
    layer: int,
    dataset: str,
    split: str | None,
    max_samples: int | None,
) -> None:
    store = ExperimentStore.from_artifact_path(output)
    if store is None:
        return
    store.ensure_initialized()
    dimensions = {
        "model": model,
        "sae": sae,
        "layer": layer,
        "dataset": dataset,
        "split": split,
        "max_samples": max_samples,
    }
    store.record_artifact_path(
        kind="activations",
        role="acts",
        path=output,
        dimensions=dimensions,
        producer="gen_activations_one.py",
        mime="application/vnd.pytorch",
    )
    store.record_artifact_path(
        kind="activations",
        role="meta",
        path=artifact_meta_path(output),
        dimensions={**dimensions, "role": "meta"},
        producer="gen_activations_one.py",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
