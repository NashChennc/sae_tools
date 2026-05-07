from __future__ import annotations

import argparse
from pathlib import Path

from workflow_common import add_registry_args, load_repo_env, require_env, resolve_registry


def _exists_rooted(root: str, relative: str) -> str:
    path = Path(relative)
    if not path.is_absolute():
        path = Path(root) / path
    if path.exists():
        return f"ready {path}"
    return f"missing {path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect registry entries and local resource availability.")
    add_registry_args(parser)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if backend registry validation fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_repo_env()
    registry = resolve_registry(args)
    backend_errors = registry.validate_backends()

    model_root = require_env("MODEL_ROOT")
    sae_root = require_env("SAE_ROOT")
    dataset_root = require_env("DATASET_ROOT")

    print("Models")
    for key, model in registry.models.items():
        print(f"  {key:<32} {_exists_rooted(model_root, model.local_path)}")

    print("SAEs")
    for key, sae in registry.saes.items():
        path = Path(sae.local_dir)
        if not path.is_absolute():
            path = Path(sae_root) / path
        path = path / sae.layer_filename(sae.default_layer)
        status = "ready" if path.exists() else "missing"
        print(f"  {key:<32} {status} {path}")

    print("Datasets")
    for key, dataset in registry.datasets.items():
        print(f"  {key:<32} {_exists_rooted(dataset_root, dataset.folder)}")

    if backend_errors:
        print("Backend validation")
        for error in backend_errors:
            print(f"  error {error}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
