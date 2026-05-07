from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sae_tools.adapters.datasets import get_adapter
from sae_tools.workflow.registry import DatasetSpec, Registry, load_registry


def add_registry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry-dir", default=str(REPO_ROOT / "configs/registry"))


def load_repo_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"{name} is not set. Configure it in {REPO_ROOT / '.env'}.")
    return value


def resolve_registry(args: argparse.Namespace) -> Registry:
    return load_registry(args.registry_dir)


def resolve_max_samples(cli_value: int | None, dataset: DatasetSpec) -> int | None:
    return dataset.max_samples if cli_value is None else cli_value


def dataset_local_path(dataset: DatasetSpec, dataset_root: str | os.PathLike[str]) -> Path:
    folder = Path(dataset.folder)
    if folder.is_absolute():
        return folder
    return Path(dataset_root) / folder


def load_dataset_from_spec(
    dataset: DatasetSpec,
    *,
    dataset_root: str | os.PathLike[str],
    max_samples: int | None,
) -> Any:
    adapter = get_adapter(dataset.adapter)
    return adapter.load(
        str(dataset_local_path(dataset, dataset_root)),
        max_samples,
        split=dataset.split,
        subset=dataset.subset,
    )


def parse_max_samples(value: str) -> int:
    if value.lower() == "all":
        return -1
    return int(value)
