from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .artifacts import normalize_n, normalize_split, safe_path_part


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' section to be a mapping.")
    return value


def _as_tuple(value: Any, *, default: Iterable[Any] = ()) -> tuple[Any, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_name: str
    local_path: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    available: str = "auto"
    description: str = ""

    @classmethod
    def from_mapping(cls, key: str, data: dict[str, Any]) -> "ModelSpec":
        safe_path_part(key, field="model")
        hf_name = data.get("hf_name") or data.get("model_name")
        local_path = data.get("local_path") or data.get("model_path")
        if not hf_name or not local_path:
            raise ValueError(f"Model '{key}' requires hf_name/model_name and local_path/model_path.")
        return cls(
            key=key,
            hf_name=str(hf_name),
            local_path=str(local_path),
            dtype=str(data.get("dtype", "bfloat16")),
            device=str(data.get("device", "cuda")),
            available=str(data.get("available", "auto")),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class SAESpec:
    key: str
    repo_id: str
    local_dir: str
    adapter: str
    default_layer: int
    layers: tuple[int, ...]
    top_k: int | None = None
    d_model: int | None = None
    d_sae: int | None = None
    compatible_models: tuple[str, ...] = ()
    file_template: str = "layer{layer}.sae.pt"
    description: str = ""

    @classmethod
    def from_mapping(cls, key: str, data: dict[str, Any]) -> "SAESpec":
        safe_path_part(key, field="sae")
        default_layer = int(data.get("default_layer", data.get("layer", 0)))
        layers = tuple(int(item) for item in _as_tuple(data.get("layers"), default=(default_layer,)))
        return cls(
            key=key,
            repo_id=str(data["repo_id"]),
            local_dir=str(data["local_dir"]),
            adapter=str(data["adapter"]),
            default_layer=default_layer,
            layers=layers,
            top_k=None if data.get("top_k") is None else int(data["top_k"]),
            d_model=None if data.get("d_model") is None else int(data["d_model"]),
            d_sae=None if data.get("d_sae") is None else int(data["d_sae"]),
            compatible_models=tuple(str(item) for item in _as_tuple(data.get("compatible_models"))),
            file_template=str(data.get("file_template", "layer{layer}.sae.pt")),
            description=str(data.get("description", "")),
        )

    def layer_filename(self, layer: int | None = None) -> str:
        layer = self.default_layer if layer is None else int(layer)
        return self.file_template.format(layer=layer)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    adapter: str
    folder: str
    data_type: str = "prompt"
    split: str | None = None
    subset: str | None = None
    max_samples: int | None = None
    label_field_override: str | None = None

    @classmethod
    def from_mapping(
        cls,
        key: str,
        data: dict[str, Any],
        *,
        default_max_samples: int | None = None,
    ) -> "DatasetSpec":
        safe_path_part(key, field="dataset")
        data_type = str(data.get("data_type", data.get("type", "prompt")))
        if data_type not in {"prompt", "response"}:
            raise ValueError(f"Dataset '{key}' has unsupported data_type: {data_type}")
        max_samples = data.get("max_samples", default_max_samples)
        if max_samples is not None:
            max_samples = int(max_samples)
        return cls(
            key=key,
            adapter=str(data.get("adapter", data.get("name", key))),
            folder=str(data["folder"]),
            data_type=data_type,
            split=None if data.get("split") is None else str(data.get("split")),
            subset=None if data.get("subset") is None else str(data.get("subset")),
            max_samples=max_samples,
            label_field_override=None if data.get("label_field") is None else str(data.get("label_field")),
        )

    @property
    def label_field(self) -> str:
        if self.label_field_override:
            return self.label_field_override
        return f"{self.data_type}_label"

    @property
    def split_part(self) -> str:
        return normalize_split(self.split)

    def n_part(self, override: int | None = None) -> str:
        return normalize_n(self.max_samples if override is None else override)


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    kind: str
    aggregations: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    top_k: int = 50
    seed_limit: int = 200
    chunk_size: int = 256

    @classmethod
    def from_mapping(cls, key: str, data: dict[str, Any]) -> "AnalysisSpec":
        safe_path_part(key, field="analysis")
        kind = str(data["kind"])
        if kind not in {"statistical", "geometric"}:
            raise ValueError(f"Analysis '{key}' has unsupported kind: {kind}")
        return cls(
            key=key,
            kind=kind,
            aggregations=tuple(str(item) for item in _as_tuple(data.get("aggregations"))),
            metrics=tuple(str(item) for item in _as_tuple(data.get("metrics"))),
            methods=tuple(str(item) for item in _as_tuple(data.get("methods"))),
            top_k=int(data.get("top_k", 50)),
            seed_limit=int(data.get("seed_limit", 200)),
            chunk_size=int(data.get("chunk_size", 256)),
        )


@dataclass(frozen=True)
class Registry:
    directory: Path
    models: dict[str, ModelSpec]
    saes: dict[str, SAESpec]
    datasets: dict[str, DatasetSpec]
    analyses: dict[str, AnalysisSpec]

    @classmethod
    def load(cls, directory: str | Path = "configs/registry") -> "Registry":
        directory = Path(directory)
        model_data = _section(_read_yaml(directory / "models.yaml"), "models")
        sae_data = _section(_read_yaml(directory / "saes.yaml"), "saes")
        dataset_file = _read_yaml(directory / "datasets.yaml")
        analysis_data = _section(_read_yaml(directory / "analyses.yaml"), "analyses")

        models = {key: ModelSpec.from_mapping(key, value) for key, value in model_data.items()}

        expanded_sae_data: dict[str, dict[str, Any]] = {}
        for key, value in sae_data.items():
            if "alias_for" in value:
                target = str(value["alias_for"])
                if target not in sae_data:
                    raise ValueError(f"SAE alias '{key}' points to unknown SAE '{target}'.")
                aliased = dict(sae_data[target])
                aliased.update({k: v for k, v in value.items() if k != "alias_for"})
                expanded_sae_data[key] = aliased
            else:
                expanded_sae_data[key] = value
        saes = {key: SAESpec.from_mapping(key, value) for key, value in expanded_sae_data.items()}

        dataset_defaults = dataset_file.get("defaults", {}) or {}
        default_max_samples = dataset_defaults.get("max_samples")
        dataset_data = _section(dataset_file, "datasets")
        datasets = {
            key: DatasetSpec.from_mapping(key, value, default_max_samples=default_max_samples)
            for key, value in dataset_data.items()
        }
        analyses = {key: AnalysisSpec.from_mapping(key, value) for key, value in analysis_data.items()}
        registry = cls(directory=directory, models=models, saes=saes, datasets=datasets, analyses=analyses)
        registry.validate_references()
        return registry

    def validate_references(self) -> None:
        for sae in self.saes.values():
            for model_key in sae.compatible_models:
                if model_key not in self.models:
                    raise ValueError(f"SAE '{sae.key}' references unknown model '{model_key}'.")

    def validate_backends(self) -> list[str]:
        errors: list[str] = []
        from sae_tools.adapters.datasets import get_adapter
        from sae_tools.adapters.models import get_model_profile
        from sae_tools.adapters.saes import get_sae_profile

        for key in self.models:
            try:
                get_model_profile(key)
            except Exception as exc:
                errors.append(f"model {key}: {exc}")
        for key in self.saes:
            try:
                get_sae_profile(key)
            except Exception as exc:
                errors.append(f"sae {key}: {exc}")
        for dataset in self.datasets.values():
            try:
                get_adapter(dataset.adapter)
            except Exception as exc:
                errors.append(f"dataset {dataset.key}: {exc}")
        return errors

    def model(self, key: str) -> ModelSpec:
        try:
            return self.models[key]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{key}'. Known models: {sorted(self.models)}") from exc

    def sae(self, key: str) -> SAESpec:
        try:
            return self.saes[key]
        except KeyError as exc:
            raise KeyError(f"Unknown SAE '{key}'. Known SAEs: {sorted(self.saes)}") from exc

    def dataset(self, key: str) -> DatasetSpec:
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset '{key}'. Known datasets: {sorted(self.datasets)}") from exc

    def analysis(self, key: str) -> AnalysisSpec:
        try:
            return self.analyses[key]
        except KeyError as exc:
            raise KeyError(f"Unknown analysis '{key}'. Known analyses: {sorted(self.analyses)}") from exc


@dataclass(frozen=True)
class ExperimentDataset:
    key: str
    max_samples: int | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    models: tuple[str, ...]
    saes: tuple[str, ...]
    layers: tuple[int, ...] | None
    datasets: tuple[ExperimentDataset, ...]
    analyses: tuple[str, ...]
    activation_overwrite: bool = False
    activation_batch_size: int = 1

    @classmethod
    def load(cls, path: str | Path, registry: Registry) -> "ExperimentSpec":
        path = Path(path)
        return cls.from_mapping(_read_yaml(path), path=path, registry=registry)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        path: str | Path = "<memory>",
        registry: Registry,
    ) -> "ExperimentSpec":
        path = Path(path)
        datasets = tuple(_parse_experiment_dataset(item) for item in data.get("datasets", ()))
        activation = data.get("activation") or {}
        spec = cls(
            path=path,
            models=tuple(str(item) for item in data.get("models", ())),
            saes=tuple(str(item) for item in data.get("saes", ())),
            layers=None if data.get("layers") is None else tuple(int(item) for item in _as_tuple(data.get("layers"))),
            datasets=datasets,
            analyses=tuple(str(item) for item in data.get("analyses", ())),
            activation_overwrite=bool(activation.get("overwrite", False)),
            activation_batch_size=int(activation.get("batch_size", 1)),
        )
        spec.validate(registry)
        return spec

    def validate(self, registry: Registry) -> None:
        for key in self.models:
            registry.model(key)
        for key in self.saes:
            registry.sae(key)
        for item in self.datasets:
            registry.dataset(item.key)
        for key in self.analyses:
            registry.analysis(key)
        for model_key in self.models:
            for sae_key in self.saes:
                compatible = registry.sae(sae_key).compatible_models
                if compatible and model_key not in compatible:
                    raise ValueError(f"Model '{model_key}' is not compatible with SAE '{sae_key}'.")
        if self.layers is not None:
            for sae_key in self.saes:
                sae = registry.sae(sae_key)
                unsupported = sorted(set(self.layers).difference(sae.layers))
                if unsupported:
                    raise ValueError(
                        f"Experiment '{self.path}' requests unsupported layers for SAE '{sae_key}': {unsupported}. "
                        f"Supported layers: {list(sae.layers)}"
                    )

    def dataset_max_samples(self, registry: Registry, dataset_key: str) -> int | None:
        for item in self.datasets:
            if item.key == dataset_key and item.max_samples is not None:
                return item.max_samples
        return registry.dataset(dataset_key).max_samples

    def statistical_analyses(self, registry: Registry) -> tuple[AnalysisSpec, ...]:
        return tuple(registry.analysis(key) for key in self.analyses if registry.analysis(key).kind == "statistical")

    def geometric_analyses(self, registry: Registry) -> tuple[AnalysisSpec, ...]:
        return tuple(registry.analysis(key) for key in self.analyses if registry.analysis(key).kind == "geometric")

    def layers_for_sae(self, registry: Registry, sae_key: str) -> tuple[int, ...]:
        sae = registry.sae(sae_key)
        return self.layers if self.layers is not None else (sae.default_layer,)

    def activation_jobs(self, registry: Registry) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for model_key in self.models:
            for sae_key in self.saes:
                for layer in self.layers_for_sae(registry, sae_key):
                    for dataset_item in self.datasets:
                        dataset = registry.dataset(dataset_item.key)
                        max_samples = self.dataset_max_samples(registry, dataset.key)
                        jobs.append(
                            {
                                "model": model_key,
                                "sae": sae_key,
                                "layer": layer,
                                "dataset": dataset.key,
                                "split": dataset.split_part,
                                "n": normalize_n(max_samples),
                                "max_samples": max_samples,
                                "batch_size": self.activation_batch_size,
                            }
                        )
        return jobs

    def stat_jobs(self, registry: Registry) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for base in self.activation_jobs(registry):
            for analysis in self.statistical_analyses(registry):
                for agg in analysis.aggregations:
                    for metric in analysis.metrics:
                        jobs.append({**base, "agg": agg, "metric": metric, "top_k": analysis.top_k})
        return jobs

    def stat_batch_jobs(self, registry: Registry) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for base in self.activation_jobs(registry):
            for analysis in self.statistical_analyses(registry):
                for agg in analysis.aggregations:
                    jobs.append({**base, "agg": agg, "metrics": analysis.metrics, "top_k": analysis.top_k})
        return jobs

    def geometric_jobs(self, registry: Registry) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for sae_key in self.saes:
            for layer in self.layers_for_sae(registry, sae_key):
                for analysis in self.geometric_analyses(registry):
                    for method in analysis.methods:
                        jobs.append(
                            {
                                "sae": sae_key,
                                "layer": layer,
                                "method": method,
                                "top_k": analysis.top_k,
                                "seed_limit": analysis.seed_limit,
                                "chunk_size": analysis.chunk_size,
                            }
                        )
        return jobs


def _parse_experiment_dataset(value: Any) -> ExperimentDataset:
    if isinstance(value, str):
        return ExperimentDataset(key=value)
    if isinstance(value, dict):
        key = str(value.get("id", value.get("dataset", value.get("key", ""))))
        if not key:
            raise ValueError(f"Experiment dataset entry requires id/dataset/key: {value}")
        max_samples = value.get("max_samples")
        return ExperimentDataset(key=key, max_samples=None if max_samples is None else int(max_samples))
    raise ValueError(f"Unsupported experiment dataset entry: {value!r}")


def load_registry(directory: str | Path = "configs/registry") -> Registry:
    return Registry.load(directory)


def load_experiment(path: str | Path, registry: Registry | None = None) -> ExperimentSpec:
    registry = load_registry() if registry is None else registry
    return ExperimentSpec.load(path, registry)
