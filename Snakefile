configfile: "configs/experiments/safety_grid.yaml"

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from sae_tools.workflow import activation_path, geometric_path, stat_metrics_path
from sae_tools.workflow.registry import ExperimentSpec, load_registry


REGISTRY = load_registry("configs/registry")
EXPERIMENT = ExperimentSpec.from_mapping(config, path="configs/experiments/safety_grid.yaml", registry=REGISTRY)


def _text(path):
    return str(path)


def _cli_n(max_samples):
    if max_samples is None or int(max_samples) <= 0:
        return -1
    return int(max_samples)


def _stat_top_k(agg, metric):
    for analysis in EXPERIMENT.statistical_analyses(REGISTRY):
        if agg in analysis.aggregations and metric in analysis.metrics:
            return analysis.top_k
    return 50


def _geo_setting(method, field):
    for analysis in EXPERIMENT.geometric_analyses(REGISTRY):
        if method in analysis.methods:
            return getattr(analysis, field)
    return 50 if field == "top_k" else 256


def _activation_input(wildcards):
    dataset = REGISTRY.dataset(wildcards.dataset)
    max_samples = EXPERIMENT.dataset_max_samples(REGISTRY, wildcards.dataset)
    return _text(
        activation_path(
            model=wildcards.model,
            sae=wildcards.sae,
            layer=int(wildcards.layer),
            dataset=wildcards.dataset,
            split=dataset.split,
            max_samples=max_samples,
        )
    )


ACTIVATION_TARGETS = [
    _text(
        activation_path(
            model=job["model"],
            sae=job["sae"],
            layer=job["layer"],
            dataset=job["dataset"],
            split=REGISTRY.dataset(job["dataset"]).split,
            max_samples=job["max_samples"],
        )
    )
    for job in EXPERIMENT.activation_jobs(REGISTRY)
]

STAT_TARGETS = [
    _text(
        stat_metrics_path(
            model=job["model"],
            sae=job["sae"],
            layer=job["layer"],
            dataset=job["dataset"],
            agg=job["agg"],
            metric=job["metric"],
        )
    )
    for job in EXPERIMENT.stat_jobs(REGISTRY)
]

GEO_TARGETS = [
    _text(
        geometric_path(
            sae=job["sae"],
            layer=job["layer"],
            method=job["method"],
        )
    )
    for job in EXPERIMENT.geometric_jobs(REGISTRY)
]


rule all:
    input:
        STAT_TARGETS + GEO_TARGETS


rule activations:
    output:
        acts="artifacts/activations/model={model}/sae={sae}/layer={layer}/dataset={dataset}/split={split}/n={n}/acts.pt"
    log:
        "logs/activations/model={model}.sae={sae}.layer={layer}.dataset={dataset}.split={split}.n={n}.log"
    params:
        max_samples=lambda w: _cli_n(w.n),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/gen_activations_one.py \
          --model {wildcards.model} \
          --sae {wildcards.sae} \
          --layer {wildcards.layer} \
          --dataset {wildcards.dataset} \
          --max-samples {params.max_samples} \
          --out {output.acts} \
          > {log} 2>&1
        """


rule stat_analysis:
    input:
        acts=_activation_input
    output:
        metrics="artifacts/analyses/stat/model={model}/sae={sae}/layer={layer}/dataset={dataset}/agg={agg}/metric={metric}/metrics.json"
    log:
        "logs/stat/model={model}.sae={sae}.layer={layer}.dataset={dataset}.agg={agg}.metric={metric}.log"
    params:
        max_samples=lambda w: _cli_n(EXPERIMENT.dataset_max_samples(REGISTRY, w.dataset)),
        top_k=lambda w: _stat_top_k(w.agg, w.metric),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/analyze_stat.py \
          --acts {input.acts} \
          --dataset {wildcards.dataset} \
          --agg {wildcards.agg} \
          --metric {wildcards.metric} \
          --max-samples {params.max_samples} \
          --top-k {params.top_k} \
          --out {output.metrics} \
          > {log} 2>&1
        """


rule geometric_analysis:
    output:
        result="artifacts/analyses/geometric/sae={sae}/layer={layer}/method={method}/{filename}"
    log:
        "logs/geometric/sae={sae}.layer={layer}.method={method}.filename={filename}.log"
    params:
        top_k=lambda w: _geo_setting(w.method, "top_k"),
        chunk_size=lambda w: _geo_setting(w.method, "chunk_size"),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/analyze_geo.py \
          --sae {wildcards.sae} \
          --layer {wildcards.layer} \
          --method {wildcards.method} \
          --top-k {params.top_k} \
          --chunk-size {params.chunk_size} \
          --out {output.result} \
          > {log} 2>&1
        """
