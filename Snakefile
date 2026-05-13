EXPERIMENT_CONFIG = config.get("experiment_config", "configs/experiments/safety_grid.yaml")
configfile: EXPERIMENT_CONFIG

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from sae_tools.workflow import activation_path, geometric_path, geometric_seed_path, stat_analysis_dir, stat_metrics_path
from sae_tools.workflow.registry import ExperimentSpec, load_registry


REGISTRY = load_registry("configs/registry")
EXPERIMENT = ExperimentSpec.from_mapping(config, path=EXPERIMENT_CONFIG, registry=REGISTRY)
EXPERIMENT_NAME = Path(EXPERIMENT_CONFIG).stem


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
            experiment=wildcards.experiment,
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
            experiment=EXPERIMENT_NAME,
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
            experiment=EXPERIMENT_NAME,
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

STAT_BATCH_JOBS = EXPERIMENT.stat_batch_jobs(REGISTRY)

STAT_BATCH_TARGETS = [
    _text(
        stat_analysis_dir(
            experiment=EXPERIMENT_NAME,
            model=job["model"],
            sae=job["sae"],
            layer=job["layer"],
            dataset=job["dataset"],
            agg=job["agg"],
        )
        / "DONE"
    )
    for job in STAT_BATCH_JOBS
]

GEO_TARGETS = [
    _text(
        geometric_path(
            experiment=EXPERIMENT_NAME,
            sae=job["sae"],
            layer=job["layer"],
            method=job["method"],
        )
    )
    for job in EXPERIMENT.geometric_jobs(REGISTRY)
]


rule all:
    input:
        STAT_BATCH_TARGETS + GEO_TARGETS


rule activations:
    output:
        acts=protected("artifacts/experiments/{experiment}/objects/activations/model={model}/sae={sae}/layer={layer}/dataset={dataset}/split={split}/n={n}/acts.pt")
    log:
        "logs/activations/experiment={experiment}.model={model}.sae={sae}.layer={layer}.dataset={dataset}.split={split}.n={n}.log"
    params:
        max_samples=lambda w: _cli_n(w.n),
        batch_size=lambda w: EXPERIMENT.activation_batch_size,
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/gen_activations_one.py \
          --model {wildcards.model} \
          --sae {wildcards.sae} \
          --layer {wildcards.layer} \
          --dataset {wildcards.dataset} \
          --max-samples {params.max_samples} \
          --batch-size {params.batch_size} \
          --out {output.acts} \
          > {log} 2>&1
        """


rule stat_analysis_batch:
    input:
        acts=_activation_input
    output:
        done="artifacts/experiments/{experiment}/objects/stat/model={model}/sae={sae}/layer={layer}/dataset={dataset}/agg={agg}/DONE"
    log:
        "logs/stat_batch/experiment={experiment}.model={model}.sae={sae}.layer={layer}.dataset={dataset}.agg={agg}.log"
    params:
        max_samples=lambda w: _cli_n(EXPERIMENT.dataset_max_samples(REGISTRY, w.dataset)),
        metrics=lambda w: ",".join(_stat_metrics(w.agg)),
        top_k=lambda w: _stat_top_k(w.agg, "f1"),
        out_dir=lambda w: _text(
            stat_analysis_dir(
                experiment=w.experiment,
                model=w.model,
                sae=w.sae,
                layer=int(w.layer),
                dataset=w.dataset,
                agg=w.agg,
            )
        ),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/analyze_stat.py \
          --acts {input.acts} \
          --dataset {wildcards.dataset} \
          --agg {wildcards.agg} \
          --metrics {params.metrics} \
          --max-samples {params.max_samples} \
          --top-k {params.top_k} \
          --out-dir {params.out_dir} \
          > {log} 2>&1
        """


rule stat_analysis:
    input:
        acts=_activation_input
    output:
        metrics="artifacts/experiments/{experiment}/objects/stat/model={model}/sae={sae}/layer={layer}/dataset={dataset}/agg={agg}/metric={metric}/metrics.json"
    log:
        "logs/stat/experiment={experiment}.model={model}.sae={sae}.layer={layer}.dataset={dataset}.agg={agg}.metric={metric}.log"
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


def _stat_metrics(agg):
    metrics = []
    for analysis in EXPERIMENT.statistical_analyses(REGISTRY):
        if agg in analysis.aggregations:
            metrics.extend(analysis.metrics)
    return tuple(dict.fromkeys(metrics))


def _stat_batch_jobs_for_geo(wildcards):
    return [
        job
        for job in STAT_BATCH_JOBS
        if job["sae"] == wildcards.sae and int(job["layer"]) == int(wildcards.layer)
    ]


def _stat_done_inputs_for_geo(wildcards):
    return [
        _text(
            stat_analysis_dir(
                experiment=wildcards.experiment,
                model=job["model"],
                sae=job["sae"],
                layer=job["layer"],
                dataset=job["dataset"],
                agg=job["agg"],
            )
            / "DONE"
        )
        for job in _stat_batch_jobs_for_geo(wildcards)
    ]


def _stat_dirs_for_geo(wildcards):
    return " ".join(
        _text(
            stat_analysis_dir(
                experiment=wildcards.experiment,
                model=job["model"],
                sae=job["sae"],
                layer=job["layer"],
                dataset=job["dataset"],
                agg=job["agg"],
            )
        )
        for job in _stat_batch_jobs_for_geo(wildcards)
    )


def _geo_seed_limit(method):
    for analysis in EXPERIMENT.geometric_analyses(REGISTRY):
        if method in analysis.methods:
            return analysis.seed_limit
    return 200


rule collect_geo_seeds:
    input:
        stats=_stat_done_inputs_for_geo
    output:
        seeds="artifacts/experiments/{experiment}/objects/geometric/sae={sae}/layer={layer}/method=seed_topk_cosine/seeds.json"
    log:
        "logs/geometric/experiment={experiment}.sae={sae}.layer={layer}.method=seed_topk_cosine.seeds.log"
    params:
        stat_dirs=_stat_dirs_for_geo,
        seed_limit=lambda w: _geo_seed_limit("seed_topk_cosine"),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/collect_stat_seeds.py \
          --stat-dirs {params.stat_dirs} \
          --seed-limit {params.seed_limit} \
          --out {output.seeds} \
          > {log} 2>&1
        """


rule geometric_analysis:
    wildcard_constraints:
        method="norm|topk_cosine"
    output:
        result="artifacts/experiments/{experiment}/objects/geometric/sae={sae}/layer={layer}/method={method}/{filename}"
    log:
        "logs/geometric/experiment={experiment}.sae={sae}.layer={layer}.method={method}.filename={filename}.log"
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


rule geometric_seed_topk:
    input:
        seeds="artifacts/experiments/{experiment}/objects/geometric/sae={sae}/layer={layer}/method=seed_topk_cosine/seeds.json"
    output:
        result="artifacts/experiments/{experiment}/objects/geometric/sae={sae}/layer={layer}/method=seed_topk_cosine/neighbors.json"
    log:
        "logs/geometric/experiment={experiment}.sae={sae}.layer={layer}.method=seed_topk_cosine.neighbors.log"
    params:
        top_k=lambda w: _geo_setting("seed_topk_cosine", "top_k"),
        chunk_size=lambda w: _geo_setting("seed_topk_cosine", "chunk_size"),
    shell:
        """
        mkdir -p $(dirname {log})
        python scripts/analyze_geo.py \
          --sae {wildcards.sae} \
          --layer {wildcards.layer} \
          --method seed_topk_cosine \
          --seeds {input.seeds} \
          --top-k {params.top_k} \
          --chunk-size {params.chunk_size} \
          --out {output.result} \
          > {log} 2>&1
        """
