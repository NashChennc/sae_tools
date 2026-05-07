# SAE Analysis Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/)

A lightweight toolkit for analyzing Sparse Autoencoders (SAEs) that integrates statistical metrics, geometric analysis, and visualization dashboards.

Implemented using `sae_lens` and `transformer_lens`, this toolkit features an Anthropic-style dashboard and analysis workflow developed entirely in pure Python and Jupyter Notebooks.

> Due to the restricted network access in my experimental environment, I intentionally avoided network dependencies during development, such as port mapping (in `sae_dashboard`) and online model loading (in `transformer_lens`), which have given me a lot of trouble.

![scatter](images/Aegis1.0_pr_space.png)
![dashboard](images/dashboard.png)
![cross_corr](images/cross_corr.png)
![dimreduction](images/dimreduction.png)

## Installation

Create the TL3/SAELens environment:

```bash
conda create -n sae-tl3 python=3.11 -y
conda activate sae-tl3
pip install -e ".[dev]"
```

Install this repo in editable mode:

```bash
pip install -e .
```

[rapids-ai](https://docs.rapids.ai/install/#system-req) is needed for geometric method module.

## Configuration

Create a `.env` file in the project root to configure paths:

```env
MODEL_ROOT=./models
SAE_ROOT=./sae_checkpoints
DATASET_ROOT=./datasets
```

Paths about models/saes/datasets in this repo are relative paths based on these roots. You can check your dataset config (and download datasets that fit the config) by running `download_datasets.py`.

Experiment orchestration is driven by YAML registry files:

```text
configs/registry/{models,saes,datasets,analyses}.yaml
configs/experiments/safety_grid.yaml
```

The registry keeps model/SAE IDs stable with the existing Python profiles, while dataset IDs distinguish prompt/response variants such as `ToxicChat_prompt`.

## Profiles and SAE Files

`0_generate_activations.py` uses named profiles so model paths, SAE paths, and file formats stay centralized:

* `qwen3-8b-guard`: `${MODEL_ROOT}/Qwen/Qwen3Guard-Gen-8B`
* `qwen3-8b`: `${MODEL_ROOT}/Qwen/Qwen3-8B`
* `qwen3-8b-base`: compatibility alias for `qwen3-8b`
* `qwen-scope-qwen3-8b-l0-50`: `${SAE_ROOT}/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50/layer{layer}.sae.pt`
* `adamkarvonen`: the existing Adam Karvonen BatchTopK checkpoint path, loaded through the original JumpReLU conversion function

Download the default Qwen-Scope layer 18 SAE into the stable `SAE_ROOT` layout:

```bash
python download_saes.py --sae_profile qwen-scope-qwen3-8b-l0-50 --layers 18
```

Inspect local resource availability:

```bash
python scripts/inspect_registry.py
```

Run a local smoke test for one deterministic activation artifact:

```bash
python scripts/gen_activations_one.py \
  --model qwen3-8b-guard \
  --sae qwen-scope-qwen3-8b-l0-50 \
  --layer 18 \
  --dataset ToxicChat_prompt \
  --max-samples 2
```

Artifacts are written to deterministic paths such as:

```text
artifacts/activations/model=qwen3-8b-guard/sae=qwen-scope-qwen3-8b-l0-50/layer=18/dataset=ToxicChat_prompt/split=default/n=1000/acts.pt
```

Activation artifacts are protected by Snakemake after completion and are reused
whenever `acts.pt` and `DONE` are present. Do not use `--forceall`, `-F`, or
`-R activations` unless you intentionally want to regenerate expensive
activation caches.

Statistical analysis writes durable calculation and visualization artifacts per
`model × sae × dataset × aggregation`:

```text
artifacts/analyses/stat/.../agg=max/
  feature_table.parquet
  summary.json
  top_features.json
  pareto_front.json
  metric=auroc/metrics.json
  plots/pr_space.color=diff.png
  plots/pr_space.color=ratio.png
  DONE
```

The scatter plots are saved together with `feature_table.parquet`, so the plots
are reproducible from the underlying per-feature metrics rather than being the
only copy of the result.

Each completed artifact has sibling `meta.json` and `DONE` files. The legacy
`0_generate_activations.py` entrypoint is still available, but it now writes the
same deterministic artifact layout instead of timestamped `results/SAE_*` runs.

Run the configured workflow with Snakemake:

```bash
# Use the project environment and install workflow dependency if needed
conda activate sae-tl3
pip install -e ".[workflow]"

# Preview jobs
snakemake -n

# Run missing activation + analysis artifacts
snakemake -j 1 --rerun-incomplete
```

Run the same workflow across all currently idle GPUs with enough free memory:

```bash
/NAS/chennc/anaconda3/bin/conda run -n sae-tl3 \
  python scripts/run_idle_gpu_workflow.py
```

The idle-GPU runner treats a GPU as available when `GPU-Util <= 0`, used memory
is at or below `--max-used-mib` MiB, and free memory is at least
`--min-free-mib` MiB. The default `--max-used-mib 512` allows the small driver
baseline on otherwise empty cards while skipping cards with real allocations.
It binds one Snakemake target per GPU with `CUDA_VISIBLE_DEVICES`, runs
activations before downstream analyses, prints active target status while jobs
are running, and writes allocation records under `runs/gpu_memory/<run_id>/`:

```text
gpu_snapshot_initial.md   # simple table of all GPUs and selection reasons
target_plan.md            # stage/target plan
memory_usage.md           # peak GPU memory table per target
logs/*.log                # per-target Snakemake logs
```

Use `--dry-run` to create the tables without launching jobs. Lower
`--min-free-mib` only when you deliberately want to use GPUs with existing
memory allocations.

## Documentation

Workflow and experiment guides live under [docs](docs/README.md):

* **Workflow**: [docs/workflow.md](docs/workflow.md)
* **Experiments**: [docs/experiments.md](docs/experiments.md)
* **GPU Runner**: [docs/gpu-runner.md](docs/gpu-runner.md)
* **Artifacts**: [docs/artifacts.md](docs/artifacts.md)
* **Compatibility**: [docs/compatibility.md](docs/compatibility.md)

For detailed module notes, refer to the internal READMEs:

* **Model & Inference**: [src/sae_tools/model/README.md](src/sae_tools/model/README.md)
* **Adapters**: [src/sae_tools/adapters/README.md](src/sae_tools/adapters/README.md)
* **Analysis**: [src/sae_tools/analysis/README.md](src/sae_tools/analysis/README.md)
* **Statistical Analysis**: [src/sae_tools/analysis/statistical/README.md](src/sae_tools/analysis/statistical/README.md)
* **Geometric Analysis**: [src/sae_tools/analysis/geometric/README.md](src/sae_tools/analysis/geometric/README.md)
* **Dashboard**: [src/sae_tools/analysis/dashboard/README.md](src/sae_tools/analysis/dashboard/README.md)

### Dataset Adapters

Dataset adapters live under `src/sae_tools/adapters/datasets`. See
[src/sae_tools/adapters/datasets/README.md](src/sae_tools/adapters/datasets/README.md)
for usage and registration conventions. They provide the following normalized
structure for Hugging Face `Datasets`:

```
prompt: str
response: str
prompt_label: Optional[str] in ["Safe", "Unsafe"]
response_label: Optional[str] in ["Safe", "Unsafe"]
category: Dict[str, float]
source: str
```

you can customize this structure by `transform` function implementation.

> TODO
> 1. classifier module
> 2. SAE realtime inference & LogitLens
> 3. batch inference in `generate_activations`
