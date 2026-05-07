# SAE Analysis Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/)

A lightweight toolkit for analyzing Sparse Autoencoders (SAEs) that integrates statistical metrics, geometric analysis, and visualization dashboards.

Implemented using `sae_lens` and `transformer_lens`, this toolkit features an Anthropic-style dashboard and analysis workflow developed entirely in pure Python and Jupyter Notebooks.

> Due to the restricted network access in my experimental environment, I intentionally avoided network dependencies during development, such as port mapping (in `sae_dashboard`) and online model loading (in `transformer_lens`), which have given me a lot of trouble.

<p>
  <img src="images/Aegis1.0_pr_space.png" alt="scatter" width="24%">
  <img src="images/dashboard.png" alt="dashboard" width="24%">
  <img src="images/cross_corr.png" alt="cross correlation" width="24%">
  <img src="images/dimreduction.png" alt="dimension reduction" width="24%">
</p>

## Start Here

- [Quick Start](docs/quick-start.md): install the environment, configure local paths, verify resources, and run a dry run.
- [Complete Usage](docs/usage.md): TUI, Snakemake, idle-GPU runner, registries, experiments, resources, and artifacts.
- [Developer Guide](docs/development.md): architecture, code layout, extension points, tests, and release checks.
- [Agent Conventions](AGENTS.md): project rules for coding agents working in this repository.

Topic references:

- [Workflow](docs/workflow.md): registry, Snakemake DAG, deterministic paths, and shared runtime helpers.
- [Experiments](docs/experiments.md): prompt and response grids plus common edits.
- [GPU Runner](docs/gpu-runner.md): idle-GPU allocation and memory records.
- [Artifacts](docs/artifacts.md): activation, statistical, and geometric output contracts.
- [Compatibility](docs/compatibility.md): TL3, SAE, model loading, and hook conventions.

## One-Minute Setup

```bash
conda create -n sae-tl3 python=3.11 -y
conda activate sae-tl3
pip install -e ".[dev,workflow,tui]"
```

Create `.env` in the repository root:

```env
MODEL_ROOT=<model-root>
SAE_ROOT=<sae-checkpoint-root>
DATASET_ROOT=<dataset-root>
```

Validate the install and registry:

```bash
python scripts/inspect_registry.py --strict
snakemake -n
python -m sae_tools_tui --help
```

Launch the experiment-management TUI:

```bash
sae-tools-tui
```

If the console script has not been generated in the active environment yet, use:

```bash
python -m sae_tools_tui
```

## Core Commands

```bash
# Snakemake preview for the default experiment.
snakemake -n

# Preview a non-default experiment.
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml

# Run missing workflow artifacts on one process.
snakemake -j 1 --rerun-incomplete

# Run missing targets across idle GPUs.
python scripts/run_idle_gpu_workflow.py

# Run only downstream stages after activations exist.
python scripts/run_idle_gpu_workflow.py --stages stat,geometric
```

Generated artifacts, logs, Snakemake state, and per-run GPU memory logs are intentionally gitignored. Stable configuration and documentation are the source-controlled contract.
