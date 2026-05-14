# Complete Usage

This guide is for running and managing SAE experiments. For implementation
details, see [Developer Guide](development.md).

## Concepts

The workflow has four durable contracts:

- Registry YAML files define stable IDs for models, SAEs, datasets, and analyses.
- Experiment YAML files select a matrix of models, SAEs, layers, datasets, and analysis groups.
- Snakemake owns the dependency DAG and decides which file targets are missing or ready.
- Artifacts use deterministic paths plus sibling `DONE` markers so expensive work can be reused.

The TUI and idle-GPU runner are frontends over this contract. They do not
replace Snakemake.

## Environment

Use the active shell/Python environment for commands:

```bash
conda activate sae-tl3
pip install -e ".[dev,workflow,tui]"
```

Required `.env` values:

```env
MODEL_ROOT=<model-root>
SAE_ROOT=<sae-checkpoint-root>
DATASET_ROOT=<dataset-root>
```

Optional values:

```env
HF_TOKEN=...
HF_ENDPOINT=https://hf-mirror.com
```

Validate the current environment:

```bash
python -c "import sae_tools"
snakemake --version
nvidia-smi
python scripts/inspect_registry.py --strict
```

## TUI

Launch:

```bash
sae-tools-tui
```

Alternative module entrypoint:

```bash
python -m sae_tools_tui
```

Preselect an experiment:

```bash
sae-tools-tui --config configs/experiments/response_grid.yaml
```

The TUI pages are:

- Experiments: scans `configs/experiments/*.yaml` and displays model, SAE, layer, dataset, target, and artifact counts.
- New Experiment: opens a separate page-by-page creation wizard from the
  Experiments page or the `c` key. The wizard validates registry IDs,
  model/SAE compatibility, shared SAE layers, per-dataset split and
  `max_samples` overrides, and writes a new YAML file under
  `configs/experiments/`.
- Checks: reports Python executable, `import sae_tools`, Snakemake, `nvidia-smi`, registry parsing, experiment parsing, requested SAE layers, dataset registry entries, and local resources.
- Resources: checks requested models, datasets, and every requested SAE layer.
- Artifacts: groups activation, statistical, and geometric targets as `done`, `missing`, `incomplete`, or `failed`.
- GPUs: shows `nvidia-smi` memory/utilization data and idle-selection reasons.
- Run: streams command output for Snakemake dry runs and idle-GPU workflow runs.

Run controls are disabled when environment checks fail. The TUI uses the current
environment and does not run `conda activate`.

## Registry Files

Resource definitions:

```text
configs/registry/models.yaml
configs/registry/saes.yaml
configs/registry/datasets.yaml
configs/registry/analyses.yaml
```

Experiment definitions:

```text
configs/experiments/safety_grid.yaml
configs/experiments/response_grid.yaml
```

Model entries provide a stable key and local path under `MODEL_ROOT`.

SAE entries provide a stable key, local directory under `SAE_ROOT`, adapter,
supported layers, default layer, and filename template.

Dataset entries provide adapter, folder under `DATASET_ROOT`, text type,
optional split/subset, and optional label-field override.

Do not put machine-specific absolute paths in registry YAML or docs. Keep roots
in `.env` and keep registry paths relative to those roots.

Analysis entries define statistical aggregations/metrics or geometric methods.

## Experiments

Default prompt grid:

```bash
snakemake -n
python scripts/run_idle_gpu_workflow.py
```

Response grid:

```bash
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml
python scripts/run_idle_gpu_workflow.py --config configs/experiments/response_grid.yaml
```

Run only specific stages:

```bash
python scripts/run_idle_gpu_workflow.py --stages activations
python scripts/run_idle_gpu_workflow.py --stages stat,geometric
```

The default stage order is:

```text
activations -> stat -> geometric
```

## Snakemake

Preview the default experiment:

```bash
snakemake -n
```

Preview a specific experiment:

```bash
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml
```

Run missing targets on one process:

```bash
snakemake -j 1 --rerun-incomplete
```

Snakemake computes targets from the selected experiment YAML. Existing
activation artifacts are reused when `acts.pt` and `DONE` exist. Avoid
`--forceall`, `-F`, and `-R activations` unless cache regeneration is intended.

## Idle-GPU Runner

The runner starts one Snakemake target per selected GPU:

```bash
python scripts/run_idle_gpu_workflow.py
```

Default idle filters:

```text
GPU-Util <= 0
used memory <= 512 MiB
free memory >= 30000 MiB
```

Useful options:

```bash
python scripts/run_idle_gpu_workflow.py --dry-run
python scripts/run_idle_gpu_workflow.py --include-gpus 0,1
python scripts/run_idle_gpu_workflow.py --exclude-gpus 6,7
python scripts/run_idle_gpu_workflow.py --max-gpus 2
python scripts/run_idle_gpu_workflow.py --max-used-mib 1024
python scripts/run_idle_gpu_workflow.py --min-free-mib 24000
python scripts/run_idle_gpu_workflow.py --poll-interval 15
```

By default the runner uses `conda run -n sae-tl3` for each Snakemake target.
Use `--no-conda-run` to run targets in the current environment. The TUI uses
`--no-conda-run` because it is launched from the desired environment.

Runner records are written under:

```text
runs/gpu_memory/<run_id>/
  gpu_snapshot_initial.md
  gpu_snapshot_final.md
  target_plan.md
  memory_usage.md
  memory_usage.csv
  logs/*.log
```

## Artifacts

Activation cache:

```text
artifacts/experiments/<experiment>/objects/activations/
  model=<model>/sae=<sae>/layer=<layer>/
  dataset=<dataset>/split=<split>/n=<max_samples>/
    acts.pt
    meta.json
    DONE
```

Statistical output:

```text
artifacts/experiments/<experiment>/objects/stat/
  model=<model>/sae=<sae>/layer=<layer>/
  dataset=<dataset>/agg=<agg>/
    feature_table.parquet
    summary.json
    top_features.json
    pareto_front.json
    metric=<metric>/metrics.json
    plots/pr_space.color=<color>.png
    DONE
```

Geometric output:

```text
artifacts/experiments/<experiment>/objects/geometric/sae=<sae>/layer=<layer>/method=norm/metrics.json
artifacts/experiments/<experiment>/objects/geometric/sae=<sae>/layer=<layer>/method=seed_topk_cosine/neighbors.json
```

Status meanings:

- `done`: target file and sibling `DONE` marker exist.
- `missing`: neither target nor log indicates a completed or attempted target.
- `incomplete`: target or marker exists without the other.
- `failed`: target is absent and a non-empty expected log exists.

## Common Tasks

Inspect registry and local paths:

```bash
python scripts/inspect_registry.py --strict
```

Download Qwen-Scope SAE layers:

```bash
python download_saes.py --sae_profile qwen-scope-qwen3-8b-l0-50 --layers 15,18,21,24,27,30,33
```

Generate one activation artifact for a smoke test:

```bash
python scripts/gen_activations_one.py \
  --model qwen3-8b \
  --sae qwen-scope-qwen3-8b-l0-50 \
  --layer 18 \
  --dataset ToxicChat_prompt \
  --max-samples 2
```

Run one statistical analysis target through Snakemake by target path:

```bash
snakemake -j 1 --rerun-incomplete \
  artifacts/experiments/response_grid/objects/stat/model=qwen3-8b/sae=qwen-scope-qwen3-8b-l0-50/layer=18/dataset=ToxicChat_prompt/agg=max/DONE
```

## Layer Trend Analysis

After stat artifacts exist for a multi-layer experiment, generate layer-trend
plots and an HTML report:

```bash
python scripts/analyze_layer_trends.py \
  --config configs/experiments/response_grid.yaml
```

Default outputs:

```text
artifacts/experiments/response_grid/reports/
  layout.json
  pages/layer_trends/
    layer_trends.html
    layer_trends.md
    layer_trends.csv
    missing_artifacts.csv
    summary.json
    plots/<metric>.png
```

The HTML report summarizes each metric's best layer by `topk_mean`, embeds one
plot per metric, and links to the CSV files for detailed inspection. The
Markdown report is still written for compatibility. The script only reads
existing `feature_table.parquet` files; it does not launch Snakemake or
recompute statistical artifacts.

Useful filters:

```bash
python scripts/analyze_layer_trends.py \
  --metrics f1,auroc \
  --saes qwen-scope-qwen3-8b-l0-50 \
  --datasets ToxicChat_response \
  --aggs max \
  --layers 15,18,21,24,27,30,33
```

Use `--strict` when every selected layer must have a complete stat artifact.

Generate a self-contained artifact plot report for the selected experiment:

```bash
sae-tools-report artifacts \
  --config configs/experiments/response_grid.yaml
```

The report is written to
`artifacts/experiments/<experiment>/reports/pages/artifacts/artifacts.html`.
It embeds PNG plots directly into one HTML file and includes browser-side
filters for model, SAE, layer, dataset, aggregation, and plot color.

## HTML Report Server

Serve the generated reports and a read-only dashboard over local HTTP:

```bash
sae-tools-report serve \
  --config configs/experiments/response_grid.yaml \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765/` in a browser. The dashboard combines experiment
status, artifact status, top feature lists, and Feature Activation heatmaps.
It does not start Snakemake, recompute artifacts, or load the full model.

Feature heatmaps require existing activation artifacts plus `MODEL_ROOT` and
`DATASET_ROOT` so the service can load the local tokenizer and dataset metadata.
If those resources are missing, the report and artifact status pages still work
and the feature API returns a clear error.

## Troubleshooting

`sae-tools-tui` is not found:

```bash
pip install -e ".[tui]"
python -m sae_tools_tui
```

`textual` is missing:

```bash
pip install -e ".[tui]"
```

`snakemake` is missing:

```bash
pip install -e ".[workflow]"
```

The registry loads but resources are missing:

- Check `.env`.
- Check whether registry paths are relative to the correct root.
- Download missing SAE layers.
- Put datasets under `DATASET_ROOT` using the folder names in `datasets.yaml`.

Interrupted run:

```bash
snakemake -j 1 --rerun-incomplete
python scripts/run_idle_gpu_workflow.py --stages stat,geometric
```
