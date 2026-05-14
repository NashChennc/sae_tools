# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable with all optionals)
pip install -e ".[dev,workflow,tui]"

# If not installed, use PYTHONPATH prefix for all commands:
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m sae_tools_tui --help
PYTHONPATH=src python -m sae_tools.reporting --help

# Run tests
PYTHONPATH=src python -m pytest                                                  # full suite
PYTHONPATH=src python -m pytest tests/test_tui_runtime.py tests/test_idle_gpu_runner.py tests/test_workflow_registry.py  # workflow/TUI focused

# Smoke checks
python -m compileall -q src scripts
python scripts/inspect_registry.py --strict
snakemake -n

# Snakemake (default experiment: safety_grid.yaml)
snakemake -n                                                                     # dry-run default experiment
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml   # dry-run a specific experiment
snakemake -j 1 --rerun-incomplete                                                # run one local process

# GPU workflow runner
python scripts/run_idle_gpu_workflow.py                                          # all stages on idle GPUs
python scripts/run_idle_gpu_workflow.py --stages stat,geometric                  # downstream stages only
python scripts/run_idle_gpu_workflow.py --dry-run                                # plan without executing

# Reporting
python scripts/analyze_layer_trends.py --config configs/experiments/response_grid.yaml
sae-tools-report artifacts --config configs/experiments/response_grid.yaml       # generate artifact HTML report
sae-tools-report serve --config configs/experiments/response_grid.yaml           # serve dashboard at http://127.0.0.1:8765

# TUI
sae-tools-tui
```

## Architecture

The pipeline boundary is:

```
single-artifact scripts -> Snakemake DAG -> idle-GPU runner -> optional TUI frontend
```

**`src/sae_tools/workflow/`** — The backbone shared by all other layers:
- `runtime.py`: target expansion, artifact scanning (`done`/`missing`/`incomplete`/`failed`), GPU classification, environment checks, and command builders. The TUI, runner, and reporting all consume this module — never duplicate its logic.
- `artifacts.py`: deterministic path helpers (`activation_path`, `stat_analysis_dir`, `geometric_path`, `done_path`). All artifact paths are functions of registry IDs, not hardcoded strings.
- `registry.py`: `Registry` loads YAML specs (models, SAEs, datasets, analyses) from `configs/registry/`. `ExperimentSpec` loads experiment matrices from `configs/experiments/` and expands them into job lists (`activation_jobs`, `stat_batch_jobs`, `geometric_jobs`).

**`scripts/`** — Single-purpose workers (one script = one Snakemake rule):
- `gen_activations_one.py` — writes `acts.pt` + `DONE` marker
- `analyze_stat.py` — writes per-aggregation stat outputs (`feature_table.parquet`, `metrics.json`, etc.) + `DONE`
- `analyze_geo.py` — writes geometric outputs (`neighbors.json`, `metrics.json`)
- `collect_stat_seeds.py` — gathers seed features from stat outputs for geometric seed-based analysis
- `run_idle_gpu_workflow.py` — polls `nvidia-smi`, classifies idle GPUs, and launches Snakemake targets on them
- `inspect_registry.py` — validates registry YAML and adapter backends

**`Snakefile`** — Owns the DAG. Three stages: `activations` → `stat_analysis_batch` → `geometric_analysis` (+ `collect_geo_seeds`). Uses `ExperimentSpec` to expand wildcards. Stat targets depend on activations; geometric targets depend on stat `DONE` markers.

**`src/sae_tools/analysis/`** — Pure analysis code:
- `statistical/`: precision/recall/F1 metrics, per-feature stat tables
- `geometric/`: norm analysis, cosine similarity, UMAP
- `dashboard/`: feature viewer, heatmap, logit lens, comparator (notebook-rendered, not served by the reporting server)

**`src/sae_tools/adapters/`** — Registry-backed adapter pattern:
- `models/profiles.py`: model loading profiles (hook point conventions, dtype, device)
- `saes/profiles.py`: SAE loading profiles (checkpoint format, hook location)
- `datasets/registry.py` + individual dataset adapters: standardized `load()` → (texts, labels) interface

**`src/sae_tools/reporting/`** — HTML report generation and read-only HTTP server. Reads existing artifacts via workflow helpers. Does NOT schedule, mutate, or regenerate artifacts.

**`src/sae_tools_tui/`** — Textual TUI frontend. Displays scans from `sae_tools.workflow.runtime` and streams runner output. No independent scheduling, no artifact mutation.

**`src/sae_tools/experiment_store/`** — File-backed experiment bundle with Parquet index tables (jobs, artifacts, features, summaries, reports). Owns `artifacts/experiments/<id>/`.

## Key Patterns

**Artifact readiness contract**: `acts.pt` + sibling `DONE` marker = ready. Never `--forceall` without explicit intent.

**Registry IDs + experiment matrices**: Stable resource IDs live in `configs/registry/*.yaml`. Experiment matrices in `configs/experiments/*.yaml` reference those IDs. Dataset IDs encode the analyzed text field (e.g., `ToxicChat_prompt`, `ToxicChat_response`).

**Path policy**: `configs/registry/*.yaml` uses relative paths (relative to `MODEL_ROOT`, `SAE_ROOT`, `DATASET_ROOT`). `.env` holds the machine-specific root prefixes. Never hardcode absolute paths in source or configs.

**Generated outputs to ignore**:

```
artifacts/  logs/  .snakemake/  runs/gpu_memory/<run_id>/  report/
*.pt  *.pth  *.pkl  *.safetensors
```
