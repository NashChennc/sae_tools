# Developer Guide

This guide is for extending and maintaining the SAE tools codebase.

## Setup

```bash
conda create -n sae-tl3 python=3.11 -y
conda activate sae-tl3
pip install -e ".[dev,workflow,tui]"
```

If the package is not installed in the current Python, use `PYTHONPATH=src` for
local checks:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m sae_tools_tui --help
```

## Code Layout

```text
src/sae_tools/
  adapters/       model, SAE, and dataset adapter registries
  analysis/       statistical, geometric, and dashboard analysis code
  experiment_store/  experiment bundle manifest and Parquet indexes
  model/          model loading, hooks, activation generation helpers
  reporting/      HTML report rendering and read-only HTTP dashboard service
  workflow/       registry models, artifact path helpers, runtime scans

scripts/
  gen_activations_one.py
  analyze_stat.py
  analyze_geo.py
  collect_stat_seeds.py
  inspect_registry.py
  run_idle_gpu_workflow.py

src/sae_tools_tui/
  __main__.py     CLI/module entrypoint
  app.py          Textual UI
  backend.py      TUI-facing backend wrapper
  widgets.py      presentation helpers

configs/
  registry/       stable model, SAE, dataset, and analysis IDs
  experiments/    experiment matrices

docs/
  user and developer documentation
```

## Architecture Rules

Single-artifact scripts do the expensive work. Snakemake owns dependencies and
reuse. The idle-GPU runner only allocates Snakemake targets to GPUs. The TUI is
a frontend over the workflow runtime and runner.

Do not duplicate target expansion, artifact status, GPU parsing, or command
construction in UI code. Use:

```python
from sae_tools.workflow import runtime
```

Important runtime helpers:

- `workflow_target_records`: expand experiment YAML into activation/stat/geometric targets.
- `scan_artifacts`: classify targets as `done`, `missing`, `incomplete`, or `failed`.
- `scan_resources`: check requested models, datasets, and every requested SAE layer.
- `check_environment`: return displayable environment checks without crashing.
- `query_gpus` and `classify_gpus`: parse `nvidia-smi` and idle-selection reasons.
- `build_snakemake_dry_run_command`: build TUI dry-run commands.
- `build_idle_runner_command`: build TUI run-missing commands.

## Reporting Service Development

The browser report service lives under `src/sae_tools/reporting/`.

Responsibilities:

- Render existing analysis outputs as HTML.
- Serve files under `artifacts/experiments/<experiment>/reports/` through safe
  path checks.
- Display experiment and artifact status from `sae_tools.workflow.runtime`.
- Reuse `FeatureActivationViewer` for token heatmaps without requiring
  `ipywidgets`.

Non-responsibilities:

- No workflow scheduling or Snakemake launches.
- No artifact deletion or regeneration.
- No full model or GPU loading for the default dashboard. Feature heatmaps may
  load a local tokenizer and dataset metadata only when requested.

## Adding or Editing Registry Entries

Models:

1. Add or edit `configs/registry/models.yaml`.
2. Add a matching profile under `src/sae_tools/adapters/models/` if needed.
3. Keep `local_path` relative to `MODEL_ROOT`; machine-specific roots belong in `.env`.

SAEs:

1. Add or edit `configs/registry/saes.yaml`.
2. Add a matching profile/adapter under `src/sae_tools/adapters/saes/` if needed.
3. List all supported layers in `layers`.
4. Keep `local_dir` relative to `SAE_ROOT`; machine-specific roots belong in `.env`.
5. Keep `file_template` compatible with `SAESpec.layer_filename(layer)`.

Datasets:

1. Add or edit `configs/registry/datasets.yaml`.
2. Add a dataset adapter under `src/sae_tools/adapters/datasets/` if needed.
3. Use IDs that encode the analyzed text field, such as `MyDataset_prompt` or `MyDataset_response`.
4. Set `label_field` when the default `<data_type>_label` is not correct.
5. Keep `folder` relative to `DATASET_ROOT`; machine-specific roots belong in `.env`.

Analyses:

1. Add or edit `configs/registry/analyses.yaml`.
2. Keep `kind` as `statistical` or `geometric`.
3. Verify the corresponding scripts can consume the configured metrics or methods.

Validate:

```bash
python scripts/inspect_registry.py --strict
snakemake -n
```

## Adding an Experiment

Create `configs/experiments/<name>.yaml`:

```yaml
models: [qwen3-8b]
saes:
  - qwen-scope-qwen3-8b-l0-50
layers: [18]
datasets:
  - ToxicChat_prompt
activation:
  overwrite: false
  batch_size: 2
analyses:
  - stat_basic
  - geo_basic
```

Rules:

- Omit `layers` only when each SAE should use its default layer.
- Set `activation.overwrite: false` unless regeneration is intentional.
- Use per-dataset `max_samples` overrides when an experiment needs a different cache key.
- Dry-run before launching GPU work.

## TUI Development

The TUI lives under `src/sae_tools_tui/`.

Responsibilities:

- Display scans from `sae_tools.workflow.runtime`.
- Create experiment YAML files through the registry-backed TUI wizard.
- Stream stdout/stderr from Snakemake dry runs and idle-GPU runner commands.
- Refresh artifact and GPU state while a command runs.
- Disable run buttons when environment checks fail.

Non-responsibilities:

- No independent scheduling.
- No automatic `conda activate`.
- No model, SAE, or dataset downloads.
- No direct artifact mutation beyond launching existing backend commands.

## Tests

Focused workflow/TUI checks:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_tui_runtime.py \
  tests/test_idle_gpu_runner.py \
  tests/test_workflow_registry.py
```

Full suite:

```bash
PYTHONPATH=src python -m pytest
```

Smoke checks:

```bash
PYTHONPATH=src python -m sae_tools_tui --help
PYTHONPATH=src python -m sae_tools.reporting --help
python -m compileall -q src scripts
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

The console script smoke check requires an installed editable package:

```bash
pip install -e ".[tui]"
sae-tools-tui --help
sae-tools-report --help
```

## Artifact and Git Hygiene

Generated outputs are not source files:

```text
artifacts/
logs/
.snakemake/
runs/gpu_memory/<run_id>/
*.pt
*.safetensors
```

Keep source control focused on:

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `tests/`
- stable planning files such as `runs/gpu_memory/allocation_table.md`

Do not delete, force-regenerate, or rewrite expensive artifacts during routine
development. Use Snakemake reuse and `--rerun-incomplete` for interrupted runs.

## Release Checklist

Before handing off substantial changes:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m sae_tools_tui --help
python -m compileall -q src scripts
python scripts/inspect_registry.py --strict
snakemake -n
```

Some checks require optional dependencies, GPUs, and configured local resources.
Report any skipped or unavailable checks explicitly.
