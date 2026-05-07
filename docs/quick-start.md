# Quick Start

This guide gets a local environment to the point where it can inspect resources,
preview workflow targets, and open the TUI.

## 1. Create the Environment

```bash
conda create -n sae-tl3 python=3.11 -y
conda activate sae-tl3
pip install -e ".[dev,workflow,tui]"
```

The base package contains model, SAE, dataset, and analysis code. Optional
extras add:

- `dev`: pytest.
- `workflow`: Snakemake.
- `tui`: Textual and Rich for `sae-tools-tui`.

RAPIDS is only needed for geometric methods that require its GPU dataframe or
nearest-neighbor stack. Install it separately for the target CUDA runtime when
those methods are used.

## 2. Configure Local Resource Roots

Create `.env` in the repository root:

```env
MODEL_ROOT=<model-root>
SAE_ROOT=<sae-checkpoint-root>
DATASET_ROOT=<dataset-root>
```

Registry paths under `configs/registry/` are relative to these roots unless the
path is already absolute.

Optional Hugging Face settings:

```env
HF_TOKEN=...
HF_ENDPOINT=https://hf-mirror.com
```

## 3. Prepare Resources

Inspect registered model, SAE, and dataset paths:

```bash
python scripts/inspect_registry.py --strict
```

Download Qwen-Scope SAE files into `SAE_ROOT`:

```bash
python download_saes.py \
  --sae_profile qwen-scope-qwen3-8b-l0-50 \
  --layers 15,18,21,24,27,30,33

python download_saes.py \
  --sae_profile qwen-scope-qwen3-8b-l0-100 \
  --layers 15,18,21,24,27,30,33
```

`scripts/inspect_registry.py` checks each SAE profile's default layer. The TUI
and workflow runtime check every layer requested by the selected experiment.

## 4. Preview the Workflow

Default experiment:

```bash
snakemake -n
```

Response-grid experiment:

```bash
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml
```

The response grid should expand to 42 activation targets, 84 statistical batch
targets, and 28 geometric targets.

## 5. Open the TUI

```bash
sae-tools-tui
```

If the console script is not available because the package has not been
installed in the active environment, use:

```bash
PYTHONPATH=src python -m sae_tools_tui
```

The TUI starts at the experiment-selection page. It scans experiment YAML files,
registry entries, artifacts, GPUs, and the current Python/shell environment. It
does not activate conda environments by itself.

## 6. Run Work

Use the TUI `Dry run` action for a live Snakemake dry-run log.

Run missing targets on idle GPUs from the shell:

```bash
python scripts/run_idle_gpu_workflow.py
```

Run only downstream stages after activation caches exist:

```bash
python scripts/run_idle_gpu_workflow.py --stages stat,geometric
```

Activation artifacts are reused when both `acts.pt` and `DONE` are present. Do
not use `snakemake --forceall`, `-F`, or `-R activations` unless regenerating
activation caches is intentional.
