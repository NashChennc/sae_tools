# Experiments

The TUI can scan and compare all experiment YAML files:

```bash
sae-tools-tui
```

The shell commands below are the backend commands that the TUI wraps.

## Prompt Grid

The default prompt experiment is:

```text
configs/experiments/safety_grid.yaml
```

Preview:

```bash
conda activate sae-tl3
snakemake -n
```

Run with automatic GPU allocation:

```bash
python scripts/run_idle_gpu_workflow.py
```

## Response Grid

The response experiment is:

```text
configs/experiments/response_grid.yaml
```

It currently includes:

```text
ToxicChat_response
Aegis2.0_response
BeaverTails_response
```

Preview:

```bash
conda activate sae-tl3
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml
```

Run:

```bash
python scripts/run_idle_gpu_workflow.py \
  --config configs/experiments/response_grid.yaml
```

The runner executes stages in order:

```text
activations -> stat -> geometric
```

Existing `acts.pt` files with sibling `DONE` are reused. Missing statistical
and geometric outputs are filled without regenerating activation caches.

## Adding a Dataset

1. Register the dataset in `configs/registry/datasets.yaml`.
2. Use a stable ID with text type suffix, such as `MyDataset_response`.
3. Set `data_type` to `prompt` or `response`.
4. If the label field is not the default `<data_type>_label`, set
   `label_field`.
5. Add the dataset ID to an experiment YAML.
6. Run `scripts/inspect_registry.py --strict`.
7. Dry-run Snakemake before launching GPU jobs.

Example:

```yaml
MyDataset_response:
  adapter: MyDataset
  folder: MyDataset
  data_type: response
  label_field: prompt_label
  split: null
```

## Changing Sample Count

The registry default is `max_samples: 1000`. Override per experiment:

```yaml
datasets:
  - dataset: BeaverTails_response
    max_samples: 2000
```

The sample count is part of the activation path (`n=<value>`), so different
sample limits produce distinct reusable caches.
