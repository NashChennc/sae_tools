# Workflow

## Layers

The experiment system has four layers:

```text
sae_tools library     model, dataset, statistical, geometric, dashboard code
CLI scripts           one artifact per command
Snakemake             dependency tracking, reruns, reuse
GPU runner            target-level GPU allocation and memory records
TUI                   frontend/status display over workflow runtime helpers
```

The TUI calls Snakemake dry-run commands and the idle-GPU runner. It does not
implement scheduling itself.

Shared scan and command-building helpers live in
`src/sae_tools/workflow/runtime.py`. Use them from both scripts and UI code so
target expansion, artifact status, resource checks, GPU classification, and
command construction stay consistent.

## Registry

Stable resource definitions live in:

```text
configs/registry/models.yaml
configs/registry/saes.yaml
configs/registry/datasets.yaml
configs/registry/analyses.yaml
```

Experiment matrices live in:

```text
configs/experiments/safety_grid.yaml
configs/experiments/response_grid.yaml
```

Dataset IDs should include the text slice being analyzed, for example
`ToxicChat_prompt` or `ToxicChat_response`. Response datasets normally use
`response_label`; `ToxicChat_response` explicitly uses `prompt_label` because
ToxicChat does not provide a response label.

## Snakemake

The default workflow is:

```bash
conda activate sae-tl3
snakemake -n
snakemake -j 1 --rerun-incomplete
```

Run a non-default experiment config:

```bash
snakemake -n --config experiment_config=configs/experiments/response_grid.yaml
```

The `Snakefile` computes targets from the selected experiment YAML. Completed
activation artifacts are protected by Snakemake and reused by downstream
statistical and geometric jobs.

## Deterministic Paths

Activation artifacts are keyed by model, SAE, layer, dataset, split, and sample
limit:

```text
artifacts/experiments/<experiment>/objects/activations/
  model=qwen3-8b/
  sae=qwen-scope-qwen3-8b-l0-50/
  layer=18/
  dataset=ToxicChat_response/
  split=default/
  n=1000/
    acts.pt
    meta.json
    DONE
```

Statistical artifacts are keyed by model, SAE, layer, dataset, and aggregation.
Metric payloads live under `metric=<name>/`.

SAE-intrinsic geometric outputs such as decoder norm are keyed by SAE and
layer. Seeded geometric outputs depend on statistical top features, so they are
also scoped by experiment:

```text
artifacts/experiments/response_grid/objects/geometric/...
```

This prevents response experiments from overwriting prompt experiment seed
neighbors.
