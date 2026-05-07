# Artifacts

Artifacts are part of the workflow contract. Do not rely on timestamped output
directories for experiments that need reuse.

## Activation Cache

Activation output:

```text
artifacts/activations/model=<model>/sae=<sae>/layer=<layer>/
  dataset=<dataset>/split=<split>/n=<max_samples>/
    acts.pt
    meta.json
    DONE
```

`acts.pt` contains:

```text
sparse_acts
valid_token_idx
seq_lens
shape
```

`meta.json` records model, SAE, layer, dataset, data type, sample count, script,
git commit, input hash, and output path. `DONE` is the completion marker used
by the workflow and UI.

## Statistical Outputs

For each dataset and aggregation:

```text
artifacts/analyses/stat/model=<model>/sae=<sae>/layer=<layer>/
  dataset=<dataset>/agg=<agg>/
    feature_table.parquet
    summary.json
    top_features.json
    pareto_front.json
    plots/pr_space.color=diff.png
    plots/pr_space.color=diff.pdf
    plots/pr_space.color=ratio.png
    plots/pr_space.color=ratio.pdf
    metric=pearson/metrics.json
    metric=auroc/metrics.json
    metric=f1/metrics.json
    DONE
```

`feature_table.parquet` is the durable source for scatter plots. The plot files
are saved for inspection, but the table is the reproducible calculation output.

## Geometric Outputs

SAE-intrinsic outputs:

```text
artifacts/analyses/geometric/sae=<sae>/layer=<layer>/method=norm/metrics.json
```

Experiment-scoped seeded outputs:

```text
artifacts/analyses/geometric/experiment=<experiment>/sae=<sae>/layer=<layer>/
  method=seed_topk_cosine/
    seeds.json
    neighbors.json
    meta.json
    DONE
```

`seed_topk_cosine` depends on statistical top features, so the experiment name
is part of the path.

## Reuse Rules

- Existing activation artifacts are reused when `acts.pt` and `DONE` exist.
- Do not use `snakemake -F`, `--forceall`, or `-R activations` unless you want
  to regenerate activation caches.
- If a run is interrupted, rerun with `--rerun-incomplete`.
- If only downstream analyses failed, rerun the GPU runner with
  `--stages stat,geometric`.

Generated `artifacts/`, `logs/`, `.snakemake/`, and per-run memory logs are
gitignored. Keep only stable docs and configuration in git.
