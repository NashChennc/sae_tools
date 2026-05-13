# Artifacts

Artifacts are stored as experiment bundles so config, workflow data, indexes,
and rendered HTML reports can be managed from one root:

```text
artifacts/experiments/<experiment>/
  manifest.json
  config.resolved.yaml
  registry.snapshot.json
  index/
    jobs.parquet
    artifacts.parquet
    features.parquet
    summaries.parquet
    reports.parquet
  objects/
  reports/
    layout.json
    pages/
```

`objects/` keeps native large artifacts. `index/` keeps compact Parquet tables
used by the dashboard and report code. `reports/layout.json` describes report
pages independently from the rendering code.

## Activation Cache

Activation output:

```text
artifacts/experiments/<experiment>/objects/activations/
  model=<model>/sae=<sae>/layer=<layer>/
  dataset=<dataset>/split=<split>/n=<max_samples>/
    acts.pt
    meta.json
    DONE
```

`acts.pt` contains `sparse_acts`, `valid_token_idx`, `seq_lens`, and `shape`.
`meta.json` records model, SAE, layer, dataset, data type, sample count, script,
git commit, input hash, and output path.

## Statistical Outputs

For each dataset and aggregation:

```text
artifacts/experiments/<experiment>/objects/stat/
  model=<model>/sae=<sae>/layer=<layer>/dataset=<dataset>/agg=<agg>/
    feature_table.parquet
    summary.json
    top_features.json
    pareto_front.json
    plots/pr_space.color=diff.png
    plots/pr_space.color=ratio.png
    metric=<metric>/metrics.json
    DONE
```

`feature_table.parquet` is the durable source for plots and for
`index/features.parquet`.

## Geometric Outputs

```text
artifacts/experiments/<experiment>/objects/geometric/
  sae=<sae>/layer=<layer>/method=norm/metrics.json
  sae=<sae>/layer=<layer>/method=topk_cosine/neighbors.json
  sae=<sae>/layer=<layer>/method=seed_topk_cosine/
    seeds.json
    neighbors.json
```

All geometric outputs are experiment scoped so seeded neighbor runs cannot
overwrite results from another experiment.

## Reports

HTML reports are written under the same bundle:

```text
artifacts/experiments/<experiment>/reports/
  layout.json
  pages/layer_trends/
    layer_trends.html
    layer_trends.md
    layer_trends.csv
    missing_artifacts.csv
    summary.json
    plots/*.png
  pages/artifacts/
    artifacts.html
```

Generate a self-contained artifact plot report with:

```bash
sae-tools-report artifacts --config configs/experiments/response_grid.yaml
```

Serve them with:

```bash
python -m sae_tools.reporting serve --config configs/experiments/response_grid.yaml
```

## Reuse Rules

- Existing activation artifacts are reused when `acts.pt` and `DONE` exist.
- If a run is interrupted, rerun with `--rerun-incomplete`.
- If only downstream analyses failed, rerun the GPU runner with
  `--stages stat,geometric`.

Generated `artifacts/`, `logs/`, `.snakemake/`, and per-run memory logs are
gitignored.
