# GPU Runner

`scripts/run_idle_gpu_workflow.py` allocates Snakemake targets to currently
empty GPUs and records memory use. It does not replace Snakemake; it launches
one Snakemake target per selected GPU.

The TUI `Run missing` action invokes this runner with `--no-conda-run`, so it
uses the environment that launched the TUI. Direct shell usage keeps the
runner's CLI defaults unless `--no-conda-run` is passed. The runner resolves
the conda executable from `CONDA_EXE`, falling back to `conda` on `PATH`.

## Empty GPU Selection

Default filters:

```text
GPU-Util <= 0
used memory <= 512 MiB
free memory >= 30000 MiB
```

The 512 MiB threshold allows the small driver baseline on otherwise empty
cards. Cards with real allocations are skipped.

Dry-run selection:

```bash
conda activate sae-tl3
python scripts/run_idle_gpu_workflow.py --dry-run
```

Limit or override selection:

```bash
python scripts/run_idle_gpu_workflow.py --include-gpus 0,1,2
python scripts/run_idle_gpu_workflow.py --exclude-gpus 6,7
python scripts/run_idle_gpu_workflow.py --max-gpus 2
python scripts/run_idle_gpu_workflow.py --max-used-mib 1024
python scripts/run_idle_gpu_workflow.py --min-free-mib 24000
```

## Status Output

The runner prints:

```text
[activations] gpu=0 target=...
[status] stage=activations pending=0 active=3 | gpu=0 used=... peak=... util=...
[stat] done gpu=1 status=success peak_delta=...
```

Use `--poll-interval` to change the status cadence:

```bash
python scripts/run_idle_gpu_workflow.py --poll-interval 15
```

## Memory Records

Every run writes:

```text
runs/gpu_memory/<run_id>/
  gpu_snapshot_initial.md
  gpu_snapshot_final.md
  target_plan.md
  memory_usage.md
  memory_usage.csv
  logs/*.log
```

The long-lived planning table is:

```text
runs/gpu_memory/allocation_table.md
```

Per-run directories are gitignored; the allocation table is intentionally kept
visible so future experiment planning can use measured peak memory.

## Stage Resume

Run only downstream stages after activations exist:

```bash
python scripts/run_idle_gpu_workflow.py --stages stat,geometric
```

Run only activation generation:

```bash
python scripts/run_idle_gpu_workflow.py --stages activations
```

Snakemake still decides whether each target is ready or missing.
