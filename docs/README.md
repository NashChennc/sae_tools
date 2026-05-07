# Documentation

This directory documents the SAE experiment workflow from first setup through
development.

## Primary Guides

- [Quick Start](quick-start.md): shortest path from a fresh environment to a verified dry run.
- [Complete Usage](usage.md): full operator guide for the TUI, Snakemake, GPU runner, registries, resources, and artifacts.
- [Developer Guide](development.md): architecture, code layout, extension points, tests, and release checks.

## Topic References

- [Workflow](workflow.md): registry, Snakemake DAG, deterministic paths, and shared runtime helpers.
- [Experiments](experiments.md): prompt and response experiment grids plus common edits.
- [GPU Runner](gpu-runner.md): automatic empty-GPU allocation and status output.
- [Artifacts](artifacts.md): saved activation/stat/geometric outputs and reuse rules.
- [Compatibility](compatibility.md): TL3, SAE, model loading, and hook conventions.

## Project Boundary

```text
sae_tools library/scripts  ->  one artifact computation per command
Snakemake                  ->  file dependency DAG and reuse
YAML registry              ->  stable resource and experiment definitions
GPU runner                 ->  idle-GPU target allocation and memory records
TUI                        ->  frontend/status display over the existing backend
```

Recommended validation before a run:

```bash
python scripts/inspect_registry.py --strict
snakemake -n
pytest -q
```
