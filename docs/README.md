# Documentation

This directory describes the production experiment workflow around `sae_tools`.
The core boundary is:

```text
sae_tools library/scripts  ->  single artifact computation
Snakemake                  ->  file dependency DAG and reuse
YAML registry              ->  stable resource and experiment definitions
GPU runner                 ->  idle-GPU target allocation and memory records
```

Start here:

- [Workflow](workflow.md): registry, Snakemake DAG, and deterministic paths.
- [Experiments](experiments.md): how to run prompt and response experiment grids.
- [GPU Runner](gpu-runner.md): automatic empty-GPU allocation and status output.
- [Artifacts](artifacts.md): saved activation/stat/geometric outputs and reuse rules.
- [Compatibility](compatibility.md): TL3, SAE, model loading, and hook conventions.

Recommended validation before a run:

```bash
/NAS/chennc/anaconda3/bin/conda run -n sae-tl3 python scripts/inspect_registry.py --strict
/NAS/chennc/anaconda3/bin/conda run -n sae-tl3 snakemake -n
/NAS/chennc/anaconda3/bin/conda run -n sae-tl3 pytest -q
```
