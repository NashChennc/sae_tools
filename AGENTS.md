# AGENTS.md

Project conventions for coding agents working in this repository.

## Scope

This file applies to the `sae_tools` repository rooted at this directory.

The project is a deterministic SAE experiment toolkit. The important boundary is:

```text
single-artifact scripts -> Snakemake DAG -> idle-GPU runner -> optional TUI frontend
```

## Required Working Style

- Read the relevant code and docs before editing.
- Preserve existing user changes in the worktree. Do not revert files unless explicitly asked.
- Keep changes scoped to the requested task.
- Prefer existing helpers and registry abstractions over ad hoc parsing.
- Use deterministic artifact path helpers from `sae_tools.workflow`.
- Do not introduce network downloads, model downloads, or dataset downloads unless the user explicitly asks.
- Do not run destructive commands such as `git reset --hard`, `git checkout --`, or artifact deletion without explicit approval.

## Environment

Recommended environment:

```bash
conda activate sae-tl3
pip install -e ".[dev,workflow,tui]"
```

If the package is not installed in the active Python, use:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m sae_tools_tui --help
```

Required `.env` keys for real runs:

```env
MODEL_ROOT=...
SAE_ROOT=...
DATASET_ROOT=...
```

The TUI and most local commands should use the current shell/Python
environment. Do not add automatic `conda activate` behavior to the TUI.

## Local Path Policy

- Machine-specific absolute paths belong only in `.env` or the caller's shell environment.
- Do not hardcode local roots such as model, SAE, dataset, conda, cache, or user-home paths in source, docs, tests, or registry YAML.
- Documentation may mention environment variable names, but examples should use placeholders rather than real local paths.
- Registry paths should stay relative to `MODEL_ROOT`, `SAE_ROOT`, or `DATASET_ROOT`.

## Workflow Rules

- Snakemake owns dependency tracking and target reuse.
- `scripts/run_idle_gpu_workflow.py` owns idle-GPU allocation.
- `src/sae_tools_tui/` is frontend/status display only.
- Do not implement a second scheduler in the TUI.
- Do not duplicate target expansion, GPU classification, command construction, resource scanning, or artifact status logic.
- Shared workflow/runtime behavior belongs in `src/sae_tools/workflow/runtime.py`.

Use these helpers when possible:

```python
from sae_tools.workflow.runtime import (
    build_idle_runner_command,
    build_runner_target_command,
    build_snakemake_dry_run_command,
    check_environment,
    classify_gpus,
    scan_artifacts,
    scan_experiments,
    scan_resources,
    workflow_target_records,
)
```

## Artifact Rules

Generated outputs are not source changes:

```text
artifacts/
logs/
.snakemake/
runs/gpu_memory/<run_id>/
*.pt
*.pth
*.pkl
*.safetensors
```

Activation artifacts are expensive and reusable. Treat `acts.pt` plus sibling
`DONE` as the ready contract. Do not use `snakemake --forceall`, `-F`, or
`-R activations` unless regeneration is explicitly intended.

Status convention:

- `done`: target and sibling `DONE` exist.
- `missing`: target has not been produced.
- `incomplete`: target or `DONE` exists without the other.
- `failed`: target is absent and a non-empty expected log exists.

## Registry and Experiment Rules

- Stable resource IDs live in `configs/registry/*.yaml`.
- Experiment matrices live in `configs/experiments/*.yaml`.
- Dataset IDs should include the analyzed text type, such as `ToxicChat_prompt` or `ToxicChat_response`.
- SAE checks must cover every experiment-requested layer, not just the default layer.
- Keep `activation.overwrite: false` unless cache regeneration is intentional.
- Dry-run Snakemake before launching GPU jobs.

Expected response-grid target counts:

```text
activations: 42
stat: 84
geometric: 28
```

## Documentation Rules

Keep the docs entrypoints aligned:

- `README.md`: short project overview and navigation.
- `docs/quick-start.md`: first-run setup.
- `docs/usage.md`: complete operator guide.
- `docs/development.md`: developer and extension guide.
- `AGENTS.md`: coding-agent project conventions.

When behavior changes, update the docs in the same change.

## Test Commands

Focused checks for workflow/TUI changes:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_tui_runtime.py \
  tests/test_idle_gpu_runner.py \
  tests/test_workflow_registry.py
```

General checks:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m sae_tools_tui --help
python -m compileall -q src scripts
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

Some checks require optional dependencies, GPUs, and configured local resources.
If a check cannot be run, state the reason clearly.
