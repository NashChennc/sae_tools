"""Analysis modules for SAE activation outputs.

The canonical analysis namespace contains:

* category: category parsing and label construction.
* statistical: feature metrics and PR-space analysis.
* geometric: decoder-vector geometry utilities.
* dashboard: interactive feature inspection views.
"""

from .artifacts import (
    activation_run_pattern,
    build_generate_activations_command,
    find_latest_activation_file,
    require_activation_file,
    require_activation_keys,
)

__all__ = [
    "activation_run_pattern",
    "build_generate_activations_command",
    "category",
    "dashboard",
    "find_latest_activation_file",
    "geometric",
    "require_activation_file",
    "require_activation_keys",
    "statistical",
]
