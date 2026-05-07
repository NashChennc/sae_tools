from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sae-tools-tui",
        description="Open the SAE tools experiment-management TUI.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root containing Snakefile and configs.")
    parser.add_argument("--config", type=Path, default=None, help="Experiment config to preselect.")
    parser.add_argument("--version", action="version", version=f"sae-tools-tui {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .app import SAEWorkflowTUI
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            print("Textual is not installed. Install the optional TUI dependencies with `pip install -e .[tui]`.", file=sys.stderr)
            return 2
        raise

    app = SAEWorkflowTUI(repo_root=args.repo_root, initial_config=args.config)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
