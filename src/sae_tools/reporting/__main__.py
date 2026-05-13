from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sae-tools-report",
        description="Serve SAE tools HTML reports and the read-only dashboard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the read-only HTML report server.")
    serve.add_argument("--repo-root", type=Path, default=None, help="Repository root containing configs and reports.")
    serve.add_argument("--config", type=Path, default=Path("configs/experiments/response_grid.yaml"))
    serve.add_argument("--registry-dir", type=Path, default=None)
    serve.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    serve.add_argument("--report-root", type=Path, default=Path("report"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .server import ReportServerConfig, serve_report

    config = ReportServerConfig.from_paths(
        repo_root=args.repo_root,
        config_path=args.config,
        registry_dir=args.registry_dir,
        artifact_root=args.artifact_root,
        report_root=args.report_root,
        host=args.host,
        port=args.port,
    )
    load_dotenv(config.repo_root / ".env")
    if args.command == "serve":
        serve_report(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
