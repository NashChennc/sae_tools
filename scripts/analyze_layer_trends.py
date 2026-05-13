from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from workflow_common import REPO_ROOT, add_registry_args, load_repo_env, resolve_registry

from sae_tools.experiment_store import ExperimentStore
from sae_tools.experiment_store.manifest import ReportPageRecord
from sae_tools.reporting.html import write_layer_trends_html
from sae_tools.reporting.layout import write_layout
from sae_tools.workflow.artifacts import done_path, experiment_dir, safe_path_part, stat_analysis_dir
from sae_tools.workflow.registry import ExperimentSpec


DEFAULT_METRICS = ("f1", "auroc", "pearson", "diff", "activation_ratio")
SUPPORTED_METRICS = ("f1", "auroc", "pearson", "diff", "activation_ratio", "precision", "recall")
TREND_COLUMNS = (
    "experiment",
    "model",
    "sae",
    "dataset",
    "agg",
    "layer",
    "metric",
    "n_features",
    "top_k",
    "all_mean",
    "all_median",
    "all_std",
    "topk_mean",
    "topk_max",
    "source_path",
)
MISSING_COLUMNS = ("experiment", "model", "sae", "dataset", "agg", "layer", "status", "reason", "path")


@dataclass(frozen=True)
class LayerTrendResult:
    output_dir: Path
    html_path: Path
    markdown_path: Path
    trend_csv: Path
    missing_csv: Path
    summary_json: Path
    plot_paths: tuple[Path, ...]
    trends: pd.DataFrame
    missing: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze statistical metric trends across SAE layers.")
    add_registry_args(parser)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/response_grid.yaml")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Reports directory root. Defaults to artifacts/experiments/<experiment>/reports.",
    )
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated feature-table metrics.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--models", default=None, help="Comma-separated model keys to include.")
    parser.add_argument("--saes", default=None, help="Comma-separated SAE keys to include.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset keys to include.")
    parser.add_argument("--aggs", default=None, help="Comma-separated aggregations to include.")
    parser.add_argument("--layers", default=None, help="Comma-separated integer layers to include.")
    parser.add_argument("--strict", action="store_true", help="Fail if any selected stat artifact is missing.")
    return parser.parse_args()


def generate_layer_trends(
    *,
    config_path: str | Path,
    registry_dir: str | Path,
    artifact_root: str | Path,
    out_root: str | Path | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    top_k: int = 50,
    models: set[str] | None = None,
    saes: set[str] | None = None,
    datasets: set[str] | None = None,
    aggs: set[str] | None = None,
    layers: set[int] | None = None,
    strict: bool = False,
) -> LayerTrendResult:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    metrics = tuple(metrics)
    unknown = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}. Supported metrics: {SUPPORTED_METRICS}")

    registry = resolve_registry(argparse.Namespace(registry_dir=str(registry_dir)))
    experiment = ExperimentSpec.load(config_path, registry)
    experiment_name = Path(config_path).stem
    output_dir = _report_output_dir(
        artifact_root=artifact_root,
        out_root=out_root,
        experiment_name=experiment_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for job in experiment.stat_batch_jobs(registry):
        if not _job_selected(job, models=models, saes=saes, datasets=datasets, aggs=aggs, layers=layers):
            continue
        stat_dir = stat_analysis_dir(
            root=artifact_root,
            experiment=experiment_name,
            model=str(job["model"]),
            sae=str(job["sae"]),
            layer=int(job["layer"]),
            dataset=str(job["dataset"]),
            agg=str(job["agg"]),
        )
        feature_table_path = stat_dir / "feature_table.parquet"
        done_marker = done_path(stat_dir / "DONE")
        status, reason = _artifact_status(feature_table_path, done_marker)
        if status != "done":
            missing_rows.append(_missing_row(experiment_name, job, status, reason, feature_table_path))
            continue

        table = pd.read_parquet(feature_table_path)
        for metric in metrics:
            if metric not in table.columns:
                missing_rows.append(
                    _missing_row(
                        experiment_name,
                        job,
                        "incomplete",
                        f"metric column '{metric}' is missing from feature_table.parquet",
                        feature_table_path,
                    )
                )
                continue
            rows.append(_trend_row(experiment_name, job, metric, table, top_k, feature_table_path))

    trend_df = pd.DataFrame(rows, columns=TREND_COLUMNS).sort_values(
        ["metric", "model", "sae", "dataset", "agg", "layer"],
        ignore_index=True,
    )
    missing_df = pd.DataFrame(missing_rows, columns=MISSING_COLUMNS).sort_values(
        ["status", "model", "sae", "dataset", "agg", "layer"],
        ignore_index=True,
    )

    trend_csv = output_dir / "layer_trends.csv"
    missing_csv = output_dir / "missing_artifacts.csv"
    trend_df.to_csv(trend_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)

    if strict and not missing_df.empty:
        summary_json = _write_summary_json(
            output_dir=output_dir,
            config_path=Path(config_path),
            metrics=metrics,
            top_k=top_k,
            trend_df=trend_df,
            missing_df=missing_df,
            plot_paths=(),
            html_path=None,
            markdown_path=None,
            trend_csv=trend_csv,
            missing_csv=missing_csv,
        )
        raise FileNotFoundError(f"{len(missing_df)} selected stat artifacts are missing or incomplete. See {missing_csv}")

    plot_paths = tuple(_write_plots(output_dir / "plots", trend_df, metrics))
    html_path = output_dir / "layer_trends.html"
    write_layer_trends_html(
        path=html_path,
        experiment_name=experiment_name,
        config_path=Path(config_path),
        metrics=metrics,
        top_k=top_k,
        trend_df=trend_df,
        missing_df=missing_df,
        trend_csv=trend_csv,
        missing_csv=missing_csv,
        plot_paths=plot_paths,
    )
    markdown_path = output_dir / "layer_trends.md"
    _write_markdown_report(
        path=markdown_path,
        experiment_name=experiment_name,
        config_path=Path(config_path),
        metrics=metrics,
        top_k=top_k,
        trend_df=trend_df,
        missing_df=missing_df,
        trend_csv=trend_csv,
        missing_csv=missing_csv,
        plot_paths=plot_paths,
    )
    summary_json = _write_summary_json(
        output_dir=output_dir,
        config_path=Path(config_path),
        metrics=metrics,
        top_k=top_k,
        trend_df=trend_df,
        missing_df=missing_df,
        plot_paths=plot_paths,
        html_path=html_path,
        markdown_path=markdown_path,
        trend_csv=trend_csv,
        missing_csv=missing_csv,
    )
    _record_report_store(
        artifact_root=artifact_root,
        experiment=experiment,
        registry=registry,
        experiment_name=experiment_name,
        output_dir=output_dir,
        html_path=html_path,
        summary_json=summary_json,
        trend_csv=trend_csv,
        missing_csv=missing_csv,
        plot_paths=plot_paths,
    )
    return LayerTrendResult(
        output_dir=output_dir,
        html_path=html_path,
        markdown_path=markdown_path,
        trend_csv=trend_csv,
        missing_csv=missing_csv,
        summary_json=summary_json,
        plot_paths=plot_paths,
        trends=trend_df,
        missing=missing_df,
    )


def _report_output_dir(
    *,
    artifact_root: str | Path,
    out_root: str | Path | None,
    experiment_name: str,
) -> Path:
    reports_root = (
        Path(out_root)
        if out_root is not None
        else experiment_dir(root=artifact_root, experiment=experiment_name) / "reports"
    )
    return reports_root / "pages" / "layer_trends"


def _record_report_store(
    *,
    artifact_root: str | Path,
    experiment: ExperimentSpec,
    registry,
    experiment_name: str,
    output_dir: Path,
    html_path: Path,
    summary_json: Path,
    trend_csv: Path,
    missing_csv: Path,
    plot_paths: Sequence[Path],
) -> None:
    store = ExperimentStore.create(
        artifact_root,
        experiment=experiment,
        registry=registry,
        experiment_id=experiment_name,
        overwrite=False,
    )
    layout_path = write_layout(store.reports_dir / "layout.json")
    for path, role, mime in [
        (html_path, "html", "text/html"),
        (summary_json, "summary", "application/json"),
        (trend_csv, "trend_csv", "text/csv"),
        (missing_csv, "missing_csv", "text/csv"),
    ]:
        store.record_artifact_path(
            kind="report",
            role=f"layer_trends_{role}",
            path=path,
            producer="analyze_layer_trends.py",
            dimensions={"page": "layer_trends", "role": role},
            mime=mime,
        )
    for plot in plot_paths:
        store.record_artifact_path(
            kind="report",
            role=f"layer_trends_plot_{plot.stem}",
            path=plot,
            producer="analyze_layer_trends.py",
            dimensions={"page": "layer_trends", "role": f"plot_{plot.stem}"},
            mime="image/png",
        )
    store.record_report(
        ReportPageRecord(
            page_id="layer_trends",
            experiment_id=experiment_name,
            path=str(_relative_to(html_path, store.root)),
            title="Layer Trend Report",
            layout_path=str(_relative_to(layout_path, store.root)),
        )
    )


def _job_selected(
    job: dict[str, object],
    *,
    models: set[str] | None,
    saes: set[str] | None,
    datasets: set[str] | None,
    aggs: set[str] | None,
    layers: set[int] | None,
) -> bool:
    return (
        (models is None or str(job["model"]) in models)
        and (saes is None or str(job["sae"]) in saes)
        and (datasets is None or str(job["dataset"]) in datasets)
        and (aggs is None or str(job["agg"]) in aggs)
        and (layers is None or int(job["layer"]) in layers)
    )


def _artifact_status(feature_table_path: Path, done_marker: Path) -> tuple[str, str]:
    table_exists = feature_table_path.exists()
    done_exists = done_marker.exists()
    if table_exists and done_exists:
        return "done", "feature_table.parquet and DONE marker exist"
    if table_exists:
        return "incomplete", "feature_table.parquet exists without DONE marker"
    if done_exists:
        return "incomplete", "DONE marker exists without feature_table.parquet"
    return "missing", "feature_table.parquet has not been produced"


def _missing_row(
    experiment_name: str,
    job: dict[str, object],
    status: str,
    reason: str,
    path: Path,
) -> dict[str, object]:
    return {
        "experiment": experiment_name,
        "model": str(job["model"]),
        "sae": str(job["sae"]),
        "dataset": str(job["dataset"]),
        "agg": str(job["agg"]),
        "layer": int(job["layer"]),
        "status": status,
        "reason": reason,
        "path": str(path),
    }


def _trend_row(
    experiment_name: str,
    job: dict[str, object],
    metric: str,
    table: pd.DataFrame,
    top_k: int,
    feature_table_path: Path,
) -> dict[str, object]:
    values = pd.to_numeric(table[metric], errors="coerce").dropna()
    n_features = int(values.shape[0])
    top_values = values.nlargest(min(top_k, n_features)) if n_features else values
    return {
        "experiment": experiment_name,
        "model": str(job["model"]),
        "sae": str(job["sae"]),
        "dataset": str(job["dataset"]),
        "agg": str(job["agg"]),
        "layer": int(job["layer"]),
        "metric": metric,
        "n_features": n_features,
        "top_k": min(top_k, n_features),
        "all_mean": _float_or_none(values.mean()),
        "all_median": _float_or_none(values.median()),
        "all_std": _float_or_none(values.std(ddof=0)),
        "topk_mean": _float_or_none(top_values.mean()),
        "topk_max": _float_or_none(top_values.max()),
        "source_path": str(feature_table_path),
    }


def _write_plots(plot_dir: Path, trend_df: pd.DataFrame, metrics: Sequence[str]) -> list[Path]:
    if trend_df.empty:
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric in metrics:
        metric_df = trend_df[trend_df["metric"] == metric]
        if metric_df.empty:
            continue
        path = plot_dir / f"{safe_path_part(metric, field='metric')}.png"
        fig, ax = plt.subplots(figsize=(11, 6))
        for (model, sae, dataset, agg), group in metric_df.groupby(["model", "sae", "dataset", "agg"], sort=True):
            group = group.sort_values("layer")
            label = f"{model} | {sae} | {dataset} | {agg}"
            ax.plot(group["layer"], group["topk_mean"], marker="o", linewidth=1.7, markersize=4, label=label)
        ax.set_title(f"{metric} top-k mean across layers")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-k mean")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _write_markdown_report(
    *,
    path: Path,
    experiment_name: str,
    config_path: Path,
    metrics: Sequence[str],
    top_k: int,
    trend_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    trend_csv: Path,
    missing_csv: Path,
    plot_paths: Sequence[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_by_metric = {plot.stem: plot for plot in plot_paths}
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Layer Trend Report: {experiment_name}\n\n")
        handle.write(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"- Config: `{config_path}`\n")
        handle.write(f"- Metrics: `{', '.join(metrics)}`\n")
        handle.write(f"- Top-k: `{top_k}`\n")
        handle.write(f"- Trend rows: `{len(trend_df)}`\n")
        handle.write(f"- Missing or incomplete artifacts: `{len(missing_df)}`\n\n")
        handle.write("## Best Layers\n\n")
        handle.write(_best_layers_table(trend_df, metrics))
        handle.write("\n## Plots\n\n")
        if not plot_paths:
            handle.write("No plots were generated because no trend rows were available.\n\n")
        for metric in metrics:
            plot = plot_by_metric.get(metric)
            if plot is None:
                handle.write(f"### {metric}\n\nNo data available.\n\n")
                continue
            rel = _relative_link(plot, path.parent)
            handle.write(f"### {metric}\n\n![{metric}]({rel})\n\n")
        handle.write("## Missing Artifacts\n\n")
        if missing_df.empty:
            handle.write("No selected stat artifacts were missing or incomplete.\n\n")
        else:
            counts = missing_df["status"].value_counts().to_dict()
            count_text = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
            handle.write(f"Missing/incomplete summary: {count_text}.\n\n")
        handle.write("## Data Files\n\n")
        handle.write(f"- [Layer trends CSV]({_relative_link(trend_csv, path.parent)})\n")
        handle.write(f"- [Missing artifacts CSV]({_relative_link(missing_csv, path.parent)})\n")


def _best_layers_table(trend_df: pd.DataFrame, metrics: Sequence[str]) -> str:
    header = "| metric | best_layer | topk_mean | model | sae | dataset | agg |\n"
    separator = "| --- | ---: | ---: | --- | --- | --- | --- |\n"
    lines = [header, separator]
    if trend_df.empty:
        lines.append("| - | - | - | - | - | - | - |\n")
        return "".join(lines)
    for metric in metrics:
        metric_df = trend_df[trend_df["metric"] == metric].copy()
        if metric_df.empty:
            lines.append(f"| {metric} | - | - | - | - | - | - |\n")
            continue
        idx = pd.to_numeric(metric_df["topk_mean"], errors="coerce").idxmax()
        row = metric_df.loc[idx]
        lines.append(
            f"| {metric} | {int(row['layer'])} | {_format_float(row['topk_mean'])} | "
            f"{row['model']} | {row['sae']} | {row['dataset']} | {row['agg']} |\n"
        )
    return "".join(lines)


def _write_summary_json(
    *,
    output_dir: Path,
    config_path: Path,
    metrics: Sequence[str],
    top_k: int,
    trend_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    plot_paths: Sequence[Path],
    html_path: Path | None,
    markdown_path: Path | None,
    trend_csv: Path,
    missing_csv: Path,
) -> Path:
    path = output_dir / "summary.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "metrics": list(metrics),
        "top_k": top_k,
        "trend_rows": int(len(trend_df)),
        "missing_artifacts": int(len(missing_df)),
        "outputs": {
            "html": None if html_path is None else str(html_path),
            "markdown": None if markdown_path is None else str(markdown_path),
            "layer_trends_csv": str(trend_csv),
            "missing_artifacts_csv": str(missing_csv),
            "plots": [str(path) for path in plot_paths],
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _parse_csv(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_layers(value: str | None) -> set[int] | None:
    parsed = _parse_csv(value)
    return None if parsed is None else {int(item) for item in parsed}


def _parse_metrics(value: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in value.split(",") if item.strip())
    if not metrics:
        raise ValueError("At least one metric is required.")
    return metrics


def _float_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.6g}"


def _relative_link(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, start=base_dir).replace(os.sep, "/")


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def main() -> None:
    args = parse_args()
    load_repo_env()
    result = generate_layer_trends(
        config_path=args.config,
        registry_dir=args.registry_dir,
        artifact_root=args.artifact_root,
        out_root=args.out_root,
        metrics=_parse_metrics(args.metrics),
        top_k=args.top_k,
        models=_parse_csv(args.models),
        saes=_parse_csv(args.saes),
        datasets=_parse_csv(args.datasets),
        aggs=_parse_csv(args.aggs),
        layers=_parse_layers(args.layers),
        strict=args.strict,
    )
    print(f"HTML {result.html_path}")
    print(f"REPORT {result.markdown_path}")
    print(f"CSV {result.trend_csv}")
    print(f"MISSING {result.missing_csv}")
    print(f"SUMMARY {result.summary_json}")


if __name__ == "__main__":
    main()
