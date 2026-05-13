from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from workflow_common import REPO_ROOT, add_registry_args, load_dataset_from_spec, load_repo_env, require_env, resolve_max_samples, resolve_registry

from sae_tools.analysis.statistical import build_sentence_feature_matrix_from_sparse, evaluate_features
from sae_tools.experiment_store import ExperimentStore
from sae_tools.model import filter_data_by_label, load_sae_predictions_pt
from sae_tools.workflow.artifacts import (
    artifact_meta_path,
    build_artifact_meta,
    mark_done,
    write_json_atomic,
)


LABEL_TO_BINARY = {"safe": 0, "unsafe": 1}
SUPPORTED_METRICS = ("pearson", "auroc", "f1", "precision", "recall", "diff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical SAE analysis artifacts.")
    add_registry_args(parser)
    parser.add_argument("--acts", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="Dataset registry key.")
    parser.add_argument("--agg", required=True, choices=["max", "mean"])
    parser.add_argument("--metric", choices=SUPPORTED_METRICS, default=None)
    parser.add_argument("--metrics", default=None, help="Comma-separated metrics for batch mode.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_activation_meta(path: Path) -> dict:
    meta_path = artifact_meta_path(path)
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _meta_max_samples(meta: dict) -> int | None:
    params = meta.get("params", {})
    value = params.get("max_samples", meta.get("max_samples"))
    return None if value is None else int(value)


def _parse_metrics(args: argparse.Namespace) -> list[str]:
    if args.metrics:
        metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    elif args.metric:
        metrics = [args.metric]
    else:
        raise ValueError("Provide either --metric or --metrics.")
    unknown = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}. Supported metrics: {SUPPORTED_METRICS}")
    return metrics


def _mean_reduce(values, reduce_indices, sample_ids, feat_indices):
    del sample_ids, feat_indices
    counts = np.diff(np.append(reduce_indices, len(values)))
    return np.add.reduceat(values, reduce_indices) / counts


def _feature_matrix(sparse_data: dict, agg: str) -> sp.csr_matrix:
    if agg == "max":
        return build_sentence_feature_matrix_from_sparse(sparse_data)
    if agg == "mean":
        return build_sentence_feature_matrix_from_sparse(sparse_data, reduce_fn=_mean_reduce)
    raise ValueError(f"Unsupported aggregation: {agg}")


def _labels(rows: list[dict], field: str) -> np.ndarray:
    labels = []
    for row in rows:
        label = row.get(field)
        key = str(label).lower()
        if key not in LABEL_TO_BINARY:
            raise ValueError(f"Unsupported label {label!r} in field '{field}'. Expected Safe/Unsafe.")
        labels.append(LABEL_TO_BINARY[key])
    if not labels:
        raise ValueError(f"No labeled rows found for field '{field}'.")
    return np.array(labels, dtype=np.int64)


def _pearson_scores(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    X = X.tocsr().astype(np.float64)
    y = y.astype(np.float64)
    n = X.shape[0]
    y_centered = y - y.mean()
    y_ss = float(np.dot(y_centered, y_centered))
    sum_x = np.asarray(X.sum(axis=0)).ravel()
    sum_x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()
    mean_x = sum_x / n
    x_ss = np.maximum(sum_x2 - n * mean_x * mean_x, 0.0)
    numerator = np.asarray(X.T.dot(y_centered)).ravel()
    denom = np.sqrt(x_ss * y_ss)
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)


def _auroc_scores(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    X = X.tocsc()
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(X.shape[1], 0.5, dtype=np.float64)

    scores = np.zeros(X.shape[1], dtype=np.float64)
    for feature_idx in range(X.shape[1]):
        values = X[:, feature_idx].toarray().ravel()
        ranks = rankdata(values, method="average")
        rank_sum_pos = ranks[pos].sum()
        scores[feature_idx] = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return scores


def _top_records(values: np.ndarray, top_k: int, extra: dict[str, np.ndarray] | None = None) -> list[dict]:
    if values.size == 0:
        return []
    k = min(top_k, values.size)
    top = np.argpartition(values, -k)[-k:]
    top = top[np.argsort(values[top])[::-1]]
    records = []
    for idx in top:
        record = {"feature": int(idx), "score": float(values[idx])}
        for key, array in (extra or {}).items():
            record[key] = float(array[idx])
        records.append(record)
    return records


def _compute_feature_table(
    X: sp.csr_matrix,
    y: np.ndarray,
    *,
    top_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    result = evaluate_features(X, y, top_k=top_k, batch_size=batch_size)
    pearson = _pearson_scores(X, y)
    auroc = _auroc_scores(X, y)
    table = pd.DataFrame(
        {
            "feature": result.feature_indices.astype(np.int64),
            "precision": result.precisions.astype(np.float64),
            "recall": result.recalls.astype(np.float64),
            "f1": result.f1_scores.astype(np.float64),
            "activation_ratio": result.activation_ratios.astype(np.float64),
            "diff": result.feature_diff.astype(np.float64),
            "pearson": pearson.astype(np.float64),
            "auroc": auroc.astype(np.float64),
        }
    )
    details = {
        "stats": result.stats,
        "separation_score": result.separation_score,
        "pareto_front_ids": [int(item) for item in result.pareto_front_ids],
    }
    return table, details


def _top_records_from_table(table: pd.DataFrame, metric: str, top_k: int) -> list[dict]:
    if table.empty:
        return []
    columns = ["feature", metric, "precision", "recall", "f1", "activation_ratio", "diff", "pearson", "auroc"]
    records = table.nlargest(min(top_k, len(table)), metric)[columns].to_dict(orient="records")
    for record in records:
        record["feature"] = int(record["feature"])
        record["score"] = float(record[metric])
        for key, value in list(record.items()):
            if key != "feature":
                record[key] = float(value)
    return records


def _metric_payload_from_table(table: pd.DataFrame, metric: str, top_k: int, details: dict) -> dict:
    return {
        "top_features": _top_records_from_table(table, metric, top_k),
        "stats": details["stats"],
        "pareto_front_ids": details["pareto_front_ids"][:top_k],
        "separation_score": details["separation_score"],
    }


def _write_parquet_atomic(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def _plot_pr_space(table: pd.DataFrame, output_file: Path, *, color_by: str, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if color_by == "diff":
        color_data = table["diff"]
        color_label = "Normalized Feature Difference"
        vmin, vmax = None, None
    elif color_by == "ratio":
        color_data = table["activation_ratio"]
        color_label = "Activation Ratio"
        vmin, vmax = 0, 1
    else:
        raise ValueError(f"Unsupported PR plot color: {color_by}")

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        x=table["recall"],
        y=table["precision"],
        c=color_data,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=20,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12, rotation=270, labelpad=20)
    ax.set_xlabel("Recall", fontsize=14, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal", "box")
    stats_text = (
        f"Total Features: {len(table)}\n"
        f"F1 Mean: {table['f1'].mean():.4f}\n"
        f"F1 Max: {table['f1'].max():.4f}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _metric_payload(metric: str, X: sp.csr_matrix, y: np.ndarray, top_k: int, batch_size: int) -> dict:
    if metric == "pearson":
        values = _pearson_scores(X, y)
        return {"top_features": _top_records(values, top_k), "stats": {"n_features": int(values.size)}}
    if metric == "auroc":
        values = _auroc_scores(X, y)
        return {"top_features": _top_records(values, top_k), "stats": {"n_features": int(values.size)}}

    result = evaluate_features(X, y, top_k=top_k, batch_size=batch_size)
    values_by_metric = {
        "f1": result.f1_scores,
        "precision": result.precisions,
        "recall": result.recalls,
        "diff": result.feature_diff,
    }
    values = values_by_metric[metric]
    extra = {
        "precision": result.precisions,
        "recall": result.recalls,
        "f1": result.f1_scores,
        "activation_ratio": result.activation_ratios,
        "diff": result.feature_diff,
    }
    return {
        "top_features": _top_records(values, top_k, extra=extra),
        "stats": result.stats,
        "pareto_front_ids": result.pareto_front_ids[:top_k],
        "separation_score": result.separation_score,
    }


def main() -> None:
    args = parse_args()
    metrics = _parse_metrics(args)
    batch_mode = args.out_dir is not None
    if not batch_mode and args.out is None:
        raise ValueError("Single metric mode requires --out. Batch mode requires --out-dir.")
    if batch_mode and (args.out_dir / "DONE").exists() and not args.overwrite:
        print(f"READY {args.out_dir}")
        return
    if not batch_mode and args.out and args.out.exists() and not args.overwrite:
        print(f"READY {args.out}")
        return

    load_repo_env()
    registry = resolve_registry(args)
    dataset_spec = registry.dataset(args.dataset)
    activation_meta = _read_activation_meta(args.acts)
    activation_params = activation_meta.get("params", {})
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = _meta_max_samples(activation_meta)
    max_samples = resolve_max_samples(max_samples, dataset_spec)
    label_field = args.label_field or dataset_spec.label_field

    sparse_data = load_sae_predictions_pt(str(args.acts))
    dataset = load_dataset_from_spec(
        dataset_spec,
        dataset_root=require_env("DATASET_ROOT"),
        max_samples=max_samples,
    )
    metadata_rows = [dict(item) for item in dataset]
    sparse_data, labeled_rows, valid_indices = filter_data_by_label(
        sparse_data,
        metadata_rows,
        label_field=label_field,
        verbose=True,
    )
    y = _labels(labeled_rows, label_field)
    X = _feature_matrix(sparse_data, args.agg)
    table, details = _compute_feature_table(X, y, top_k=args.top_k, batch_size=args.batch_size)

    common = {
        "model": activation_params.get("model"),
        "sae": activation_params.get("sae"),
        "layer": activation_params.get("layer"),
        "dataset": args.dataset,
        "label_field": label_field,
        "aggregation": args.agg,
        "max_samples": max_samples,
        "n_labeled": int(len(labeled_rows)),
        "n_valid_indices": int(len(valid_indices)),
        "top_k": args.top_k,
    }

    if batch_mode:
        assert args.out_dir is not None
        out_dir = args.out_dir
        feature_table_path = out_dir / "feature_table.parquet"
        summary_path = out_dir / "summary.json"
        top_features_path = out_dir / "top_features.json"
        pareto_path = out_dir / "pareto_front.json"
        _write_parquet_atomic(table, feature_table_path)
        write_json_atomic(
            summary_path,
            {
                **common,
                "metrics": metrics,
                "stats": details["stats"],
                "separation_score": details["separation_score"],
                "feature_table": str(feature_table_path),
            },
        )
        write_json_atomic(
            top_features_path,
            {metric: _top_records_from_table(table, metric, args.top_k) for metric in metrics},
        )
        write_json_atomic(pareto_path, {"feature_ids": details["pareto_front_ids"]})

        for metric in metrics:
            metric_path = out_dir / f"metric={metric}" / "metrics.json"
            payload = _metric_payload_from_table(table, metric, args.top_k, details)
            payload.update({**common, "metric": metric})
            write_json_atomic(metric_path, payload)
            write_json_atomic(
                artifact_meta_path(metric_path),
                build_artifact_meta(
                    script="analyze_stat.py",
                    repo_dir=REPO_ROOT,
                    params={**common, "metric": metric, "output": str(metric_path)},
                    inputs={"activation_meta": activation_meta},
                ),
            )
            mark_done(metric_path)

        plot_paths: list[Path] = []
        if common.get("model") and common.get("sae") and common.get("layer") is not None:
            for color in ("diff", "ratio"):
                plot_path = out_dir / "plots" / f"pr_space.color={color}.png"
                _plot_pr_space(
                    table,
                    plot_path,
                    color_by=color,
                    title=f"{common['model']} {args.dataset} {args.agg} PR Space",
                )
                plot_paths.extend([plot_path, plot_path.with_suffix(".pdf")])

        write_json_atomic(
            artifact_meta_path(summary_path),
            build_artifact_meta(
                script="analyze_stat.py",
                repo_dir=REPO_ROOT,
                params={**common, "metrics": metrics, "out_dir": str(out_dir)},
                inputs={"activation_meta": activation_meta},
            ),
        )
        mark_done(summary_path)
        _record_store_outputs(
            out_dir=out_dir,
            table=table,
            common=common,
            metrics=metrics,
            artifact_paths=[
                feature_table_path,
                summary_path,
                top_features_path,
                pareto_path,
                *(out_dir / f"metric={metric}" / "metrics.json" for metric in metrics),
                *plot_paths,
            ],
        )
        print(f"DONE {out_dir}")
        return

    assert args.out is not None
    metric = metrics[0]
    out_dir = args.out.parent.parent if args.out.parent.name.startswith("metric=") else args.out.parent
    payload = _metric_payload_from_table(table, metric, args.top_k, details)
    payload.update({**common, "metric": metric})
    write_json_atomic(args.out, payload)
    _write_parquet_atomic(table, out_dir / "feature_table.parquet")
    write_json_atomic(out_dir / "summary.json", {**common, "metrics": metrics, "stats": details["stats"]})
    write_json_atomic(out_dir / "top_features.json", {metric: _top_records_from_table(table, metric, args.top_k)})
    write_json_atomic(out_dir / "pareto_front.json", {"feature_ids": details["pareto_front_ids"]})
    write_json_atomic(
        artifact_meta_path(args.out),
        build_artifact_meta(
            script="analyze_stat.py",
            repo_dir=REPO_ROOT,
            params={**common, "metric": metric, "output": str(args.out)},
            inputs={"activation_meta": activation_meta},
        ),
    )
    mark_done(args.out)
    _record_store_outputs(
        out_dir=out_dir,
        table=table,
        common=common,
        metrics=metrics,
        artifact_paths=[
            args.out,
            out_dir / "feature_table.parquet",
            out_dir / "summary.json",
            out_dir / "top_features.json",
            out_dir / "pareto_front.json",
        ],
    )
    print(f"DONE {args.out}")


def _record_store_outputs(
    *,
    out_dir: Path,
    table: pd.DataFrame,
    common: dict[str, object],
    metrics: list[str],
    artifact_paths: list[Path],
) -> None:
    store = ExperimentStore.from_artifact_path(out_dir)
    if store is None:
        return
    model = common.get("model")
    sae = common.get("sae")
    layer = common.get("layer")
    dataset = common.get("dataset")
    agg = common.get("aggregation")
    if not model or not sae or layer is None or not dataset or not agg:
        return

    store.ensure_initialized()
    dimensions = {
        "model": str(model),
        "sae": str(sae),
        "layer": int(layer),
        "dataset": str(dataset),
        "agg": str(agg),
    }
    store.replace_stat_features(table, **dimensions)
    for path in artifact_paths:
        role = path.stem
        if path.name == "metrics.json" and path.parent.name.startswith("metric="):
            role = path.parent.name
        role_id = role.replace("=", "-").replace(",", "-")
        store.record_artifact_path(
            kind="stat",
            role=role,
            path=path,
            producer="analyze_stat.py",
            dimensions={**dimensions, "role": role_id},
            mime=_mime_for_path(path),
        )


def _mime_for_path(path: Path) -> str | None:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    if path.suffix == ".png":
        return "image/png"
    if path.suffix == ".pdf":
        return "application/pdf"
    return None


if __name__ == "__main__":
    main()
