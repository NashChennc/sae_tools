from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

from workflow_common import REPO_ROOT, add_registry_args, load_dataset_from_spec, load_repo_env, require_env, resolve_max_samples, resolve_registry

from sae_tools.analysis.statistical import build_sentence_feature_matrix_from_sparse, evaluate_features
from sae_tools.model import filter_data_by_label, load_sae_predictions_pt
from sae_tools.workflow.artifacts import artifact_meta_path, build_artifact_meta, mark_done, write_json_atomic


LABEL_TO_BINARY = {"safe": 0, "unsafe": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one statistical analysis artifact.")
    add_registry_args(parser)
    parser.add_argument("--acts", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="Dataset registry key.")
    parser.add_argument("--agg", required=True, choices=["max", "mean"])
    parser.add_argument("--metric", required=True, choices=["pearson", "auroc", "f1", "precision", "recall", "diff"])
    parser.add_argument("--out", required=True, type=Path)
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
    if args.out.exists() and not args.overwrite:
        print(f"READY {args.out}")
        return

    load_repo_env()
    registry = resolve_registry(args)
    dataset_spec = registry.dataset(args.dataset)
    activation_meta = _read_activation_meta(args.acts)
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
    payload = _metric_payload(args.metric, X, y, args.top_k, args.batch_size)
    payload.update(
        {
            "dataset": args.dataset,
            "label_field": label_field,
            "aggregation": args.agg,
            "metric": args.metric,
            "n_labeled": int(len(labeled_rows)),
            "n_valid_indices": int(len(valid_indices)),
        }
    )
    write_json_atomic(args.out, payload)
    meta = build_artifact_meta(
        script="analyze_stat.py",
        repo_dir=REPO_ROOT,
        params={
            "acts": str(args.acts),
            "dataset": args.dataset,
            "agg": args.agg,
            "metric": args.metric,
            "label_field": label_field,
            "max_samples": max_samples,
            "top_k": args.top_k,
            "output": str(args.out),
        },
        inputs={"activation_meta": activation_meta},
    )
    write_json_atomic(artifact_meta_path(args.out), meta)
    mark_done(args.out)
    print(f"DONE {args.out}")


if __name__ == "__main__":
    main()
