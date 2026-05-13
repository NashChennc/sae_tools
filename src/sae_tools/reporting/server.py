from __future__ import annotations

import json
import os
import traceback
import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from sae_tools.adapters.datasets import get_adapter
from sae_tools.model import load_sae_predictions_pt
from sae_tools.workflow.artifacts import activation_path, done_path, experiment_dir, safe_path_part, stat_analysis_dir
from sae_tools.workflow.registry import ExperimentSpec, Registry
from sae_tools.workflow.runtime import ArtifactRecord, default_repo_root, scan_artifacts, scan_experiments

from .html import escape, html_page, render_table, status_badge


@dataclass(frozen=True)
class ReportServerConfig:
    repo_root: Path
    config_path: Path
    registry_dir: Path
    artifact_root: Path
    report_root: Path
    host: str = "127.0.0.1"
    port: int = 8765

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: str | Path | None = None,
        config_path: str | Path = "configs/experiments/response_grid.yaml",
        registry_dir: str | Path | None = None,
        artifact_root: str | Path = "artifacts",
        report_root: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> "ReportServerConfig":
        root = default_repo_root() if repo_root is None else Path(repo_root)
        config_resolved = _resolve_under(root, config_path)
        artifact_resolved = _resolve_under(root, artifact_root)
        report_resolved = (
            _resolve_under(root, report_root)
            if report_root is not None
            else experiment_dir(root=artifact_resolved, experiment=config_resolved.stem) / "reports"
        )
        return cls(
            repo_root=root.resolve(),
            config_path=config_resolved,
            registry_dir=_resolve_under(root, registry_dir or "configs/registry"),
            artifact_root=artifact_resolved,
            report_root=report_resolved,
            host=host,
            port=int(port),
        )


@dataclass(frozen=True)
class Selection:
    model: str
    sae: str
    layer: int
    dataset: str
    agg: str


class ReportData:
    def __init__(self, config: ReportServerConfig, env: Mapping[str, str] | None = None) -> None:
        self.config = config
        self.env = os.environ if env is None else env

    @property
    def experiment_name(self) -> str:
        return self.config.config_path.stem

    @lru_cache(maxsize=8)
    def registry_and_experiment(self, config_path: str) -> tuple[Registry, ExperimentSpec]:
        registry = Registry.load(self.config.registry_dir)
        experiment = ExperimentSpec.load(config_path, registry)
        return registry, experiment

    def registry(self) -> Registry:
        registry, _experiment = self.registry_and_experiment(str(self.config.config_path))
        return registry

    def experiment(self) -> ExperimentSpec:
        _registry, experiment = self.registry_and_experiment(str(self.config.config_path))
        return experiment

    def stat_jobs(self) -> list[dict[str, Any]]:
        registry, experiment = self.registry_and_experiment(str(self.config.config_path))
        return experiment.stat_batch_jobs(registry)

    def first_selection(self) -> Selection | None:
        for job in self.stat_jobs():
            return Selection(
                model=str(job["model"]),
                sae=str(job["sae"]),
                layer=int(job["layer"]),
                dataset=str(job["dataset"]),
                agg=str(job["agg"]),
            )
        return None

    def selection_from_query(self, query: Mapping[str, Sequence[str]]) -> Selection:
        default = self.first_selection()
        if default is None:
            raise ValueError(f"Experiment has no statistical jobs: {self.config.config_path}")
        return Selection(
            model=_query_one(query, "model", default.model),
            sae=_query_one(query, "sae", default.sae),
            layer=int(_query_one(query, "layer", str(default.layer))),
            dataset=_query_one(query, "dataset", default.dataset),
            agg=_query_one(query, "agg", default.agg),
        )

    def dashboard_options(self) -> dict[str, list[str]]:
        jobs = self.stat_jobs()
        options = {
            "models": sorted({str(job["model"]) for job in jobs}),
            "saes": sorted({str(job["sae"]) for job in jobs}),
            "layers": sorted({str(job["layer"]) for job in jobs}, key=lambda item: int(item)),
            "datasets": sorted({str(job["dataset"]) for job in jobs}),
            "aggs": sorted({str(job["agg"]) for job in jobs}),
            "metrics": sorted({str(metric) for job in jobs for metric in job.get("metrics", ())}),
        }
        if not options["metrics"]:
            options["metrics"] = ["f1", "auroc", "pearson"]
        return options

    def experiment_summaries(self) -> list[dict[str, object]]:
        summaries = scan_experiments(
            self.config.repo_root / "configs/experiments",
            self.config.registry_dir,
            repo_root=self.config.repo_root,
            artifact_root=self.config.artifact_root,
        )
        return [
            {
                "experiment": summary.name,
                "models": summary.models,
                "saes": summary.saes,
                "layers": summary.layers,
                "datasets": summary.datasets,
                "targets": summary.activation_targets + summary.stat_targets + summary.geometric_targets,
                "done": summary.done,
                "missing": summary.missing,
                "incomplete": summary.incomplete,
                "failed": summary.failed,
                "error": summary.error,
            }
            for summary in summaries
        ]

    def artifact_rows(self) -> list[dict[str, object]]:
        records = scan_artifacts(
            self.config.config_path,
            self.config.registry_dir,
            repo_root=self.config.repo_root,
            artifact_root=self.config.artifact_root,
        )
        rows: list[dict[str, object]] = []
        for stage, stage_records in records.items():
            for record in stage_records:
                rows.append(_artifact_row(stage, record))
        return rows

    def features(self, selection: Selection, *, metric: str, top_k: int) -> dict[str, Any]:
        _validate_selection(selection)
        metric = safe_path_part(metric, field="metric")
        stat_dir = stat_analysis_dir(
            root=self.config.artifact_root,
            experiment=self.experiment_name,
            model=selection.model,
            sae=selection.sae,
            layer=selection.layer,
            dataset=selection.dataset,
            agg=selection.agg,
        )
        top_features_path = stat_dir / "top_features.json"
        if top_features_path.exists():
            with top_features_path.open("r", encoding="utf-8") as handle:
                top_features = json.load(handle)
            records = list(top_features.get(metric, []))[:top_k]
        else:
            table_path = stat_dir / "feature_table.parquet"
            if not table_path.exists():
                raise FileNotFoundError(f"Stat feature table not found: {table_path}")
            table = pd.read_parquet(table_path)
            if metric not in table.columns:
                raise ValueError(f"Metric column not found in feature table: {metric}")
            records = table.nlargest(min(top_k, len(table)), metric).to_dict(orient="records")
            for record in records:
                record["feature"] = int(record["feature"])
                record["score"] = float(record[metric])
        return {
            "selection": selection.__dict__,
            "metric": metric,
            "top_k": top_k,
            "features": _json_safe(records),
        }

    def feature_activations(
        self,
        selection: Selection,
        *,
        feature: int,
        threshold: float,
        max_display: int,
        show_full_text: bool = False,
    ) -> dict[str, Any]:
        viewer = self._viewer(selection)
        html = viewer.render_feature_activations_html(
            feature,
            threshold=threshold,
            max_display=max_display,
            show_full_text=show_full_text,
        )
        return {
            "selection": selection.__dict__,
            "feature": int(feature),
            "threshold": threshold,
            "max_display": max_display,
            "html": html,
        }

    @lru_cache(maxsize=8)
    def _viewer(self, selection: Selection):
        from transformers import AutoTokenizer

        from sae_tools.analysis.dashboard.viewer import FeatureActivationViewer

        registry, experiment = self.registry_and_experiment(str(self.config.config_path))
        dataset = registry.dataset(selection.dataset)
        max_samples = experiment.dataset_max_samples(registry, selection.dataset)
        acts_path = activation_path(
            root=self.config.artifact_root,
            experiment=self.experiment_name,
            model=selection.model,
            sae=selection.sae,
            layer=selection.layer,
            dataset=selection.dataset,
            split=dataset.split,
            max_samples=max_samples,
        )
        if not acts_path.exists() or not done_path(acts_path).exists():
            raise FileNotFoundError(f"Activation artifact is not ready: {acts_path}")
        sparse_data = load_sae_predictions_pt(str(acts_path))
        num_samples = len(sparse_data["seq_lens"])

        dataset_root = self.env.get("DATASET_ROOT")
        if not dataset_root:
            raise EnvironmentError("DATASET_ROOT is not set; feature activation dashboard cannot load metadata.")
        dataset_path = Path(dataset.folder)
        if not dataset_path.is_absolute():
            dataset_path = Path(dataset_root) / dataset_path
        adapter = get_adapter(dataset.adapter)
        rows = adapter.load(str(dataset_path), num_samples, split=dataset.split, subset=dataset.subset)
        metadata = [dict(item) for item in rows]

        model_root = self.env.get("MODEL_ROOT")
        if not model_root:
            raise EnvironmentError("MODEL_ROOT is not set; feature activation dashboard cannot load tokenizer.")
        model = registry.model(selection.model)
        model_path = Path(model.local_path)
        if not model_path.is_absolute():
            model_path = Path(model_root) / model_path
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        return FeatureActivationViewer(sparse_data, metadata, tokenizer)


def serve_report(config: ReportServerConfig) -> None:
    data = ReportData(config)
    handler = _make_handler(config, data)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"Serving SAE tools report at http://{config.host}:{config.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SAE tools report server.")
    finally:
        server.server_close()


def render_index(config: ReportServerConfig, data: ReportData) -> str:
    experiment_name = data.experiment_name
    html_report = config.report_root / "pages" / "layer_trends" / "layer_trends.html"
    artifact_report = config.report_root / "pages" / "artifacts" / "artifacts.html"
    report_link = f"/layer-trends/{urllib.parse.quote(experiment_name)}/" if html_report.exists() else "#"
    report_status = "done" if html_report.exists() else "missing"
    artifact_link = f"/artifact-plots/{urllib.parse.quote(experiment_name)}/" if artifact_report.exists() else "#"
    artifact_status = "done" if artifact_report.exists() else "missing"
    rows = data.experiment_summaries()
    body = f"""
<section class="grid">
  <section class="card"><div class="label">Selected experiment</div><div class="value">{escape(experiment_name)}</div></section>
  <section class="card"><div class="label">HTML report</div><div class="value">{status_badge(report_status)}</div></section>
  <section class="card"><div class="label">Artifact plot report</div><div class="value">{status_badge(artifact_status)}</div></section>
  <section class="card"><div class="label">Store reports</div><div class="value">{escape(config.report_root)}</div></section>
</section>
<section class="panel">
  <h2>Selected Report</h2>
  <div class="links">
    <a href="{report_link}">Open layer trend report</a>
    <a href="{artifact_link}">Open artifact plot report</a>
    <a href="/dashboard">Open dashboard</a>
  </div>
</section>
<section class="panel">
  <h2>Experiments</h2>
  {render_table(rows, (("experiment", "experiment"), ("models", "models"), ("saes", "SAEs"), ("layers", "layers"), ("datasets", "datasets"), ("targets", "targets"), ("done", "done"), ("missing", "missing"), ("incomplete", "incomplete"), ("failed", "failed"), ("error", "error")), numeric={"models", "saes", "layers", "datasets", "targets", "done", "missing", "incomplete", "failed"})}
</section>
"""
    return html_page(title="SAE Tools Reports", subtitle=str(config.config_path), body=body)


def render_dashboard(config: ReportServerConfig, data: ReportData) -> str:
    options = data.dashboard_options()
    first = data.first_selection()
    artifact_rows = data.artifact_rows()[:500]
    controls = _controls_html(options, first)
    scripts = _dashboard_scripts()
    body = f"""
<section class="panel">
  <h2>Feature Explorer</h2>
  {controls}
  <div id="feature-error" class="error" style="display: none;"></div>
  <h3>Top Features</h3>
  <div id="features" class="muted">Choose filters and load features.</div>
  <h3>Feature Activations</h3>
  <div id="activation" class="muted">Select a feature to inspect activation contexts.</div>
</section>
<section class="panel">
  <h2>Artifact Status</h2>
  {render_table(artifact_rows, (("stage", "stage"), ("status", "status"), ("target", "target"), ("log", "log"), ("reason", "reason")))}
</section>
"""
    return html_page(title="SAE Tools Dashboard", subtitle=str(config.config_path), body=body, scripts=scripts)


def _make_handler(config: ReportServerConfig, data: ReportData):
    class ReportRequestHandler(BaseHTTPRequestHandler):
        server_version = "SAEToolsReport/0.1"

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path in {"", "/", "/index.html"}:
                    self._send_html(render_index(config, data))
                elif parsed.path == "/dashboard":
                    self._send_html(render_dashboard(config, data))
                elif parsed.path == "/api/features":
                    selection = data.selection_from_query(query)
                    metric = _query_one(query, "metric", "f1")
                    top_k = int(_query_one(query, "top_k", "50"))
                    self._send_json(data.features(selection, metric=metric, top_k=top_k))
                elif parsed.path == "/api/feature-activations":
                    selection = data.selection_from_query(query)
                    feature = int(_query_one(query, "feature", "0"))
                    threshold = float(_query_one(query, "threshold", "0"))
                    max_display = int(_query_one(query, "max_display", "10"))
                    show_full_text = _query_one(query, "show_full_text", "false").lower() in {"1", "true", "yes"}
                    self._send_json(
                        data.feature_activations(
                            selection,
                            feature=feature,
                            threshold=threshold,
                            max_display=max_display,
                            show_full_text=show_full_text,
                        )
                    )
                elif parsed.path.startswith("/layer-trends/"):
                    self._serve_layer_trends(parsed.path)
                elif parsed.path.startswith("/artifact-plots/"):
                    self._serve_artifact_plots(parsed.path)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._send_error(exc)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _serve_layer_trends(self, path: str) -> None:
            parts = [urllib.parse.unquote(part) for part in path.split("/") if part]
            if len(parts) < 2:
                self.send_error(HTTPStatus.NOT_FOUND, "Layer trend report not found")
                return
            experiment = safe_path_part(parts[1], field="experiment")
            rest = parts[2:] or ["layer_trends.html"]
            if parts[2:] == [] and not path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", f"/layer-trends/{urllib.parse.quote(experiment)}/")
                self.end_headers()
                return
            base = experiment_dir(root=config.artifact_root, experiment=experiment) / "reports" / "pages" / "layer_trends"
            if not base.exists():
                base = config.report_root / "layer_trends" / f"experiment={experiment}"
            target = _safe_join(base, Path(*rest))
            if target.is_dir():
                target = _safe_join(base, Path(*rest) / "layer_trends.html")
            self._send_file(target)

        def _serve_artifact_plots(self, path: str) -> None:
            parts = [urllib.parse.unquote(part) for part in path.split("/") if part]
            if len(parts) < 2:
                self.send_error(HTTPStatus.NOT_FOUND, "Artifact plot report not found")
                return
            experiment = safe_path_part(parts[1], field="experiment")
            rest = parts[2:] or ["artifacts.html"]
            if parts[2:] == [] and not path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", f"/artifact-plots/{urllib.parse.quote(experiment)}/")
                self.end_headers()
                return
            base = experiment_dir(root=config.artifact_root, experiment=experiment) / "reports" / "pages" / "artifacts"
            target = _safe_join(base, Path(*rest))
            if target.is_dir():
                target = _safe_join(base, Path(*rest) / "artifacts.html")
            self._send_file(target)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path.name}")
                return
            content_type = _content_type(path)
            data_bytes = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data_bytes)))
            self.end_headers()
            self.wfile.write(data_bytes)

        def _send_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_error(self, exc: Exception) -> None:
            if isinstance(exc, PermissionError):
                self.send_error(HTTPStatus.FORBIDDEN, str(exc))
                return
            if isinstance(exc, ValueError):
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            message = f"{exc.__class__.__name__}: {exc}"
            body = html_page(
                title="Report Error",
                body=f'<section class="error">{escape(message)}</section><pre>{escape(traceback.format_exc())}</pre>',
            )
            self._send_html(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return ReportRequestHandler


def _controls_html(options: Mapping[str, Sequence[str]], first: Selection | None) -> str:
    first = first or Selection("", "", 0, "", "")
    return f"""
<div class="controls">
  {_select("model", options.get("models", ()), first.model)}
  {_select("sae", options.get("saes", ()), first.sae)}
  {_select("layer", options.get("layers", ()), str(first.layer))}
  {_select("dataset", options.get("datasets", ()), first.dataset)}
  {_select("agg", options.get("aggs", ()), first.agg)}
  {_select("metric", options.get("metrics", ()), "f1")}
  <label>top k<input id="top_k" type="number" min="1" max="500" value="50"></label>
  <label>threshold<input id="threshold" type="number" step="0.01" value="0"></label>
  <label>max display<input id="max_display" type="number" min="1" max="50" value="10"></label>
  <button id="load-features">Load Features</button>
</div>
"""


def _select(name: str, values: Sequence[str], selected: str) -> str:
    opts = []
    for value in values:
        marker = " selected" if str(value) == str(selected) else ""
        opts.append(f'<option value="{escape(value)}"{marker}>{escape(value)}</option>')
    return f'<label>{escape(name)}<select id="{escape(name)}">{"".join(opts)}</select></label>'


def _dashboard_scripts() -> str:
    return """
<script>
function value(id) { return document.getElementById(id).value; }
function params(extra) {
  const base = {
    model: value("model"),
    sae: value("sae"),
    layer: value("layer"),
    dataset: value("dataset"),
    agg: value("agg")
  };
  return new URLSearchParams(Object.assign(base, extra));
}
function showError(text) {
  const box = document.getElementById("feature-error");
  box.style.display = text ? "block" : "none";
  box.textContent = text || "";
}
async function loadFeatures() {
  showError("");
  document.getElementById("features").textContent = "Loading...";
  const query = params({ metric: value("metric"), top_k: value("top_k") });
  const response = await fetch("/api/features?" + query.toString());
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  const rows = payload.features || [];
  if (!rows.length) {
    document.getElementById("features").textContent = "No features found.";
    return;
  }
  const tableRows = rows.map((row) => {
    const fid = row.feature;
    const score = Number(row.score ?? row[payload.metric] ?? 0).toPrecision(5);
    return `<tr><td><button class="secondary" data-feature="${fid}">${fid}</button></td><td class="num">${score}</td><td class="num">${Number(row.f1 ?? 0).toPrecision(5)}</td><td class="num">${Number(row.precision ?? 0).toPrecision(5)}</td><td class="num">${Number(row.recall ?? 0).toPrecision(5)}</td></tr>`;
  }).join("");
  document.getElementById("features").innerHTML = `<div class="table-wrap"><table><thead><tr><th>feature</th><th>score</th><th>f1</th><th>precision</th><th>recall</th></tr></thead><tbody>${tableRows}</tbody></table></div>`;
  document.querySelectorAll("[data-feature]").forEach((button) => {
    button.addEventListener("click", () => loadActivation(button.getAttribute("data-feature")));
  });
}
async function loadActivation(feature) {
  showError("");
  document.getElementById("activation").textContent = "Loading...";
  const query = params({ feature, threshold: value("threshold"), max_display: value("max_display") });
  const response = await fetch("/api/feature-activations?" + query.toString());
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  document.getElementById("activation").innerHTML = payload.html || "";
}
document.getElementById("load-features").addEventListener("click", () => {
  loadFeatures().catch((error) => showError(error.message));
});
</script>
"""


def _artifact_row(stage: str, record: ArtifactRecord) -> dict[str, object]:
    return {
        "stage": stage,
        "status": record.status,
        "target": record.target,
        "log": "-" if record.log_path is None else record.log_path,
        "reason": record.reason,
    }


def _query_one(query: Mapping[str, Sequence[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default
    return str(values[0])


def _resolve_under(root: str | Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(root) / candidate).resolve()


def _safe_join(root: Path, relative: Path) -> Path:
    root_resolved = root.resolve()
    target = (root_resolved / relative).resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise PermissionError(f"Refusing to serve path outside report root: {relative}")
    return target


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".csv":
        return "text/csv; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def _validate_selection(selection: Selection) -> None:
    safe_path_part(selection.model, field="model")
    safe_path_part(selection.sae, field="sae")
    safe_path_part(selection.dataset, field="dataset")
    safe_path_part(selection.agg, field="agg")
    int(selection.layer)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if pd.isna(value):
        return None
    return value
