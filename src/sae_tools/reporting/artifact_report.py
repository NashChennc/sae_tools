from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sae_tools.experiment_store import ExperimentStore
from sae_tools.experiment_store.manifest import ReportPageRecord
from sae_tools.workflow.artifacts import experiment_dir
from sae_tools.workflow.registry import ExperimentSpec, Registry
from sae_tools.workflow.runtime import scan_artifacts

from .html import escape, html_page, render_table
from .layout import write_layout
from .server import ReportServerConfig


@dataclass(frozen=True)
class ArtifactPlot:
    model: str
    sae: str
    layer: int
    dataset: str
    agg: str
    color: str
    path: Path
    size_bytes: int
    modified_at: str
    data_uri: str | None = None
    error: str = ""


@dataclass(frozen=True)
class ArtifactReportResult:
    html_path: Path
    plot_count: int
    embedded_count: int
    error_count: int


def generate_artifact_report(config: ReportServerConfig) -> ArtifactReportResult:
    """Generate a self-contained HTML artifact plot report for one experiment."""
    experiment_name = config.config_path.stem
    output_dir = experiment_dir(root=config.artifact_root, experiment=experiment_name) / "reports" / "pages" / "artifacts"
    html_path = output_dir / "artifacts.html"
    plots = collect_artifact_plots(config)
    status_rows, status_counts = _artifact_status_summary(config)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        render_artifact_report_html(
            config=config,
            plots=plots,
            status_rows=status_rows,
            status_counts=status_counts,
        ),
        encoding="utf-8",
    )
    _record_report_store(config=config, html_path=html_path)
    return ArtifactReportResult(
        html_path=html_path,
        plot_count=len(plots),
        embedded_count=sum(1 for plot in plots if plot.data_uri),
        error_count=sum(1 for plot in plots if plot.error),
    )


def collect_artifact_plots(config: ReportServerConfig) -> list[ArtifactPlot]:
    experiment_name = config.config_path.stem
    stat_root = experiment_dir(root=config.artifact_root, experiment=experiment_name) / "objects" / "stat"
    plots: list[ArtifactPlot] = []
    if not stat_root.exists():
        return plots
    for path in sorted(stat_root.glob("model=*/sae=*/layer=*/dataset=*/agg=*/plots/*.png")):
        dims = _plot_dimensions(stat_root, path)
        stat = path.stat()
        data_uri: str | None = None
        error = ""
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            data_uri = f"data:image/png;base64,{encoded}"
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
        plots.append(
            ArtifactPlot(
                model=dims["model"],
                sae=dims["sae"],
                layer=int(dims["layer"]),
                dataset=dims["dataset"],
                agg=dims["agg"],
                color=_plot_color(path),
                path=path.relative_to(config.repo_root) if _is_relative_to(path, config.repo_root) else path,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                data_uri=data_uri,
                error=error,
            )
        )
    return plots


def render_artifact_report_html(
    *,
    config: ReportServerConfig,
    plots: Sequence[ArtifactPlot],
    status_rows: Sequence[dict[str, object]],
    status_counts: Counter[str],
) -> str:
    experiment_name = config.config_path.stem
    generated = datetime.now(timezone.utc).isoformat()
    embedded_count = sum(1 for plot in plots if plot.data_uri)
    error_count = sum(1 for plot in plots if plot.error)
    cards = (
        ("Experiment", experiment_name),
        ("Plots", len(plots)),
        ("Embedded", embedded_count),
        ("Plot errors", error_count),
        ("Targets done", status_counts.get("done", 0)),
        ("Generated", generated),
    )
    card_html = "".join(
        f'<section class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></section>'
        for label, value in cards
    )
    body = f"""
<section class="grid">{card_html}</section>
<section class="panel">
  <h2>Artifact Status</h2>
  {render_table(status_rows, (("stage", "stage"), ("done", "done"), ("missing", "missing"), ("incomplete", "incomplete"), ("failed", "failed")), numeric={"done", "missing", "incomplete", "failed"})}
</section>
<section class="panel">
  <h2>Plots</h2>
  {_filter_controls(plots)}
  <p class="muted"><span id="plot-count">{len(plots)}</span> visible plots.</p>
  <div class="artifact-plot-grid">
    {"".join(_plot_card(plot) for plot in plots)}
  </div>
</section>
"""
    return html_page(
        title=f"Artifact Plot Report: {experiment_name}",
        subtitle=str(config.config_path),
        body=body,
        scripts=_filter_script(),
        extra_head=f"<style>{_artifact_report_css()}</style>",
    )


def _artifact_status_summary(config: ReportServerConfig) -> tuple[list[dict[str, object]], Counter[str]]:
    scanned = scan_artifacts(
        config.config_path,
        config.registry_dir,
        repo_root=config.repo_root,
        artifact_root=config.artifact_root,
    )
    status_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for stage, records in scanned.items():
        stage_counts = Counter(record.status for record in records)
        status_counts.update(stage_counts)
        rows.append(
            {
                "stage": stage,
                "done": stage_counts.get("done", 0),
                "missing": stage_counts.get("missing", 0),
                "incomplete": stage_counts.get("incomplete", 0),
                "failed": stage_counts.get("failed", 0),
            }
        )
    return rows, status_counts


def _filter_controls(plots: Sequence[ArtifactPlot]) -> str:
    return f"""
<div class="controls">
  <label>search<input id="artifact-search" type="search" placeholder="path, dataset, SAE..."></label>
  {_select("artifact-model", "model", sorted({plot.model for plot in plots}))}
  {_select("artifact-sae", "SAE", sorted({plot.sae for plot in plots}))}
  {_select("artifact-layer", "layer", sorted({str(plot.layer) for plot in plots}, key=lambda item: int(item)))}
  {_select("artifact-dataset", "dataset", sorted({plot.dataset for plot in plots}))}
  {_select("artifact-agg", "agg", sorted({plot.agg for plot in plots}))}
  {_select("artifact-color", "color", sorted({plot.color for plot in plots}))}
</div>
"""


def _select(element_id: str, label: str, values: Sequence[str]) -> str:
    options = ['<option value="">all</option>']
    options.extend(f'<option value="{escape(value)}">{escape(value)}</option>' for value in values)
    return f'<label>{escape(label)}<select id="{escape(element_id)}">{"".join(options)}</select></label>'


def _plot_card(plot: ArtifactPlot) -> str:
    attrs = {
        "model": plot.model,
        "sae": plot.sae,
        "layer": str(plot.layer),
        "dataset": plot.dataset,
        "agg": plot.agg,
        "color": plot.color,
        "search": f"{plot.model} {plot.sae} {plot.layer} {plot.dataset} {plot.agg} {plot.color} {plot.path}",
    }
    attr_html = " ".join(f'data-{key}="{escape(value)}"' for key, value in attrs.items())
    image_html = (
        f'<img class="artifact-plot-image" src="{escape(plot.data_uri)}" alt="{escape(plot.path)}">'
        if plot.data_uri
        else f'<div class="error">{escape(plot.error or "Plot could not be embedded.")}</div>'
    )
    return f"""
<article class="artifact-plot-card" {attr_html}>
  <h3>{escape(plot.dataset)} L{plot.layer} {escape(plot.agg)} {escape(plot.color)}</h3>
  {image_html}
  <dl>
    <dt>model</dt><dd>{escape(plot.model)}</dd>
    <dt>SAE</dt><dd>{escape(plot.sae)}</dd>
    <dt>path</dt><dd>{escape(plot.path)}</dd>
    <dt>size</dt><dd>{_format_bytes(plot.size_bytes)}</dd>
    <dt>modified</dt><dd>{escape(plot.modified_at)}</dd>
  </dl>
</article>
"""


def _filter_script() -> str:
    return """
<script>
const plotCards = Array.from(document.querySelectorAll(".artifact-plot-card"));
const filterIds = ["artifact-model", "artifact-sae", "artifact-layer", "artifact-dataset", "artifact-agg", "artifact-color"];
function filterPlots() {
  const search = document.getElementById("artifact-search").value.trim().toLowerCase();
  let visible = 0;
  for (const card of plotCards) {
    const matchesSearch = !search || card.dataset.search.toLowerCase().includes(search);
    const matchesFilters = filterIds.every((id) => {
      const value = document.getElementById(id).value;
      if (!value) return true;
      const key = id.replace("artifact-", "");
      return card.dataset[key] === value;
    });
    const show = matchesSearch && matchesFilters;
    card.style.display = show ? "" : "none";
    if (show) visible += 1;
  }
  document.getElementById("plot-count").textContent = String(visible);
}
document.getElementById("artifact-search").addEventListener("input", filterPlots);
for (const id of filterIds) {
  document.getElementById(id).addEventListener("change", filterPlots);
}
</script>
"""


def _artifact_report_css() -> str:
    return """
.artifact-plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 14px;
}
.artifact-plot-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 12px;
}
.artifact-plot-card h3 {
  margin-top: 0;
}
.artifact-plot-image {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}
.artifact-plot-card dl {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  gap: 4px 10px;
  margin: 10px 0 0;
  font-size: 12px;
}
.artifact-plot-card dt {
  color: var(--muted);
}
.artifact-plot-card dd {
  margin: 0;
  overflow-wrap: anywhere;
}
"""


def _plot_dimensions(stat_root: Path, path: Path) -> dict[str, str]:
    parts = path.relative_to(stat_root).parts
    if len(parts) < 7:
        raise ValueError(f"Unexpected stat plot path: {path}")
    return {
        "model": _path_value(parts[0], "model"),
        "sae": _path_value(parts[1], "sae"),
        "layer": _path_value(parts[2], "layer"),
        "dataset": _path_value(parts[3], "dataset"),
        "agg": _path_value(parts[4], "agg"),
    }


def _path_value(part: str, key: str) -> str:
    prefix = f"{key}="
    if not part.startswith(prefix):
        raise ValueError(f"Expected {prefix!r} path part, got {part!r}")
    return part[len(prefix) :]


def _plot_color(path: Path) -> str:
    stem = path.stem
    marker = ".color="
    if marker in stem:
        return stem.split(marker, 1)[1]
    return stem


def _record_report_store(*, config: ReportServerConfig, html_path: Path) -> None:
    registry = Registry.load(config.registry_dir)
    experiment = ExperimentSpec.load(config.config_path, registry)
    experiment_name = config.config_path.stem
    store = ExperimentStore.create(
        config.artifact_root,
        experiment=experiment,
        registry=registry,
        experiment_id=experiment_name,
        overwrite=False,
    )
    layout_path = write_layout(store.reports_dir / "layout.json")
    store.record_artifact_path(
        kind="report",
        role="artifact_plots_html",
        path=html_path,
        producer="sae-tools-report artifacts",
        dimensions={"page": "artifact_plots", "role": "html"},
        mime="text/html",
    )
    store.record_report(
        ReportPageRecord(
            page_id="artifact_plots",
            experiment_id=experiment_name,
            path=str(_relative_to(html_path, store.root)),
            title="Artifact Plot Report",
            layout_path=str(_relative_to(layout_path, store.root)),
        )
    )


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
