from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportBlock(_Model):
    id: str
    kind: Literal["summary", "table", "plot_grid", "links", "dashboard"]
    title: str
    source: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ReportPage(_Model):
    id: str
    title: str
    route: str
    blocks: list[ReportBlock]


class ReportLayout(_Model):
    schema_version: int = 1
    default_page: str = "dashboard"
    pages: list[ReportPage]


def default_layout() -> ReportLayout:
    return ReportLayout(
        pages=[
            ReportPage(
                id="dashboard",
                title="SAE Tools Dashboard",
                route="/dashboard",
                blocks=[
                    ReportBlock(id="feature_explorer", kind="dashboard", title="Feature Explorer"),
                    ReportBlock(id="artifact_status", kind="table", title="Artifact Status", source="index/jobs.parquet"),
                ],
            ),
            ReportPage(
                id="layer_trends",
                title="Layer Trend Report",
                route="/layer-trends/{experiment}/",
                blocks=[
                    ReportBlock(id="summary", kind="summary", title="Summary", source="reports/pages/layer_trends/summary.json"),
                    ReportBlock(id="best_layers", kind="table", title="Best Layers", source="reports/pages/layer_trends/layer_trends.csv"),
                    ReportBlock(id="plots", kind="plot_grid", title="Plots", source="reports/pages/layer_trends/plots"),
                    ReportBlock(id="data_files", kind="links", title="Data Files", source="reports/pages/layer_trends"),
                ],
            ),
            ReportPage(
                id="artifact_plots",
                title="Artifact Plot Report",
                route="/artifact-plots/{experiment}/",
                blocks=[
                    ReportBlock(id="summary", kind="summary", title="Summary", source="reports/pages/artifacts/artifacts.html"),
                    ReportBlock(id="plots", kind="plot_grid", title="Plots", source="reports/pages/artifacts/artifacts.html"),
                ],
            ),
        ]
    )


def write_layout(path: str | Path, layout: ReportLayout | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (layout or default_layout()).model_dump(mode="json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_layout(path: str | Path) -> ReportLayout:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ReportLayout.model_validate(json.load(handle))
