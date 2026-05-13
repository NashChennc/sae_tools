from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from jinja2 import Environment, select_autoescape


BASE_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --panel-soft: #f0f3f7;
  --text: #18212d;
  --muted: #657181;
  --line: #d9e0e8;
  --accent: #1565c0;
  --accent-soft: #e7f0fb;
  --ok: #1b7f4b;
  --warn: #9a6700;
  --bad: #b42318;
  --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.48 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.shell { max-width: 1280px; margin: 0 auto; padding: 24px; }
.topbar { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 18px; }
.nav { display: flex; flex-wrap: wrap; gap: 8px; }
.nav a {
  display: inline-flex; align-items: center; min-height: 32px; padding: 5px 10px;
  border: 1px solid var(--line); border-radius: 6px; background: var(--panel);
}
h1 { margin: 0 0 4px; font-size: 24px; letter-spacing: 0; }
h2 { margin: 26px 0 10px; font-size: 18px; letter-spacing: 0; }
h3 { margin: 18px 0 8px; font-size: 15px; letter-spacing: 0; }
.subtitle { color: var(--muted); margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px; box-shadow: var(--shadow);
}
.card .label { color: var(--muted); font-size: 12px; }
.card .value { font-size: 20px; font-weight: 650; margin-top: 2px; }
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; margin: 12px 0; box-shadow: var(--shadow);
}
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
table { width: 100%; border-collapse: collapse; min-width: 720px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: var(--panel-soft); color: #344054; font-size: 12px; font-weight: 650; }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); }
.badge {
  display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px;
  border-radius: 999px; font-size: 12px; font-weight: 650; background: var(--panel-soft);
}
.badge.done { color: var(--ok); background: #e8f5ee; }
.badge.missing { color: var(--warn); background: #fff4d6; }
.badge.incomplete, .badge.failed { color: var(--bad); background: #fdeceb; }
.plot { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.links { display: flex; flex-wrap: wrap; gap: 10px; }
.error { color: var(--bad); background: #fdeceb; border: 1px solid #f4b8b0; border-radius: 8px; padding: 10px; }
.controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; align-items: end; }
label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
select, input, button {
  min-height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #fff;
  color: var(--text); padding: 5px 8px; font: inherit;
}
button { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 650; cursor: pointer; }
button.secondary { background: #fff; color: var(--accent); }
pre { white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 12px; border-radius: 8px; overflow-x: auto; }
"""

_HTML_ENV = Environment(autoescape=select_autoescape(default=True))
_PAGE_TEMPLATE = _HTML_ENV.from_string(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>{{ css | safe }}</style>
  {{ extra_head | safe }}
</head>
<body>
  <main class="shell">
    <div class="topbar">
      <div>
        <h1>{{ title }}</h1>
        {% if subtitle %}<p class="subtitle">{{ subtitle }}</p>{% endif %}
      </div>
      <nav class="nav">
        <a href="/">Reports</a>
        <a href="/dashboard">Dashboard</a>
      </nav>
    </div>
    {{ body | safe }}
  </main>
  {{ scripts | safe }}
</body>
</html>
"""
)


def escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def html_page(
    *,
    title: str,
    body: str,
    subtitle: str = "",
    scripts: str = "",
    extra_head: str = "",
) -> str:
    return _PAGE_TEMPLATE.render(
        title=str(title),
        subtitle=str(subtitle),
        body=body,
        scripts=scripts,
        extra_head=extra_head,
        css=BASE_CSS,
    )


def status_badge(status: object) -> str:
    text = escape(status)
    css = text.lower()
    return f'<span class="badge {css}">{text}</span>'


def format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.6g}"


def relative_link(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, start=base_dir).replace(os.sep, "/")


def render_table(
    rows: Iterable[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
    *,
    numeric: set[str] | None = None,
    empty_text: str = "No rows.",
) -> str:
    numeric = numeric or set()
    row_list = list(rows)
    if not row_list:
        return f'<p class="muted">{escape(empty_text)}</p>'
    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body_rows: list[str] = []
    for row in row_list:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            css = ' class="num"' if key in numeric else ""
            if key == "status":
                cell = status_badge(value)
            elif key in numeric:
                cell = format_float(value)
            else:
                cell = escape(value)
            cells.append(f"<td{css}>{cell}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def write_layer_trends_html(
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
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_by_metric = {plot.stem: plot for plot in plot_paths}
    generated = datetime.now(timezone.utc).isoformat()

    cards = [
        ("Generated", generated),
        ("Metrics", ", ".join(metrics)),
        ("Top-k", top_k),
        ("Trend rows", len(trend_df)),
        ("Missing artifacts", len(missing_df)),
    ]
    card_html = "".join(
        f'<section class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></section>'
        for label, value in cards
    )

    best_rows = _best_layer_rows(trend_df, metrics)
    best_table = render_table(
        best_rows,
        (
            ("metric", "metric"),
            ("best_layer", "best layer"),
            ("topk_mean", "top-k mean"),
            ("model", "model"),
            ("sae", "sae"),
            ("dataset", "dataset"),
            ("agg", "agg"),
        ),
        numeric={"best_layer", "topk_mean"},
    )

    plot_sections: list[str] = []
    for metric in metrics:
        plot = plot_by_metric.get(metric)
        if plot is None:
            plot_sections.append(f"<h3>{escape(metric)}</h3><p class=\"muted\">No data available.</p>")
            continue
        rel = relative_link(plot, path.parent)
        plot_sections.append(f'<h3>{escape(metric)}</h3><img class="plot" src="{escape(rel)}" alt="{escape(metric)} plot">')

    if missing_df.empty:
        missing_html = '<p class="muted">No selected stat artifacts were missing or incomplete.</p>'
    else:
        counts = missing_df["status"].value_counts().sort_index().to_dict()
        count_text = ", ".join(f"{status}: {count}" for status, count in counts.items())
        preview = missing_df.head(100).to_dict(orient="records")
        missing_html = (
            f"<p>Missing/incomplete summary: <strong>{escape(count_text)}</strong>.</p>"
            + render_table(
                preview,
                (
                    ("status", "status"),
                    ("model", "model"),
                    ("sae", "sae"),
                    ("layer", "layer"),
                    ("dataset", "dataset"),
                    ("agg", "agg"),
                    ("reason", "reason"),
                ),
                numeric={"layer"},
            )
        )

    body = f"""
<section class="grid">{card_html}</section>
<section class="panel">
  <h2>Best Layers</h2>
  {best_table}
</section>
<section class="panel">
  <h2>Plots</h2>
  {''.join(plot_sections)}
</section>
<section class="panel">
  <h2>Missing Artifacts</h2>
  {missing_html}
</section>
<section class="panel">
  <h2>Data Files</h2>
  <div class="links">
    <a href="{escape(relative_link(trend_csv, path.parent))}">Layer trends CSV</a>
    <a href="{escape(relative_link(missing_csv, path.parent))}">Missing artifacts CSV</a>
  </div>
</section>
"""
    page = html_page(
        title=f"Layer Trend Report: {experiment_name}",
        subtitle=f"Config: {config_path}",
        body=body,
    )
    path.write_text(page, encoding="utf-8")
    return path


def _best_layer_rows(trend_df: pd.DataFrame, metrics: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if trend_df.empty or metric not in set(trend_df["metric"]):
            rows.append({"metric": metric, "best_layer": None, "topk_mean": None, "model": "-", "sae": "-", "dataset": "-", "agg": "-"})
            continue
        metric_df = trend_df[trend_df["metric"] == metric].copy()
        idx = pd.to_numeric(metric_df["topk_mean"], errors="coerce").idxmax()
        row = metric_df.loc[idx]
        rows.append(
            {
                "metric": metric,
                "best_layer": int(row["layer"]),
                "topk_mean": row["topk_mean"],
                "model": row["model"],
                "sae": row["sae"],
                "dataset": row["dataset"],
                "agg": row["agg"],
            }
        )
    return rows
