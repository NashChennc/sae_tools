from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, DataTable, Footer, Header, RichLog, Static, TabbedContent, TabPane

from sae_tools.workflow.runtime import STAGES

from .backend import ExperimentDetails, TUIBackend
from .widgets import artifact_counts, bool_text, compact_path, format_bytes, status_text, target_counts


class SAEWorkflowTUI(App):
    CSS = """
    Screen {
        background: #101214;
        color: #d7dde4;
    }

    #main {
        height: 1fr;
    }

    #sidebar {
        width: 39%;
        min-width: 54;
        border-right: solid #30363d;
        padding: 0 1;
    }

    #content {
        width: 1fr;
        padding: 0 1;
    }

    .title {
        height: 1;
        color: #f0f3f6;
        text-style: bold;
        margin: 0 0 1 0;
    }

    #selection {
        height: 1;
        color: #9fb0c2;
    }

    DataTable {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    #run-controls {
        height: auto;
        margin: 0 0 1 0;
    }

    #run-buttons {
        height: 3;
    }

    Button {
        margin-right: 1;
        min-width: 16;
    }

    Checkbox {
        margin-right: 2;
    }

    #run-log {
        height: 1fr;
        border: solid #30363d;
    }

    #artifact-browser {
        height: 1fr;
    }

    #artifact-files {
        width: 58%;
    }

    #artifact-detail {
        width: 1fr;
        height: 1fr;
        border: solid #30363d;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("d", "dry_run", "Dry run"),
        ("m", "run_missing", "Run missing"),
    ]

    def __init__(self, *, repo_root: str | Path | None = None, initial_config: str | Path | None = None) -> None:
        super().__init__()
        self.backend = TUIBackend(repo_root=repo_root)
        self.initial_config = Path(initial_config) if initial_config is not None else None
        self.selected_config: Path | None = None
        self.details: ExperimentDetails | None = None
        self.experiment_paths: list[Path] = []
        self.artifact_file_paths: list[Path] = []
        self.selected_artifact_file: Path | None = None
        self.running_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Static("Experiments", classes="title")
                yield DataTable(id="experiments")
            with Vertical(id="content"):
                yield Static("No experiment selected", id="selection")
                with TabbedContent():
                    with TabPane("Checks", id="checks-tab"):
                        yield DataTable(id="checks")
                    with TabPane("Resources", id="resources-tab"):
                        yield DataTable(id="resources")
                    with TabPane("Artifacts", id="artifacts-tab"):
                        yield DataTable(id="artifacts")
                    with TabPane("Browse", id="browse-tab"):
                        with Horizontal(id="artifact-browser"):
                            yield DataTable(id="artifact-files")
                            yield RichLog(id="artifact-detail", highlight=False, markup=False)
                    with TabPane("GPUs", id="gpus-tab"):
                        yield DataTable(id="gpus")
                    with TabPane("Run", id="run-tab"):
                        with Vertical(id="run-controls"):
                            with Horizontal(id="stages"):
                                yield Checkbox("activations", value=True, id="stage-activations")
                                yield Checkbox("stat", value=True, id="stage-stat")
                                yield Checkbox("geometric", value=True, id="stage-geometric")
                            with Horizontal(id="run-buttons"):
                                yield Button("Dry run", id="dry-run", variant="primary", disabled=True)
                                yield Button("Run missing", id="run-missing", variant="success", disabled=True)
                                yield Button("Refresh", id="refresh", variant="default")
                        yield RichLog(id="run-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._configure_tables()
        self.refresh_all()
        self.set_interval(10, self.refresh_dynamic)

    def action_refresh(self) -> None:
        self.refresh_all()

    def action_dry_run(self) -> None:
        self.start_command("dry")

    def action_run_missing(self) -> None:
        self.start_command("run")

    def on_data_table_row_selected(self, event: object) -> None:
        data_table = getattr(event, "data_table", None)
        table_id = getattr(data_table, "id", None)
        if table_id == "artifact-files":
            row_key = getattr(event, "row_key", None)
            if row_key is not None:
                self.show_artifact_detail(Path(str(getattr(row_key, "value", row_key))))
                return
            row_index = getattr(event, "cursor_row", getattr(data_table, "cursor_row", None))
            if row_index is not None and 0 <= row_index < len(self.artifact_file_paths):
                self.show_artifact_detail(self.artifact_file_paths[row_index])
            return
        if table_id != "experiments":
            return
        row_key = getattr(event, "row_key", None)
        if row_key is not None:
            key_value = getattr(row_key, "value", row_key)
            for path in self.experiment_paths:
                if str(path) == str(key_value):
                    self.select_experiment(path)
                    return
        row_index = getattr(event, "cursor_row", getattr(data_table, "cursor_row", None))
        if row_index is None or row_index < 0 or row_index >= len(self.experiment_paths):
            return
        self.select_experiment(self.experiment_paths[row_index])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh":
            self.refresh_all()
        elif event.button.id == "dry-run":
            self.start_command("dry")
        elif event.button.id == "run-missing":
            self.start_command("run")

    def _configure_tables(self) -> None:
        self._reset_table(
            "experiments",
            "experiment",
            "models",
            "SAEs",
            "layers",
            "datasets",
            "targets",
            "artifacts",
            "error",
        )
        self.query_one("#experiments", DataTable).cursor_type = "row"
        self._reset_table("checks", "check", "status", "detail")
        self._reset_table("resources", "kind", "key", "status", "path", "detail")
        self._reset_table("artifacts", "stage", "status", "target", "log", "reason")
        self._reset_table("artifact-files", "kind", "role", "size", "modified", "path")
        self.query_one("#artifact-files", DataTable).cursor_type = "row"
        self._reset_table("gpus", "gpu", "name", "used", "free", "util", "idle", "reason")

    def _reset_table(self, table_id: str, *columns: str) -> DataTable:
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=True)
        table.add_columns(*columns)
        return table

    def refresh_all(self) -> None:
        self.populate_experiments()
        if self.selected_config is None and self.initial_config is not None:
            self.select_experiment(self.initial_config)
        elif self.selected_config is None and self.experiment_paths:
            self.select_experiment(self.experiment_paths[0])
        elif self.selected_config is not None:
            self.select_experiment(self.selected_config)
        else:
            self.populate_checks(None)
            self.populate_empty_details()
        self.populate_gpus()

    def refresh_dynamic(self) -> None:
        if self.selected_config is not None:
            self.populate_artifacts(self.selected_config)
            self.populate_artifact_files(self.selected_config)
        self.populate_gpus()

    def populate_experiments(self) -> None:
        table = self._reset_table(
            "experiments",
            "experiment",
            "models",
            "SAEs",
            "layers",
            "datasets",
            "targets",
            "artifacts",
            "error",
        )
        table.cursor_type = "row"
        summaries = self.backend.list_experiments()
        self.experiment_paths = [summary.path for summary in summaries]
        for summary in summaries:
            table.add_row(
                summary.name,
                str(summary.models),
                str(summary.saes),
                str(summary.layers),
                str(summary.datasets),
                target_counts(summary.activation_targets, summary.stat_targets, summary.geometric_targets),
                artifact_counts(summary.done, summary.missing, summary.incomplete, summary.failed),
                summary.error,
                key=str(summary.path),
            )

    def select_experiment(self, config_path: Path) -> None:
        if self.selected_config != Path(config_path):
            self.selected_artifact_file = None
        self.selected_config = Path(config_path)
        selection = self.query_one("#selection", Static)
        try:
            self.details = self.backend.load_details(self.selected_config)
        except Exception as exc:
            self.details = None
            selection.update(f"{self.selected_config}: {exc}")
            self.populate_checks(self.selected_config)
            self.populate_empty_details()
            self.set_run_buttons(False)
            return
        selection.update(
            f"{self.details.experiment.path.name}  "
            f"models={len(self.details.experiment.models)} "
            f"saes={len(self.details.experiment.saes)} "
            f"datasets={len(self.details.experiment.datasets)}"
        )
        self.populate_checks(self.selected_config)
        self.populate_resources(self.details)
        self.populate_artifacts(self.selected_config)
        self.populate_artifact_files(self.selected_config)
        self.set_run_buttons(self.details.runnable)

    def populate_checks(self, config_path: Path | None) -> None:
        table = self._reset_table("checks", "check", "status", "detail")
        checks = self.backend.environment_checks(config_path)
        for check in checks:
            table.add_row(check.name, bool_text(check.ok), check.detail)

    def populate_empty_details(self) -> None:
        self._reset_table("resources", "kind", "key", "status", "path", "detail")
        self._reset_table("artifacts", "stage", "status", "target", "log", "reason")
        self._reset_table("artifact-files", "kind", "role", "size", "modified", "path")
        self.artifact_file_paths = []
        self.selected_artifact_file = None
        self.query_one("#artifact-detail", RichLog).clear()
        self.set_run_buttons(False)

    def populate_resources(self, details: ExperimentDetails) -> None:
        table = self._reset_table("resources", "kind", "key", "status", "path", "detail")
        for record in details.resources:
            table.add_row(record.kind, record.key, bool_text(record.available), compact_path(record.path), record.detail)

    def populate_artifacts(self, config_path: Path) -> None:
        table = self._reset_table("artifacts", "stage", "status", "target", "log", "reason")
        try:
            records_by_stage = self.backend.artifact_records(config_path)
        except Exception as exc:
            table.add_row("error", status_text("failed"), str(config_path), "-", str(exc))
            return
        for stage in STAGES:
            for record in records_by_stage.get(stage, []):
                table.add_row(
                    stage,
                    status_text(record.status),
                    compact_path(record.target),
                    compact_path(record.log_path),
                    record.reason,
                )

    def populate_artifact_files(self, config_path: Path) -> None:
        table = self._reset_table("artifact-files", "kind", "role", "size", "modified", "path")
        table.cursor_type = "row"
        detail = self.query_one("#artifact-detail", RichLog)
        try:
            records = self.backend.artifact_files(config_path)
        except Exception as exc:
            self.artifact_file_paths = []
            table.add_row("error", "scan", "-", "-", str(exc))
            detail.clear()
            detail.write(str(exc))
            return
        self.artifact_file_paths = [record.path for record in records]
        for record in records:
            table.add_row(
                record.kind,
                record.role,
                format_bytes(record.size_bytes),
                record.modified_at,
                compact_path(record.path, max_len=90),
                key=str(record.path),
            )
        if self.selected_artifact_file in self.artifact_file_paths:
            self.show_artifact_detail(self.selected_artifact_file)
        elif records:
            self.selected_artifact_file = None
            detail.clear()
            detail.write(f"{len(records)} files scanned. Select a row to view details.")
        else:
            self.selected_artifact_file = None
            detail.clear()
            detail.write("No artifact files found for this experiment.")

    def show_artifact_detail(self, path: Path) -> None:
        self.selected_artifact_file = path
        detail = self.query_one("#artifact-detail", RichLog)
        detail.clear()
        try:
            detail.write(self.backend.artifact_file_detail(path))
        except Exception as exc:
            detail.write(f"{path}: {exc}")

    def populate_gpus(self) -> None:
        table = self._reset_table("gpus", "gpu", "name", "used", "free", "util", "idle", "reason")
        _selected, rows, error = self.backend.gpu_rows()
        if error is not None:
            table.add_row("-", "-", "-", "-", "-", status_text("failed"), error)
            return
        for row in rows:
            table.add_row(
                str(row["gpu"]),
                str(row["name"]),
                f"{row['used_mib']} / {row['total_mib']} MiB",
                f"{row['free_mib']} MiB",
                f"{row['util_pct']}%",
                str(row["selected"]),
                str(row["reason"]),
            )

    def set_run_buttons(self, enabled: bool) -> None:
        busy = self.running_task is not None and not self.running_task.done()
        for button_id in ("dry-run", "run-missing"):
            button = self.query_one(f"#{button_id}", Button)
            button.disabled = not enabled or busy

    def selected_stages(self) -> list[str]:
        stages = []
        for stage in STAGES:
            checkbox = self.query_one(f"#stage-{stage}", Checkbox)
            if checkbox.value:
                stages.append(stage)
        return stages or list(STAGES)

    def start_command(self, kind: str) -> None:
        if self.selected_config is None or self.details is None:
            return
        if not self.details.runnable:
            self.append_log("Environment checks are failing; run controls are disabled.")
            return
        if self.running_task is not None and not self.running_task.done():
            self.append_log("A command is already running.")
            return
        stages = self.selected_stages()
        command = (
            self.backend.dry_run_command(self.selected_config, stages)
            if kind == "dry"
            else self.backend.run_missing_command(self.selected_config, stages)
        )
        log = self.query_one("#run-log", RichLog)
        log.clear()
        self.running_task = asyncio.create_task(self._stream_command(command))
        self.set_run_buttons(False)

    async def _stream_command(self, command: list[str]) -> None:
        try:
            async for event in self.backend.stream_command(command):
                self.append_log(event.text)
                if event.kind == "exit":
                    self.refresh_dynamic()
        finally:
            self.set_run_buttons(self.details.runnable if self.details is not None else False)

    def append_log(self, message: str) -> None:
        log = self.query_one("#run-log", RichLog)
        log.write(message)
