from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    SelectionList,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from sae_tools.workflow.runtime import STAGES

from .backend import ExperimentDetails, TUIBackend
from .widgets import artifact_counts, bool_text, compact_path, format_bytes, status_text, target_counts


@dataclass
class CreateExperimentDraft:
    name: str = ""
    models: set[str] = field(default_factory=set)
    saes: set[str] = field(default_factory=set)
    layers: set[int] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)
    analyses: set[str] = field(default_factory=set)
    dataset_splits: dict[str, str] = field(default_factory=dict)
    dataset_max_samples: dict[str, str] = field(default_factory=dict)
    batch_size: str = "2"
    overwrite: bool = False


class CreateExperimentScreen(Screen[Path | None]):
    OPTION_PAGE_SIZE = 8

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("left", "back", "Back"),
        ("right", "next", "Next"),
    ]

    def __init__(self, backend: TUIBackend) -> None:
        super().__init__()
        self.backend = backend
        self.registry = backend.load_registry()
        self.draft = CreateExperimentDraft()
        self.page_key = "name"
        self.option_offsets: dict[str, int] = {}
        self.visible_option_values: set[Any] = set()
        self.name_input: Input | None = None
        self.batch_size_input: Input | None = None
        self.overwrite_switch: Switch | None = None
        self.selection_list: SelectionList[Any] | None = None
        self.page_hint: Static | None = None
        self.option_prev_button: Button | None = None
        self.option_next_button: Button | None = None
        self.dataset_split_input: Input | None = None
        self.dataset_max_samples_input: Input | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="create-screen"):
            yield Static("", id="wizard-progress")
            yield Static("", id="wizard-title")
            yield Static("", id="wizard-error")
            yield Vertical(id="wizard-body")
            with Horizontal(id="wizard-actions"):
                yield Button("Cancel", id="wizard-cancel")
                yield Button("Back", id="wizard-back")
                yield Button("Next", id="wizard-next", variant="primary")

    def on_mount(self) -> None:
        self._render_page()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_back(self) -> None:
        self._go_back()

    def action_next(self) -> None:
        self._go_next()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._go_next()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        if event.selection_list is not self.selection_list:
            return
        self._sync_selection_page()
        self._render_selection_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "wizard-cancel":
            event.stop()
            self.dismiss(None)
        elif button_id == "wizard-back":
            event.stop()
            self._go_back()
        elif button_id == "wizard-next":
            event.stop()
            self._go_next()
        elif event.button is self.option_prev_button:
            event.stop()
            self._turn_option_page(-1)
        elif event.button is self.option_next_button:
            event.stop()
            self._turn_option_page(1)

    def _pages(self) -> list[str]:
        pages = ["name", "models", "saes", "layers", "datasets"]
        pages.extend(f"dataset:{key}" for key in sorted(self.draft.datasets))
        pages.extend(["analyses", "review"])
        return pages

    def _page_index(self) -> int:
        pages = self._pages()
        if self.page_key not in pages:
            self.page_key = "datasets" if self.page_key.startswith("dataset:") else pages[0]
        return pages.index(self.page_key)

    def _render_page(self, error: str = "") -> None:
        pages = self._pages()
        index = self._page_index()
        title = self._page_title(self.page_key)
        self.query_one("#wizard-progress", Static).update(f"Step {index + 1} of {len(pages)}")
        self.query_one("#wizard-title", Static).update(title)
        self.query_one("#wizard-error", Static).update(error)

        body = self.query_one("#wizard-body", Vertical)
        for child in list(body.children):
            child.remove()

        if self.page_key == "name":
            self._render_name_page(body)
        elif self.page_key in {"models", "saes", "layers", "datasets", "analyses"}:
            self._render_selection_page(body, self.page_key)
        elif self.page_key.startswith("dataset:"):
            self._render_dataset_options_page(body, self.page_key.split(":", 1)[1])
        elif self.page_key == "review":
            self._render_review_page(body)

        self._update_nav()

    def _render_name_page(self, body: Vertical) -> None:
        self.name_input = Input(value=self.draft.name, placeholder="my-experiment", classes="wizard-input")
        self.batch_size_input = Input(value=self.draft.batch_size, type="integer", classes="wizard-input")
        self.overwrite_switch = Switch(value=self.draft.overwrite)
        body.mount(
            Static("Create a source-controlled experiment YAML under configs/experiments.", classes="wizard-hint"),
            Label("Experiment name"),
            self.name_input,
            Label("Activation batch size"),
            self.batch_size_input,
            Horizontal(
                Static("Overwrite activation cache", classes="wizard-switch-label"),
                self.overwrite_switch,
                classes="wizard-overwrite-row",
            ),
            Static("Overwrite should normally stay off so reusable activation caches are preserved.", classes="wizard-hint"),
        )

    def _render_selection_page(self, body: Vertical, page: str) -> None:
        options = self._selection_options(page)
        offset = self._clamped_option_offset(page, len(options))
        visible = options[offset : offset + self.OPTION_PAGE_SIZE]
        selected = self._selected_values(page)
        self.visible_option_values = {value for _label, value in visible}

        self.page_hint = Static(self._selection_hint(page), classes="wizard-hint")
        body.mount(self.page_hint)
        if visible:
            self.selection_list = SelectionList[Any](
                *((label, value, value in selected) for label, value in visible),
                classes="wizard-selection",
            )
            body.mount(
                self.selection_list
            )
        else:
            self.selection_list = None
            body.mount(Static("No options available for this step.", classes="wizard-hint"))

        page_count = max(1, (len(options) + self.OPTION_PAGE_SIZE - 1) // self.OPTION_PAGE_SIZE)
        current_page = offset // self.OPTION_PAGE_SIZE + 1
        self.option_prev_button = Button("Previous options", disabled=current_page <= 1)
        self.option_next_button = Button("Next options", disabled=current_page >= page_count)
        body.mount(
            Horizontal(
                self.option_prev_button,
                Static(f"Options page {current_page} of {page_count}", classes="option-page-label"),
                self.option_next_button,
                classes="option-pager",
            )
        )

    def _render_dataset_options_page(self, body: Vertical, dataset_key: str) -> None:
        dataset = self.registry.dataset(dataset_key)
        split_value = self.draft.dataset_splits.get(dataset_key, "")
        max_samples_value = self.draft.dataset_max_samples.get(dataset_key, "")
        self.dataset_split_input = Input(
            value=split_value,
            placeholder=dataset.split or "leave empty for registry default",
            classes="wizard-input",
        )
        self.dataset_max_samples_input = Input(
            value=max_samples_value,
            type="integer",
            placeholder="leave empty for all rows",
            classes="wizard-input",
        )
        body.mount(
            Static(
                f"{dataset_key} ({dataset.data_type}); registry split: {dataset.split or 'default'}",
                classes="wizard-hint",
            ),
            Label("Split override"),
            self.dataset_split_input,
            Label("Max samples override"),
            self.dataset_max_samples_input,
            Static("Leave both fields empty to use the registry defaults.", classes="wizard-hint"),
        )

    def _render_review_page(self, body: Vertical) -> None:
        try:
            data = self._build_data()
            spec = self.backend.validate_experiment(data, self.draft.name)
            path = self.backend.experiment_path(self.draft.name)
            summary = (
                f"Path: {path}\n"
                f"Activation targets: {len(spec.activation_jobs(self.registry))}\n"
                f"Stat targets: {len(spec.stat_batch_jobs(self.registry))}\n"
                f"Geometric targets: {len(spec.geometric_jobs(self.registry))}"
            )
            preview = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True).strip()
        except Exception as exc:
            summary = f"Validation failed: {exc}"
            preview = ""
        body.mount(
            Static(summary, classes="review-summary", markup=False),
            Label("YAML preview"),
            Static(preview or "Fix previous pages before saving.", classes="review-yaml", markup=False),
        )

    def _update_nav(self) -> None:
        index = self._page_index()
        next_button = self.query_one("#wizard-next", Button)
        back_button = self.query_one("#wizard-back", Button)
        back_button.disabled = index == 0
        next_button.label = "Save" if self.page_key == "review" else "Next"
        next_button.variant = "success" if self.page_key == "review" else "primary"

    def _page_title(self, page: str) -> str:
        if page.startswith("dataset:"):
            dataset_key = page.split(":", 1)[1]
            dataset_pages = [item for item in self._pages() if item.startswith("dataset:")]
            return f"Dataset options: {dataset_key} ({dataset_pages.index(page) + 1}/{len(dataset_pages)})"
        return {
            "name": "Experiment basics",
            "models": "Choose models",
            "saes": "Choose SAEs",
            "layers": "Choose layers",
            "datasets": "Choose datasets",
            "analyses": "Choose analyses",
            "review": "Review and save",
        }[page]

    def _selection_options(self, page: str) -> list[tuple[str, Any]]:
        if page == "models":
            return [(f"{key}  ({value.hf_name})", key) for key, value in sorted(self.registry.models.items())]
        if page == "saes":
            return [
                (f"{key}  (layers: {list(value.layers)})", key)
                for key, value in sorted(self.registry.saes.items())
            ]
        if page == "layers":
            return [(f"Layer {layer}", layer) for layer in self._common_sae_layers()]
        if page == "datasets":
            return [(f"{key}  ({value.data_type})", key) for key, value in sorted(self.registry.datasets.items())]
        if page == "analyses":
            return [(f"{key}  ({value.kind})", key) for key, value in sorted(self.registry.analyses.items())]
        return []

    def _selection_hint(self, page: str) -> str:
        if page == "models":
            return "Select one or more model registry IDs."
        if page == "saes":
            compatibility = self._compatibility_errors()
            if compatibility:
                return "\n".join(compatibility)
            return "Select one or more SAE registry IDs. Compatibility is checked before continuing."
        if page == "layers":
            if not self.draft.saes:
                return "Select SAEs first. You can go Back to edit them."
            layers = self._common_sae_layers()
            if not layers:
                return "Selected SAEs have no common supported layers. Leave empty only if their default layers are intended."
            return "Leave empty to use each SAE's default_layer, or select shared layers for every selected SAE."
        if page == "datasets":
            return "Select datasets. The next pages let you set split and max_samples per dataset."
        if page == "analyses":
            return "Select statistical and/or geometric analysis groups."
        return ""

    def _render_selection_hint(self) -> None:
        try:
            if self.page_hint is not None:
                self.page_hint.update(self._selection_hint(self.page_key))
        except Exception:
            pass

    def _selected_values(self, page: str) -> set[Any]:
        if page == "models":
            return set(self.draft.models)
        if page == "saes":
            return set(self.draft.saes)
        if page == "layers":
            return set(self.draft.layers)
        if page == "datasets":
            return set(self.draft.datasets)
        if page == "analyses":
            return set(self.draft.analyses)
        return set()

    def _set_selected_values(self, page: str, values: set[Any]) -> None:
        if page == "models":
            self.draft.models = {str(value) for value in values}
        elif page == "saes":
            self.draft.saes = {str(value) for value in values}
            self._normalize_layers_for_saes()
        elif page == "layers":
            self.draft.layers = {int(value) for value in values}
        elif page == "datasets":
            self.draft.datasets = {str(value) for value in values}
            self._normalize_dataset_maps()
        elif page == "analyses":
            self.draft.analyses = {str(value) for value in values}

    def _sync_selection_page(self) -> None:
        if self.page_key not in {"models", "saes", "layers", "datasets", "analyses"}:
            return
        if self.selection_list is None:
            return
        selected = set(self._selected_values(self.page_key))
        selected.difference_update(self.visible_option_values)
        selected.update(self.selection_list.selected)
        self._set_selected_values(self.page_key, selected)

    def _turn_option_page(self, direction: int) -> None:
        self._capture_current_page()
        options = self._selection_options(self.page_key)
        offset = self._clamped_option_offset(self.page_key, len(options))
        self.option_offsets[self.page_key] = offset + direction * self.OPTION_PAGE_SIZE
        self._render_page()

    def _clamped_option_offset(self, page: str, option_count: int) -> int:
        max_offset = max(0, ((option_count - 1) // self.OPTION_PAGE_SIZE) * self.OPTION_PAGE_SIZE)
        offset = min(max(0, self.option_offsets.get(page, 0)), max_offset)
        self.option_offsets[page] = offset
        return offset

    def _go_back(self) -> None:
        self._capture_current_page()
        pages = self._pages()
        index = self._page_index()
        if index > 0:
            self.page_key = pages[index - 1]
        self._render_page()

    def _go_next(self) -> None:
        self._capture_current_page()
        error = self._validate_current_page()
        if error:
            self._render_page(error)
            return
        if self.page_key == "review":
            self._save()
            return
        pages = self._pages()
        index = self._page_index()
        if index < len(pages) - 1:
            self.page_key = pages[index + 1]
        self._render_page()

    def _capture_current_page(self) -> None:
        if self.page_key == "name":
            if self.name_input is not None:
                self.draft.name = self.name_input.value.strip()
            if self.batch_size_input is not None:
                self.draft.batch_size = self.batch_size_input.value.strip()
            if self.overwrite_switch is not None:
                self.draft.overwrite = self.overwrite_switch.value
        elif self.page_key in {"models", "saes", "layers", "datasets", "analyses"}:
            try:
                self._sync_selection_page()
            except Exception:
                pass
        elif self.page_key.startswith("dataset:"):
            dataset_key = self.page_key.split(":", 1)[1]
            if self.dataset_split_input is not None:
                self.draft.dataset_splits[dataset_key] = self.dataset_split_input.value.strip()
            if self.dataset_max_samples_input is not None:
                self.draft.dataset_max_samples[dataset_key] = self.dataset_max_samples_input.value.strip()

    def _validate_current_page(self) -> str | None:
        try:
            if self.page_key == "name":
                if not self.draft.name:
                    return "Experiment name is required."
                path = self.backend.experiment_path(self.draft.name)
                if path.exists():
                    return f"Experiment already exists: {path.name}"
                batch_size = int(self.draft.batch_size or "1")
                if batch_size <= 0:
                    return "Activation batch size must be greater than zero."
            elif self.page_key == "models" and not self.draft.models:
                return "Select at least one model."
            elif self.page_key == "saes":
                if not self.draft.saes:
                    return "Select at least one SAE."
                compatibility = self._compatibility_errors()
                if compatibility:
                    return compatibility[0]
            elif self.page_key == "datasets" and not self.draft.datasets:
                return "Select at least one dataset."
            elif self.page_key.startswith("dataset:"):
                dataset_key = self.page_key.split(":", 1)[1]
                text = self.draft.dataset_max_samples.get(dataset_key, "")
                if text:
                    int(text)
            elif self.page_key == "analyses" and not self.draft.analyses:
                return "Select at least one analysis."
            elif self.page_key == "review":
                self.backend.validate_experiment(self._build_data(), self.draft.name)
        except ValueError as exc:
            return str(exc)
        return None

    def _save(self) -> None:
        try:
            path = self.backend.save_experiment(self._build_data(), self.draft.name)
        except FileExistsError as exc:
            self._render_page(str(exc))
            return
        except Exception as exc:
            self._render_page(f"Validation failed: {exc}")
            return
        self.dismiss(path)

    def _build_data(self) -> dict[str, Any]:
        batch_size = int(self.draft.batch_size or "1")
        datasets: list[dict[str, Any]] = []
        for key in sorted(self.draft.datasets):
            entry: dict[str, Any] = {"id": key}
            split = self.draft.dataset_splits.get(key, "").strip()
            max_samples = self.draft.dataset_max_samples.get(key, "").strip()
            if split:
                entry["split"] = split
            if max_samples:
                entry["max_samples"] = int(max_samples)
            datasets.append(entry)

        data: dict[str, Any] = {
            "models": sorted(self.draft.models),
            "saes": sorted(self.draft.saes),
            "datasets": datasets,
            "activation": {
                "overwrite": self.draft.overwrite,
                "batch_size": batch_size,
            },
            "analyses": sorted(self.draft.analyses),
        }
        if self.draft.layers:
            data["layers"] = sorted(self.draft.layers)
        return data

    def _common_sae_layers(self) -> list[int]:
        layer_sets = [set(self.registry.sae(key).layers) for key in self.draft.saes]
        if not layer_sets:
            return []
        return sorted(set.intersection(*layer_sets))

    def _normalize_layers_for_saes(self) -> None:
        common = set(self._common_sae_layers())
        if common:
            self.draft.layers.intersection_update(common)
        else:
            self.draft.layers.clear()

    def _normalize_dataset_maps(self) -> None:
        for mapping in (self.draft.dataset_splits, self.draft.dataset_max_samples):
            for key in list(mapping):
                if key not in self.draft.datasets:
                    del mapping[key]

    def _compatibility_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.draft.models or not self.draft.saes:
            return errors
        for sae_key in sorted(self.draft.saes):
            compatible = set(self.registry.sae(sae_key).compatible_models)
            if not compatible:
                continue
            incompatible = sorted(self.draft.models.difference(compatible))
            if incompatible:
                errors.append(f"{sae_key} is not compatible with model(s): {', '.join(incompatible)}")
        return errors


class SAEWorkflowTUI(App):
    CSS = """
    Screen {
        background: #101214;
        color: #d7dde4;
    }

    CreateExperimentScreen {
        overflow-y: hidden;
    }

    #selection {
        height: 1;
        color: #9fb0c2;
    }

    TabbedContent {
        height: 1fr;
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

    #experiments-pane {
        height: 1fr;
    }

    #experiment-actions {
        height: 3;
        margin-bottom: 1;
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

    #create-screen {
        height: 1fr;
        overflow-y: hidden;
        padding: 1 2;
    }

    #wizard-progress {
        height: 1;
        color: #8b949e;
    }

    #wizard-title {
        height: 1;
        text-style: bold;
        color: #d7dde4;
    }

    #wizard-error {
        height: 2;
        color: #f85149;
    }

    #wizard-body {
        height: 1fr;
        overflow-y: hidden;
        border: solid #30363d;
        padding: 1 2;
    }

    #wizard-body Label {
        margin-top: 1;
        color: #9fb0c2;
    }

    #wizard-body Input {
        width: 42;
    }

    .wizard-selection {
        height: 10;
        margin: 1 0;
    }

    #wizard-actions {
        height: 3;
        margin-top: 1;
    }

    .option-pager {
        height: 3;
    }

    .option-page-label {
        width: 1fr;
        content-align: center middle;
        color: #8b949e;
    }

    .wizard-hint {
        color: #8b949e;
    }

    .wizard-switch-label {
        width: 32;
        content-align: left middle;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "create_experiment", "Create"),
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
        yield Static("No experiment selected", id="selection")
        with TabbedContent():
            with TabPane("Experiments", id="experiments-tab"):
                with Vertical(id="experiments-pane"):
                    with Horizontal(id="experiment-actions"):
                        yield Button("New Experiment", id="new-experiment", variant="primary")
                    yield DataTable(id="experiments")
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

    def action_create_experiment(self) -> None:
        self.open_create_experiment()

    def open_create_experiment(self) -> None:
        self.push_screen(CreateExperimentScreen(self.backend), self._on_experiment_created)

    def _on_experiment_created(self, path: Path | None) -> None:
        if path is None:
            return
        self.populate_experiments()
        self.select_experiment(path)

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
        elif event.button.id == "new-experiment":
            self.open_create_experiment()
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
        if not rows:
            backend, _ = self.backend.gpu_backend()
            if backend == "apple-silicon":
                table.add_row("-", "-", "-", "-", "-", "-", "No NVIDIA GPUs on this Mac. GPU runner requires NVIDIA+CUDA.")
            else:
                table.add_row("-", "-", "-", "-", "-", "-", "No GPUs detected. Check nvidia-smi or GPU drivers.")
            return
        for row in rows:
            reason = str(row.get("reason", ""))
            notes = str(row.get("notes", ""))
            if notes:
                reason = f"{reason}  [{notes}]"
            table.add_row(
                str(row["gpu"]),
                str(row["name"]),
                f"{row['used_mib']} / {row['total_mib']} MiB",
                f"{row['free_mib']} MiB",
                f"{row['util_pct']}%",
                str(row["selected"]),
                reason,
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
