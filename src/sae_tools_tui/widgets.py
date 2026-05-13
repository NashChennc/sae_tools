from __future__ import annotations

from pathlib import Path

from rich.text import Text


STATUS_STYLE = {
    "done": "green",
    "missing": "yellow",
    "incomplete": "orange3",
    "failed": "red",
}


def status_text(value: str) -> Text:
    return Text(value, style=STATUS_STYLE.get(value, "white"))


def bool_text(value: bool) -> Text:
    return Text("ok" if value else "missing", style="green" if value else "red")


def compact_path(path: Path | None, *, max_len: int = 76) -> str:
    if path is None:
        return "-"
    text = str(path)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def target_counts(activations: int, stat: int, geometric: int) -> str:
    return f"A {activations} / S {stat} / G {geometric}"


def artifact_counts(done: int, missing: int, incomplete: int, failed: int) -> str:
    return f"{done} done, {missing} missing, {incomplete} incomplete, {failed} failed"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
