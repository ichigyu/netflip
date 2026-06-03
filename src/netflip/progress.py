"""Progress reporting helpers shared by CLI-facing workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

from netflip.manifest import JSONScalar

ProgressReporter: TypeAlias = Callable[[str], None]


def report_progress(
    progress: ProgressReporter | None,
    message: str,
) -> None:
    """Emit one progress message when a reporter is configured."""
    if progress is not None:
        progress(message)


def format_progress_metrics(metrics: Mapping[str, JSONScalar]) -> str:
    """Format metric mappings as stable key=value progress text."""
    return " ".join(
        f"{metric_name}={_format_progress_metric_value(metric_value)}"
        for metric_name, metric_value in sorted(metrics.items())
    )


def _format_progress_metric_value(value: JSONScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)
