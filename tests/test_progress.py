from __future__ import annotations

from netflip.progress import format_progress_metrics, report_progress


def test_report_progress_emits_only_when_reporter_is_configured() -> None:
    messages: list[str] = []

    report_progress(None, "ignored")
    report_progress(messages.append, "visible")

    assert messages == ["visible"]


def test_format_progress_metrics_uses_stable_key_value_text() -> None:
    assert format_progress_metrics(
        {
            "top1_accuracy": 0.9209123,
            "cross_entropy": 0.294785,
            "is_clean": True,
            "label": "baseline",
            "missing": None,
        }
    ) == (
        "cross_entropy=0.294785 is_clean=true label=baseline missing=null "
        "top1_accuracy=0.920912"
    )
