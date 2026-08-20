#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-29"


"""
dashboard_utils.py
_________________________________________
Shared Jinja2-dashboard-context helpers reused by every model's report
renderer (currently `arycolbring/narative/advirender.py` and
`arycolbring/narative/rearender.py`).

These were previously copy-pasted, near-identically, into both renderer
modules. Centralizing them here means a fix or a new color/icon only has
to happen once, and keeps model-specific renderers focused on their own
context shape instead of generic scorecard/gauge bookkeeping.
"""

import math
from typing import Any, Dict, List

from ..configs import logger

# Cycle of card colors + matching FontAwesome icon per color, shared by
# every scorecard grid in the project (training + inference dashboards).
_COLOR_CYCLE = ["blue", "green", "purple", "orange", "cyan", "red"]
_ICON_MAP: Dict[str, str] = {
    "blue":   "fas fa-chart-line",
    "green":  "fas fa-network-wired",
    "purple": "fas fa-bullseye",
    "orange": "fas fa-gauge-high",
    "cyan":   "fas fa-desktop",
    "red":    "fas fa-chart-column",
}

# Metric-name substrings that identify a *ranking/quality* metric, i.e.
# one that belongs on a training/evaluation report and is expressed as a
# 0-1 quality gauge (Precision@K, Recall@K, NDCG, AUC, MRR, F1, ...).
_GAUGE_KEYWORDS = ("ndcg", "map", "mrr", "precision",
                   "recall", "accuracy", "auc", "f1")

_LABEL_ABBREVIATIONS = {
    "ndcg": "NDCG",
    "map": "MAP",
    "mrr": "MRR",
    "auc": "AUC",
    "ctr": "CTR",
    "precision": "Precision",
    "recall": "Recall",
}


def bealabel(label: str) -> str:
    """Turn a snake_case metric name into a human-friendly title, with
    known ML abbreviations (NDCG, MRR, AUC, ...) re-capitalized and the
    `_at_` token rendered as an attached "@" (e.g. `ndcg_at_10` ->
    "NDCG@10", not "Ndcg @ 10").

    Works token-by-token (split on "_") rather than doing global
    substring replacement on a space-joined string -- substring
    replacement was the bug in the original implementation: replacing
    "at" as a plain substring after the underscores were already turned
    into spaces left stray spaces around the "@" and missed the
    "ndcg"/"precision" tokens (which had already been widened to
    "ndcg " / "precision ").
    """
    try:
        tokens = str(label).split("_")
        parts: List[str] = list()
        for token in tokens:
            lower = token.lower()
            if lower == "at":
                parts.append("@")
            elif lower in _LABEL_ABBREVIATIONS:
                parts.append(_LABEL_ABBREVIATIONS[lower])
            else:
                parts.append(token.capitalize())

        out = ""
        for idx, part in enumerate(parts):
            glue_to_prev = part == "@" or (idx > 0 and parts[idx - 1] == "@")
            out += part if (glue_to_prev or not out) else f" {part}"
        return out
    except Exception:
        return str(label)


def safe_float(value: Any, precision: int = 4) -> str:
    """Format a numeric value for display, tolerating NaN and non-numeric
    input instead of raising."""
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            return f"{value:.{precision}f}"
        return str(value)
    except Exception:
        return str(value)


def detect_gauge_metric(metric_name: str) -> bool:
    """True if `metric_name` looks like a ranking/quality metric
    (Precision@K, Recall@K, NDCG, AUC, MRR, F1, ...).

    This is the correct thing to check for a *training/evaluation*
    dashboard. It should NOT be applied to production inference metrics
    (latency, QPS, throughput) -- those are never "quality gauges".
    """
    metric_name = metric_name.lower()
    return any(k in metric_name for k in _GAUGE_KEYWORDS)


def generate_scorecards(metrics: Dict[str, Any],
                         sub_label: str = "Metric",
                        ) -> List[Dict[str, Any]]:
    """Generate scorecards dynamically from a flat metrics dict.

    Each card carries both `icon_class` (a bare FontAwesome class string)
    and `icon_emoji` (the same icon pre-wrapped in an `<i>` tag) so either
    template convention already in use across the project keeps working.
    """
    if not metrics:
        return list()
    cards = list()
    try:
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            color = _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]
            icon_class = _ICON_MAP[color]
            cards.append({"label"      : bealabel(metric_name),
                          "value"      : safe_float(metric_value),
                          "sub"        : sub_label,
                          "color"      : color,
                          "icon_class" : icon_class,
                          "icon_emoji" : f'<i class="{icon_class}"></i>'})
        logger.info("Generated %d scorecards.", len(cards))
        return cards
    except Exception as exc:
        logger.error("Scorecard generation failed.", exc_info=True)
        raise RuntimeError("Failed generating scorecards.") from exc


def generate_gauges(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate gauge widgets for the ranking/quality metrics found in
    `metrics` (see `detect_gauge_metric`). Non-quality metrics (latency,
    QPS, ...) are silently skipped -- callers that only ever pass
    inference-performance metrics will simply get an empty list back."""
    gauges = list()
    try:
        for metric_name, metric_value in metrics.items():
            if not detect_gauge_metric(metric_name):
                continue
            try:
                percent = float(metric_value) * 100
            except (ValueError, TypeError):
                continue
            gauges.append({"label"  : bealabel(metric_name),
                          "value"   : metric_value,
                          "display" : f"{percent:.2f}%",
                          "percent" : round(percent, 2)})
        logger.info("Generated %d gauges.", len(gauges))
        return gauges
    except Exception as exc:
        logger.error("Gauge generation failed.", exc_info=True)
        raise RuntimeError("Failed generating gauges.") from exc


def normalize_charts(charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of loosely-shaped chart-spec dicts into the
    fixed shape the report templates expect."""
    if not charts:
        return list()
    normalized = list()
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        title = chart.get("title", "Untitled Chart")
        normalized.append({"title": title,
                          "type"  : chart.get("type"),
                          "data"  : chart.get("data"),
                          "full"  : "importance" in title.lower()})
    return normalized


def overall_score_percent(metrics: Dict[str, Any]) -> float:
    """Average of every detected quality-gauge metric, as a 0-100
    percent. Returns 0 if there are no quality metrics in `metrics`
    (which is the expected/correct case for an inference-only report)."""
    try:
        gauge_values = [float(v) for k, v in metrics.items()
                        if detect_gauge_metric(k)]
        if not gauge_values:
            return 0.0
        return round(sum(gauge_values) / len(gauge_values) * 100, 2)
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    logger.info("dashboard_utils self-test placeholder.")
