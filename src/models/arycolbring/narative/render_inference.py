#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-31"


"""
render_inference.py
-------------------
Inference dashboard renderer for AryColBring collaborative filtering model.

Generates comprehensive HTML reports for the production inference phase including:
- Inference metrics (latency, throughput, QPS)
- Prediction statistics
- Top-N recommendation analysis with confidence badges
- User-item interaction patterns
- Model performance monitoring

Modular structure:
  templates/base.html.j2         — layout (sidebar, tabs, footer)
  templates/overview.html.j2     — scorecards, charts, gauges
  templates/rankings.html.j2     — predictions table
  templates/diagnostics.html.j2  — KPI cards, metrics detail
  templates/config.html.j2       — inference parameters

Static assets are copied to the output directory so the HTML is self-contained.
"""

import os
import sys
import json
import math
import shutil
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

import numpy as np
import pandas as pd

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[3]))

TEMPLATE_DIR = LocDir.parent / "templates"
STATIC_DIR   = LocDir.parent / "static"

try:
    from configs import logger, _cfg
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    _cfg = None


# ================================================================
# JINJA2 ENVIRONMENT
# ================================================================

def get_env() -> Environment:
    """Initialize Jinja2 environment."""
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(
            loader        = FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape    = select_autoescape(["html", "xml"]),
            trim_blocks   = True,
            lstrip_blocks = True,
        )
        logger.info("Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing Jinja environment.", exc_info=True)
        raise RuntimeError("Jinja environment initialization failed.") from exc


# ================================================================
# STATIC ASSET COPYING
# ================================================================

def copy_static_assets(output_dir: Path) -> Dict[str, str]:
    """Copy all static assets (css/, js/, vendor/) into output directory."""
    logger.debug("Copying static assets to output directory.")
    dest_static = output_dir / "static"

    if dest_static.exists():
        shutil.rmtree(dest_static)

    try:
        shutil.copytree(STATIC_DIR, dest_static, dirs_exist_ok=False)
        logger.info("Static assets copied to %s", dest_static)
    except Exception as exc:
        logger.error("Failed copying static assets.", exc_info=True)
        raise RuntimeError("Static asset copy failed.") from exc

    return {
        "static_css":    "static/css",
        "static_js":     "static/js",
        "static_vendor": "static/vendor",
        "static_img":    "static",
    }


# ================================================================
# HELPERS
# ================================================================

def beautify_label(label: str) -> str:
    try:
        label = label.replace("_", " ")
        replacements = {
            "ndcg": "NDCG", "map": "MAP", "mrr": "MRR",
            "auc": "AUC", "ctr": "CTR", "at": "@",
            "precision": "Precision", "recall": "Recall",
            "latency": "Latency", "throughput": "Throughput", "qps": "QPS",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label.title()
    except Exception:
        return str(label)


def safe_float(value: Any, precision: int = 4) -> str:
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            return f"{value:.{precision}f}"
        return str(value)
    except Exception:
        return str(value)


def detect_gauge_metric(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    keywords = ["ndcg", "map", "mrr", "precision", "recall", "accuracy", "auc", "f1", "coverage"]
    return any(k in metric_name for k in keywords)


# ================================================================
# CONTEXT BUILDERS
# ================================================================

def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not metrics:
        return []
    cards = []
    color_cycle = ["blue", "green", "purple", "orange", "cyan", "red"]
    icon_map = {
        "blue":   "fas fa-chart-line",
        "green":  "fas fa-network-wired",
        "purple": "fas fa-bullseye",
        "orange": "fas fa-gauge-high",
        "cyan":   "fas fa-desktop",
        "red":    "fas fa-chart-column",
    }
    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        color = color_cycle[idx % len(color_cycle)]
        cards.append({
            "label": beautify_label(metric_name),
            "value": safe_float(metric_value),
            "sub": "Inference metric",
            "color": color,
            "icon_class": icon_map[color],
        })
    logger.info("Generated %d scorecards.", len(cards))
    return cards


def generate_gauges(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    gauges = []
    for metric_name, metric_value in metrics.items():
        if not detect_gauge_metric(metric_name):
            continue
        try:
            percent = float(metric_value) * 100
        except (ValueError, TypeError):
            continue
        gauges.append({
            "label": beautify_label(metric_name),
            "value": metric_value,
            "display": f"{percent:.2f}%",
            "percent": round(percent, 2),
        })
    logger.info("Generated %d gauges.", len(gauges))
    return gauges


def generate_stat_minis(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = []
    inf = context.get("inference_statistics", {})
    stats.append({"label": "Predictions",  "value": str(inf.get("n_predictions", 0)),                     "percent": 100})
    stats.append({"label": "Users Served", "value": str(inf.get("n_users_served", 0)),                    "percent": 100})
    stats.append({"label": "Avg Latency",  "value": f"{inf.get('avg_latency_ms', 0):.2f} ms",             "percent": 100})
    stats.append({"label": "Coverage",     "value": f"{inf.get('coverage', 0):.2%}",                       "percent": 100})
    logger.info("Generated %d mini stat cards.", len(stats))
    return stats


def normalize_charts(charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not charts:
        return []
    normalized = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        normalized.append({
            "title": chart.get("title", "Untitled Chart"),
            "type":  chart.get("type"),
            "data":  chart.get("data"),
            "full":  "importance" in chart.get("title", "").lower(),
        })
    return normalized


def build_inference_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build full inference dashboard rendering context."""
    logger.debug("Entering build_inference_context().")
    try:
        context = deepcopy(context_data)
        metrics = context.get("metrics", {})

        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"]     = generate_gauges(metrics)
        context["stat_minis"] = generate_stat_minis(context)
        context["charts"]     = normalize_charts(context.get("charts", []))
        context["bar_labels"] = list(metrics.keys())  if metrics else []
        context["bar_data"]   = list(metrics.values()) if metrics else []

        # Predictions
        predictions = context.get("predictions", [])
        if predictions:
            pred_df = pd.DataFrame(predictions)
            context["rankings"]       = pred_df.to_dict(orient="records")
            context["total_rankings"] = int(pred_df.shape[0])
        else:
            context["rankings"]       = []
            context["total_rankings"] = 0

        context.setdefault("generated_at",    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title",      "AryColBring Inference Dashboard")
        context.setdefault("subtitle",        "Production Inference Monitoring")
        context.setdefault("experiment_name", context.get("experiment_name", "Default Experiment"))
        context.setdefault("model_version",   context.get("model_version", "1.0.0"))
        context.setdefault("batch_size",      context.get("batch_size", 100))
        context.setdefault("num_threads",     context.get("num_threads", 4))

        # Overall score
        try:
            gauge_values = [float(v) for k, v in metrics.items() if detect_gauge_metric(k)]
            osv = round(sum(gauge_values) / len(gauge_values) * 100, 2) if gauge_values else 0
            context["overall_score_percent"] = osv
        except (ValueError, TypeError):
            context["overall_score_percent"] = 0
        context["overall_score"] = f'{context["overall_score_percent"]}%'

        context.setdefault("inference_params", {
            "batch_size":    context.get("batch_size", 100),
            "num_threads":   context.get("num_threads", 4),
            "cache_enabled": context.get("cache_enabled", True),
        })

        logger.info("Inference context built: %d metrics, %d predictions, %d gauges.",
                     len(metrics), context["total_rankings"], len(context["gauges"]))
        return context

    except Exception as exc:
        logger.error("Context building failed.", exc_info=True)
        raise RuntimeError("Failed building inference context.") from exc


# ================================================================
# RENDERING
# ================================================================

def render_inference_report(context: Dict[str, Any], output_path: Path) -> str:
    """Render inference dashboard HTML using base.html.j2."""
    logger.debug("Rendering inference report.")
    try:
        env = get_env()
        template = env.get_template("base.html.j2")
        html = template.render(**context)
        logger.info("Rendered HTML size = %.2f KB", len(html.encode("utf-8")) / 1024)
        return html
    except Exception as exc:
        logger.error("Rendering failed.", exc_info=True)
        raise RuntimeError("Inference dashboard rendering failed.") from exc


# ================================================================
# MAIN PIPELINE
# ================================================================

def generate_inference_report(
    context_data : Dict[str, Any],
    output_dir   : Optional[str | Path] = None,
    report_name  : Optional[str]        = None,
) -> Path:
    """
    Full pipeline:
      1. Build context from inference data
      2. Copy static assets (css, js, vendor) to output
      3. Render HTML from Jinja2 templates
      4. Write output file
    """
    logger.debug("Starting inference report generation pipeline.")

    try:
        # 1. Output directory
        if output_dir is None:
            output_dir = LocDir.parent / "output"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug("Output directory = %s", output_dir)

        # 2. Report filename
        if report_name is None:
            datepf      = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"{datepf}_Inference_Report.html"
        output_path = output_dir / report_name

        # 3. Copy static assets
        static_paths = copy_static_assets(output_dir)

        # 4. Build context
        context = build_inference_context(context_data)
        context.update(static_paths)

        # 5. Save context JSON (debug mode)
        dlevel = True
        if _cfg:
            dlevel = _cfg.get("logging", "level", fallback="DEBUG") in ["DEBUG", "INFO"]
        if dlevel:
            ctx_json_path = output_dir / "InferenceContext.json"
            with open(ctx_json_path, "w", encoding="utf-8") as fx:
                json.dump(context, fx, ensure_ascii=False, indent=2, default=str)
            logger.debug("Context JSON saved to %s", ctx_json_path)

        # 6. Render HTML
        html = render_inference_report(context=context, output_path=output_path)

        # 7. Write output
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Inference dashboard generated: %s", output_path)
        logger.info("  Metrics:     %d", len(context.get("metrics", {})))
        logger.info("  Predictions: %d", context.get("total_rankings", 0))
        logger.info("  Static:      %s", output_dir / "static")

        return output_path

    except Exception as exc:
        logger.error("Pipeline failed.", exc_info=True)
        raise RuntimeError("Inference report generation failed.") from exc


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate inference dashboard HTML report.")
    parser.add_argument("-i", "--input",  type=str, help="Path to InferenceContext.json")
    parser.add_argument("-o", "--output", type=str, help="Output directory", default=None)
    parser.add_argument("-n", "--name",   type=str, help="Output HTML filename", default=None)
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
        print(f"Loaded context from: {input_path}")
    else:
        context_data = {
            "metrics": {
                "precision_at_10": 0.245,
                "recall_at_10": 0.167,
                "auc": 0.801,
                "mrr": 0.356,
                "ndcg_at_10": 0.438,
                "coverage": 0.75,
                "avg_latency_ms": 12.5,
                "qps": 850,
            },
            "inference_statistics": {
                "n_predictions": 50000,
                "n_users_served": 5000,
                "avg_latency_ms": 12.5,
                "coverage": 0.75,
            },
            "model_version": "1.0.0",
            "batch_size": 100,
            "num_threads": 4,
            "experiment_name": "Production Inference Run",
            "predictions": [
                {"user_id": 1, "item_id": 100, "score": 0.95, "rank": 1},
                {"user_id": 1, "item_id": 200, "score": 0.87, "rank": 2},
                {"user_id": 2, "item_id": 150, "score": 0.92, "rank": 1},
                {"user_id": 3, "item_id": 300, "score": 0.45, "rank": 1},
                {"user_id": 3, "item_id": 100, "score": 0.38, "rank": 2},
            ],
            "charts": [
                {"title": "Latency Distribution", "type": "histogram", "data": [5, 10, 15, 20, 25]},
                {"title": "QPS Over Time", "type": "line", "data": [800, 820, 850, 870, 890]},
            ],
        }
        print("No input file specified, using sample data.")

    output = generate_inference_report(
        context_data=context_data,
        output_dir=args.output,
        report_name=args.name,
    )
    print(f"\nInference report generated: {output}")
