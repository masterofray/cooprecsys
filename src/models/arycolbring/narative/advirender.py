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
advirender.py
------------------
Training dashboard renderer for AryColBring collaborative filtering model.

Generates comprehensive HTML reports for the training phase including:
- Training metrics (loss curves, convergence)
- Evaluation metrics (Precision@K, Recall@K, AUC, MRR)
- Model hyperparameters
- Data statistics
- Embedding visualizations

Modular structure:
  templates/base.html.j2        — layout (sidebar, tabs, footer)
  templates/overview.html.j2    — scorecards, charts, gauges
  templates/rankings.html.j2    — filterable table
  templates/diagnostics.html.j2 — metrics comparison, history
  templates/config.html.j2      — config tables

Static assets are copied to the output directory so the HTML is self-contained.
"""

import os
import sys
import json
import math
import shutil
import numpy  as np
import pandas as pd
from pathlib  import Path
from copy     import deepcopy
from datetime import datetime
from typing   import Any, Dict, List, Optional
from jinja2   import Environment, FileSystemLoader, select_autoescape

LocDir    = Path(__file__).parent.resolve()
Tplatedir = LocDir / "templates"
Staticdir = LocDir / "sttrain"

sys.path.append(str(LocDir.parents[2]))
from configs import logger, _cfg


# ================================================================
# JINJA2 ENVIRONMENT
# ================================================================
def get_env() -> Environment:
    """Initialize Jinja2 environment."""
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(
            loader        = FileSystemLoader(str(Tplatedir)),
            autoescape    = select_autoescape(["html", "xml"]),
            trim_blocks   = True,
            lstrip_blocks = True,
        )
        logger.debug("Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing Jinja environment.", exc_info = True)
        raise RuntimeError() from exc


# ================================================================
# STATIC ASSET COPYING
# ================================================================
def copy_static_assets(output_dir: Path) -> Dict[str, str]:
    """
    Copy all static assets (css/, js/, vendor/) into the output directory
    and return relative paths for Jinja2 template variables.
    """
    logger.debug("Copying static assets to output directory.")
    dest_static = output_dir / "static"

    # Remove old static dir if exists (clean copy)
    if dest_static.exists():
        shutil.rmtree(dest_static)

    try:
        # Copy entire static tree (css/, js/, vendor/ + vendor/webfonts/)
        shutil.copytree(Staticdir, dest_static, dirs_exist_ok=False)
        logger.info("Static assets copied to %s", dest_static)
    except Exception as exc:
        logger.error("Failed copying static assets.", exc_info=True)
        raise RuntimeError("Static asset copy failed.") from exc

    # Return relative paths from output HTML to static dirs
    paths = {
        "static_css":    "static/css",
        "static_js":     "static/js",
        "static_vendor": "static/vendor",
        "static_img":    "static",
    }
    logger.debug("Static paths: %s", paths)
    return paths


# ================================================================
# LABEL / VALUE HELPERS
# ================================================================

def beautify_label(label: str) -> str:
    """Beautify metric labels for display."""
    try:
        label = label.replace("_", " ")
        replacements = {
            "ndcg": "NDCG", "map": "MAP", "mrr": "MRR",
            "auc": "AUC", "ctr": "CTR", "at": "@",
            "precision": "Precision", "recall": "Recall",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label.title()
    except Exception:
        return str(label)


def safe_float(value: Any, precision: int = 4) -> str:
    """Safely format numeric values."""
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            return f"{value:.{precision}f}"
        return str(value)
    except Exception:
        return str(value)


def detect_gauge_metric(metric_name: str) -> bool:
    """Detect whether metric should become gauge widget."""
    metric_name = metric_name.lower()
    keywords = ["ndcg", "map", "mrr", "precision", "recall", "accuracy", "auc", "f1"]
    return any(k in metric_name for k in keywords)


# ================================================================
# CONTEXT BUILDERS
# ================================================================

def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate scorecards dynamically from metrics."""
    if not metrics:
        return []
    cards = []
    color_cycle = ["blue", "green", "purple", "orange", "cyan", "red"]
    icon_map = {
        "blue": "fas fa-chart-line",
        "green": "fas fa-network-wired",
        "purple": "fas fa-bullseye",
        "orange": "fas fa-gauge-high",
        "cyan": "fas fa-desktop",
        "red": "fas fa-chart-column",
    }
    try:
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            color = color_cycle[idx % len(color_cycle)]
            cards.append({
                "label": beautify_label(metric_name),
                "value": safe_float(metric_value),
                "sub": "Training metric",
                "color": color,
                "icon_emoji": f'<i class="{icon_map[color]}"></i>',
            })
        logger.info("Generated %d scorecards.", len(cards))
        return cards
    except Exception as exc:
        logger.error("Scorecard generation failed.", exc_info=True)
        raise RuntimeError("Failed generating scorecards.") from exc


def generate_gauges(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate gauge widgets dynamically."""
    gauges = []
    try:
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
    except Exception as exc:
        logger.error("Gauge generation failed.", exc_info=True)
        raise RuntimeError("Failed generating gauges.") from exc


def generate_stat_minis(training_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate mini statistics cards."""
    stats = []
    try:
        data_stats = training_context.get("data_statistics", {})
        stats.append({"label": "Users",        "value": str(data_stats.get("n_users", 0)),        "percent": 100})
        stats.append({"label": "Items",        "value": str(data_stats.get("n_items", 0)),        "percent": 100})
        stats.append({"label": "Interactions", "value": str(data_stats.get("n_interactions", 0)), "percent": 100})
        sparsity = data_stats.get("sparsity", data_stats.get("density", 0))
        stats.append({"label": "Sparsity",     "value": f"{sparsity:.2%}",                        "percent": 100})
        logger.info("Generated %d mini stat cards.", len(stats))
    except Exception as exc:
        logger.error("Mini stat generation failed.", exc_info=True)
    return stats


def normalize_charts(charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize chart metadata."""
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


def build_training_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build full training dashboard rendering context."""
    logger.debug("Entering build_training_context().")
    try:
        context = deepcopy(context_data)
        metrics = context.get("metrics", {})

        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"]     = generate_gauges(metrics)
        context["stat_minis"] = generate_stat_minis(context)
        context["charts"]     = normalize_charts(context.get("charts", []))
        context["bar_labels"] = list(metrics.keys())  if metrics else []
        context["bar_data"]   = list(metrics.values()) if metrics else []

        context.setdefault("generated_at",    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title",      "AryColBring Training Dashboard")
        context.setdefault("subtitle",        "Collaborative Filtering Training Report")
        context.setdefault("experiment_name", context.get("experiment_name", "Default Experiment"))
        context.setdefault("model_type",      "Matrix Factorization")
        context.setdefault("loss_function",   context.get("loss", "logistic"))
        context.setdefault("epochs",          context.get("epochs", 0))
        context.setdefault("no_components",   context.get("no_components", 10))
        context.setdefault("training_time_sec", context.get("training_time_sec", 0))

        # Overall score
        try:
            gauge_values = [float(v) for k, v in metrics.items() if detect_gauge_metric(k)]
            osv = round(sum(gauge_values) / len(gauge_values) * 100, 2) if gauge_values else 0
            context["overall_score_percent"] = osv
        except (ValueError, TypeError):
            context["overall_score_percent"] = 0
        context["overall_score"] = f'{context["overall_score_percent"]}%'

        context.setdefault("training_params", {
            "learning_rate":     context.get("learning_rate", 0.05),
            "item_alpha":        context.get("item_alpha", 0.0),
            "user_alpha":        context.get("user_alpha", 0.0),
            "learning_schedule": context.get("learning_schedule", "adagrad"),
        })

        logger.info("Training context built: %d metrics, %d scorecards, %d gauges.",
                     len(metrics), len(context["scorecards"]), len(context["gauges"]))
        return context

    except Exception as exc:
        logger.error("Context building failed.", exc_info=True)
        raise RuntimeError("Failed building training context.") from exc


# ================================================================
# RENDERING
# ================================================================

def render_training_report(context: Dict[str, Any], output_path: Path) -> str:
    """Render training dashboard HTML using base.html.j2."""
    logger.debug("Rendering training report.")
    try:
        env = get_env()
        template = env.get_template("base.html.j2")
        html = template.render(**context)
        logger.info("Rendered HTML size = %.2f KB", len(html.encode("utf-8")) / 1024)
        return html
    except Exception as exc:
        logger.error("Rendering failed.", exc_info=True)
        raise RuntimeError("Training dashboard rendering failed.") from exc


# ================================================================
# MAIN PIPELINE
# ================================================================

def generate_training_report(
    context_data : Dict[str, Any],
    output_dir   : Optional[str | Path] = None,
    report_name  : Optional[str]        = None,
) -> Path:
    """
    Full pipeline:
      1. Build context from training data
      2. Copy static assets (css, js, vendor) to output
      3. Render HTML from Jinja2 templates
      4. Write output file
    """
    logger.debug("Starting training report generation pipeline.")

    try:
        # ── 1. Output directory ──
        if output_dir is None:
            output_dir = LocDir.parent / "output"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug("Output directory = %s", output_dir)

        # ── 2. Report filename ──
        if report_name is None:
            datepf      = datetime.now().strftime("%Y%m%d")
            report_name = f"{datepf}_Train_Report.html"
        output_path = output_dir / report_name

        # ── 3. Copy static assets ──
        static_paths = copy_static_assets(output_dir)

        # ── 4. Build context ──
        context = build_training_context(context_data)
        context.update(static_paths)

        # ── 5. Save context JSON (debug mode) ──
        dlevel = True
        if _cfg:
            dlevel = _cfg.get("logging", "level", fallback="DEBUG") in ["DEBUG", "INFO"]
        if dlevel:
            ctx_json_path = output_dir / "TrainingContext.json"
            with open(ctx_json_path, "w", encoding="utf-8") as fx:
                json.dump(context, fx, ensure_ascii=False, indent=2, default=str)
            logger.debug("Context JSON saved to %s", ctx_json_path)

        # ── 6. Render HTML ──
        html = render_training_report(context=context, output_path=output_path)

        # ── 7. Write output ──
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Training dashboard generated: %s", output_path)
        logger.info("  Metrics: %d", len(context.get("metrics", {})))
        logger.info("  Charts:  %d", len(context.get("charts", [])))
        logger.info("  Static:  %s", output_dir / "static")

        return output_path

    except Exception as exc:
        logger.error("Pipeline failed.", exc_info=True)
        raise RuntimeError("Training report generation failed.") from exc


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training dashboard HTML report.")
    parser.add_argument("-i", "--input",  type=str, help="Path to TrainingContext.json")
    parser.add_argument("-o", "--output", type=str, help="Output directory", default=None)
    parser.add_argument("-n", "--name",   type=str, help="Output HTML filename", default=None)
    args = parser.parse_args()

    # Load context data
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
        print(f"Loaded context from: {input_path}")
    else:
        # Sample data for testing
        context_data = {
            "metrics": {
                "precision_at_10": 0.234,
                "recall_at_10": 0.156,
                "auc": 0.789,
                "mrr": 0.345,
                "ndcg_at_10": 0.421,
            },
            "data_statistics": {
                "n_users": 10000,
                "n_items": 5000,
                "n_interactions": 150000,
                "sparsity": 0.997,
            },
            "loss": "warp",
            "epochs": 20,
            "no_components": 32,
            "learning_rate": 0.05,
            "experiment_name": "Sample Training Run",
            "charts": [
                {"title": "Loss Curve", "type": "line", "data": [1.0, 0.8, 0.6, 0.4, 0.2]},
                {"title": "Precision@K", "type": "bar", "data": [0.1, 0.2, 0.3, 0.4]},
            ],
        }
        print("No input file specified, using sample data.")

    output = generate_training_report(
        context_data=context_data,
        output_dir=args.output,
        report_name=args.name,
    )
    print(f"\nReport generated: {output}")
