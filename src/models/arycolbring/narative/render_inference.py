#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-31"


"""
render_inference.py
-------------------
Inference dashboard renderer for AryColBring collaborative filtering model.

Generates comprehensive HTML reports for the production inference phase including:
- Inference metrics (latency, throughput)
- Prediction statistics
- Top-N recommendation analysis
- User-item interaction patterns
- Model performance monitoring
"""

import os
import sys
import json
import math
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

import numpy as np
import pandas as pd

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[3]))
from configs import logger

TEMPLATE_DIR = LocDir / "templates"
STATIC_DIR   = LocDir / "static"

# Try to import from configs, fallback to basic logging
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


def get_env() -> Environment:
    """Initialize Jinja2 environment."""
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(
            loader        = FileSystemLoader(TEMPLATE_DIR), # nosec B701
            autoescape    = select_autoescape(["html", "xml"]),
            trim_blocks   = True,
            lstrip_blocks = True)
        logger.info("Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing Jinja environment.", exc_info=True)
        raise RuntimeError("Jinja environment initialization failed.") from exc


def static_prefix(output_path: Optional[str | Path] = None) -> Dict[str, str]:
    """Generate relative static asset paths."""
    logger.debug("Generating static asset prefix for output_path = %s", output_path)
    try:
        if output_path is None:
            rel_static = Path("../static")
        else:
            output_dir = Path(output_path).resolve().parent
            rel_static = Path(os.path.relpath(STATIC_DIR, output_dir))
        paths = {
            "static_css": (rel_static / "css").as_posix(),
            "static_js": (rel_static / "js").as_posix(),
            "static_img": rel_static.as_posix()
        }
        logger.debug("Static paths generated: %s", paths)
        return paths
    except Exception as exc:
        logger.error("Failed generating static prefixes.", exc_info=True)
        raise RuntimeError("Static asset prefix generation failed.") from exc


def beautify_label(label: str) -> str:
    """Beautify metric labels for display."""
    logger.debug("Beautifying label: %s", label)
    try:
        label = label.replace("_", " ")
        replacements = {
            "ndcg": "NDCG",
            "map": "MAP",
            "mrr": "MRR",
            "auc": "AUC",
            "ctr": "CTR",
            "at": "@",
            "precision": "Precision",
            "recall": "Recall",
            "latency": "Latency",
            "throughput": "Throughput",
            "qps": "QPS",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        beautified = label.title()
        logger.debug("Beautified label result: %s", beautified)
        return beautified
    except Exception:
        logger.warning("Failed beautifying label = %s", label)
        return str(label)


def safe_float(value: Any, precision: int = 4) -> str:
    """Safely format numeric values."""
    logger.debug("Formatting numeric value = %s", value)
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                logger.warning("NaN detected during float formatting.")
                return "NaN"
            return f"{value:.{precision}f}"
        return str(value)
    except Exception:
        logger.warning("Failed formatting numeric value = %s", value)
        return str(value)


def detect_gauge_metric(metric_name: str) -> bool:
    """Detect whether metric should become gauge widget."""
    logger.debug("Detecting gauge metric for key = %s", metric_name)
    metric_name = metric_name.lower()
    keywords = ["ndcg", "map", "mrr", "precision", "recall", "accuracy", "auc", "f1", "coverage"]
    result = any(k in metric_name for k in keywords)
    logger.debug("Gauge detection result=%s for metric = %s", result, metric_name)
    return result


def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate scorecards dynamically from metrics."""
    logger.debug("Generating scorecards from metrics.")
    if not metrics:
        logger.warning("Empty metrics dictionary detected.")
        return list()
    cards: List[Dict[str, Any]] = list()
    color_cycle = ["blue", "green", "purple", "orange", "cyan", "red"]
    try:
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            iconx = f'data_{idx+1}.ico'
            logger.debug("Generating scorecard for metric=%s", metric_name)
            cards.append({
                "label": beautify_label(metric_name),
                "value": safe_float(metric_value),
                "sub": "Inference metric",
                "color": color_cycle[idx % len(color_cycle)],
                "icon": iconx
            })
        logger.info("Generated %d scorecards.", len(cards))
        return cards
    except Exception as exc:
        logger.error("Scorecard generation failed.", exc_info=True)
        raise RuntimeError("Failed generating scorecards.") from exc


def generate_gauges(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate gauge widgets dynamically."""
    logger.debug("Generating gauge widgets.")
    gauges: List[Dict[str, Any]] = list()
    try:
        for metric_name, metric_value in metrics.items():
            logger.debug("Evaluating metric=%s for gauge generation.", metric_name)
            if not detect_gauge_metric(metric_name):
                logger.debug("Metric=%s skipped from gauges.", metric_name)
                continue
            try:
                percent = float(metric_value) * 100
            except ValueError:
                logger.warning("Gauge metric=%s not numeric.", metric_name)
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


def generate_stat_minis(inference_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate mini statistics cards for inference data."""
    logger.debug("Generating stat mini cards.")
    stats: List[Dict[str, Any]] = list()
    try:
        inference_stats = inference_context.get("inference_statistics", {})
        
        n_predictions = inference_stats.get("n_predictions", 0)
        n_users_served = inference_stats.get("n_users_served", 0)
        avg_latency_ms = inference_stats.get("avg_latency_ms", 0)
        coverage = inference_stats.get("coverage", 0)
        
        stats.append({"label": "Predictions", "value": str(n_predictions), "percent": 100})
        stats.append({"label": "Users Served", "value": str(n_users_served), "percent": 100})
        stats.append({"label": "Avg Latency", "value": f"{avg_latency_ms:.2f} ms", "percent": 100})
        stats.append({"label": "Coverage", "value": f"{coverage:.2%}", "percent": 100})
        
        logger.info("Generated %d mini stat cards.", len(stats))
    except Exception as exc:
        logger.error("Mini stat generation failed.", exc_info=True)
        raise RuntimeError("Failed generating mini stats.") from exc
    finally:
        return stats


def normalize_charts(charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize chart metadata."""
    logger.debug("Normalizing chart metadata.")
    if not charts:
        logger.warning("No charts supplied.")
        return list()
    normalized: List[Dict[str, Any]] = list()
    try:
        for idx, chart in enumerate(charts):
            logger.debug("Normalizing chart index=%d", idx)
            if not isinstance(chart, dict):
                logger.warning("Chart index=%d invalid type=%s", idx, type(chart))
                continue
            normalized.append({
                "title": chart.get("title", "Untitled Chart"),
                "type": chart.get("type"),
                "data": chart.get("data"),
                "full": "importance" in chart.get("title", "").lower()
            })
        logger.info("Normalized %d charts.", len(normalized))
        return normalized
    except Exception as exc:
        logger.error("Chart normalization failed.", exc_info=True)
        raise RuntimeError("Failed normalizing charts.") from exc


def build_inference_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build full inference dashboard rendering context."""
    logger.debug("Entering build_inference_context().")
    try:
        context = deepcopy(context_data)
        
        metrics = context.get("metrics", dict())
        logger.debug("Metrics count = %d", len(metrics))
        
        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"] = generate_gauges(metrics)
        context["stat_minis"] = generate_stat_minis(context)
        context["charts"] = normalize_charts(context.get("charts", []))
        context["bar_labels"] = list(metrics.keys()) if metrics else []
        context["bar_data"] = list(metrics.values()) if metrics else []
        
        # Add prediction rankings if available
        predictions = context.get("predictions", [])
        if predictions:
            pred_df = pd.DataFrame(predictions)
            context["rankings"] = pred_df.to_dict(orient="records")
            context["total_rankings"] = int(pred_df.shape[0])
        else:
            context["rankings"] = []
            context["total_rankings"] = 0
        
        context.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title", "AryColBring Inference Dashboard")
        context.setdefault("subtitle", "Production Inference Monitoring")
        context.setdefault("experiment_name", context.get("experiment_name", "Default Experiment"))
        context.setdefault("model_version", context.get("model_version", "1.0.0"))
        context.setdefault("batch_size", context.get("batch_size", 100))
        context.setdefault("num_threads", context.get("num_threads", 4))
        
        # Calculate overall score percent for main gauge
        try:
            mtep = context["bar_data"][4:]
            osv = round(sum(mtep)/len(mtep) * 100, 2) if mtep else 0
            context["overall_score_percent"] = osv
        except (ValueError, TypeError):
            context["overall_score_percent"] = 0
        context['overall_score'] = f'{context["overall_score_percent"]}%'
        
        context.setdefault("inference_params", {
            "batch_size": context.get("batch_size", 100),
            "num_threads": context.get("num_threads", 4),
            "cache_enabled": context.get("cache_enabled", True),
        })
        
        logger.info("Inference dashboard context built successfully.")
        return context
    
    except Exception as exc:
        logger.error("Context building failed.", exc_info=True)
        raise RuntimeError("Failed building inference dashboard context.") from exc


def render_inference_report(context: Dict[str, Any], output_path: str | Path) -> str:
    """Render inference dashboard HTML."""
    logger.debug("Entering render_inference_report().")
    try:
        env = get_env()
        logger.debug("Loading template=inference_report.html.j2")
        template = env.get_template("inference_report.html.j2")
        ctx = dict(context)
        
        logger.debug("Injecting static asset paths.")
        ctx.update(static_prefix(output_path))
        
        logger.debug("Rendering HTML template.")
        html = template.render(**ctx)
        
        logger.info("Inference dashboard rendered successfully.")
        logger.debug("Rendered HTML size=%.2f KB", len(html.encode("utf-8")) / 1024)
        return html
    
    except Exception as exc:
        logger.error("Unexpected rendering failure.", exc_info=True)
        raise RuntimeError("Inference dashboard rendering failed.") from exc


def generate_inference_report(
    context_data: Dict[str, Any],
    output_dir: Optional[str | Path] = None,
    report_name: Optional[str] = None,
) -> Path:
    """Main inference report generation pipeline."""
    logger.info("Starting inference report generation pipeline.")
    try:
        if output_dir is None:
            output_dir = LocDir.parent / "output"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug("Output directory ensured=%s", output_dir)
        
        if report_name is None:
            datepf = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"{datepf}_inference_report.html"
        
        output_path = output_dir / report_name
        
        context = build_inference_context(context_data)
        
        dlevel = True if _cfg and _cfg.get('logging', 'level', fallback='DEBUG') in ['DEBUG', 'INFO'] else False
        if dlevel:
            with open(output_dir / 'InferenceContext.json', 'w', encoding='utf-8') as fx:
                json.dump(context, fx, ensure_ascii=False, indent=2)
        
        html = render_inference_report(context=context, output_path=output_path)
        logger.debug("Writing rendered HTML to disk.")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.debug("Output HTML path=%s .\n\n", output_path)
        
        logger.info("Inference dashboard generated successfully.")
        logger.info("Dashboard path   = %s", output_path)
        logger.info("Total metrics    = %d", len(context.get("metrics", {})))
        logger.info("Total predictions= %d", context.get("total_rankings", 0))
        logger.info("Total charts     = %d", len(context.get("charts", [])))
        return output_path
    
    except Exception as exc:
        logger.error("Inference report generation pipeline failed.", exc_info=True)
        raise RuntimeError("Inference report generation failed.") from exc


if __name__ == "__main__":
    # Example usage
    sample_context = {
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
        ],
        "charts": [
            {"title": "Latency Distribution", "type": "histogram", "data": [5, 10, 15, 20, 25]},
            {"title": "QPS Over Time", "type": "line", "data": [800, 820, 850, 870, 890]},
        ]
    }
    output = generate_inference_report(sample_context)
    print(f"Inference report generated: {output}")
