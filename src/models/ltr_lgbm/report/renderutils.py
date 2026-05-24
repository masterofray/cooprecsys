#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-23"

import os
import sys
import json
import math
import pandas as pd
from pathlib  import Path
from datetime import datetime
from typing   import Any, Dict, List, Optional
from jinja2   import (Environment, FileSystemLoader,
                     select_autoescape)

LocDir = Path(__file__).resolve()
TEMPLATE_DIR = LocDir.parent / "templates"
STATIC_DIR   = LocDir.parent / "static"

sys.path.append(str(LocDir.parents[3]))
from configs import logger

# JINJA ENVIRONMENT
#__________________________________________________________
def get_env() -> Environment:
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(loader        = FileSystemLoader(TEMPLATE_DIR),
                          autoescape    = select_autoescape(["html", "xml"]),
                          trim_blocks   = True,
                          lstrip_blocks = True)
        logger.info("Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing Jinja environment.", exc_info = True)
        raise RuntimeError("Jinja environment initialization failed.") from exc

def static_prefix(output_path: Optional[str | Path] = None) -> Dict[str, str]:
    """Generate relative static asset paths.
    Returns will type as Dict[str, str] as 
    Static asset mappings."""
    logger.debug(
    "Generating static asset prefix for output_path = %s", output_path)
    try:
        if output_path is None:
            rel_static = Path("../static")
        else:
            output_dir = Path(output_path).resolve().parent
            rel_static = Path(os.path.relpath(STATIC_DIR, output_dir))
        paths = {"static_css" : (rel_static / "css").as_posix(),
                 "static_js"  : (rel_static / "js").as_posix(),
                 "static_img" : rel_static.as_posix()}
        logger.debug("Static paths generated: %s",paths)
        return paths
    except Exception as exc:
        logger.error("Failed generating static prefixes.", exc_info = True)
        raise RuntimeError("Static asset prefix generation failed."
        ) from exc


# LABEL BEAUTIFIER
#__________________________________________________________
def beautify_label(label: str) -> str:
    logger.debug("Beautifying label: %s", label)
    try:
        label = label.replace("_", " ")
        replacements = {"ndcg": "NDCG",
                        "map" : "MAP",
                        "mrr" : "MRR",
                        "auc" : "AUC",
                        "ctr" : "CTR",
                        "gat" : "@"}
        for old, new in replacements.items():
            label = label.replace(old, new)
        beautified = label.title()
        logger.debug("Beautified label result: %s", beautified)
        return beautified
    except Exception as exc:
        logger.error("Failed beautifying label = %s",
            label, exc_info = True)
        return str(label)


# SAFE FLOAT FORMATTER
#__________________________________________________________
def safe_float(value: Any, precision: int = 4) -> str:
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


# GAUGE DETECTION
#__________________________________________________________
def detect_gauge_metric(metric_name: str) -> bool:
    """Detect whether metric should become gauge widget."""
    logger.debug("Detecting gauge metric for key = %s", metric_name)
    metric_name = metric_name.lower()
    keywords    = ["ndcg", "map", "mrr", "precision",
                   "recall", "accuracy", "auc", "f1"]
    result      = any(k in metric_name for k in keywords)
    logger.debug("Gauge detection result=%s for metric = %s",
                 result, metric_name)
    return result


# CONTEXT LOADER
#__________________________________________________________
def load_context(json_path: str | Path) -> Dict[str, Any]:
    """Load dashboard context JSON."""
    logger.debug("Entering load_context(json_path = %s)",json_path)
    json_path = Path(json_path)
    if not json_path.exists():
        logger.error("Context JSON not found: %s", json_path)
        raise FileNotFoundError(f"Context JSON missing: {json_path}")
    try:
        logger.debug("Opening JSON context file.")
        with open(json_path, "r", encoding = "utf-8") as f:
            context = json.load(f)
        logger.info("Context JSON loaded successfully.")
        logger.debug("Context keys: %s", list(context.keys()))
        return context
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON structure detected.", exc_info = True)
        raise ValueError(f"Malformed JSON: {exc}") from exc
    except PermissionError as exc:
        logger.error("Permission denied reading context JSON.", exc_info = True)
        raise RuntimeError("Permission denied reading JSON.") from exc
    except Exception as exc:
        logger.error("Unexpected context loading failure.", exc_info = True)
        raise RuntimeError("Failed loading dashboard context.") from exc


# DATAFRAME RECONSTRUCTION
#__________________________________________________________
def load_prediction_dataframe(context: Dict[str, Any]) -> pd.DataFrame:
    logger.debug("Entering load_prediction_dataframe().")
    prediction_data = context.get("predictiondata")
    if prediction_data is None:
        logger.warning("'predictiondata' missing from context.")
        return pd.DataFrame()
    try:
        logger.debug("Predictiondata type=%s", type(prediction_data))
        if not isinstance(prediction_data, list):
            raise ValueError("predictiondata must be list[dict].")
        df = pd.DataFrame(prediction_data)
        logger.info("Prediction dataframe reconstructed successfully.")
        logger.debug("Prediction dataframe shape=%s", df.shape)
        logger.debug("Prediction dataframe columns=%s", list(df.columns))
        return df
    except ValueError:
        logger.error("Invalid predictiondata structure.", exc_info=True)
        raise
    except Exception as exc:
        logger.error("Unexpected dataframe reconstruction failure.", exc_info=True)
        raise RuntimeError("Prediction dataframe reconstruction failed.") from exc


# SCORECARD GENERATOR
#__________________________________________________________
def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate scorecards dynamically.
    Args:
        metrics:
            Metrics dictionary.
    Returns:
        List[Dict[str, Any]]:
            Scorecards list.
    """
    logger.debug("Generating scorecards from metrics.")
    if not metrics:
        logger.warning("Empty metrics dictionary detected.")
        return list()
    cards: List[Dict[str, Any]] = list()
    color_cycle = ["blue", "green", "purple", "orange", "cyan", "red"]
    try:
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            logger.debug("Generating scorecard for metric=%s", metric_name)
            cards.append({ "label" : beautify_label(metric_name),
                           "value" : safe_float(metric_value),
                           "sub"   : "Auto-generated metric",
                           "color" : color_cycle[idx % len(color_cycle)],
                           "icon"  : "fas fa-chart-line"})
        logger.info("Generated %d scorecards.", len(cards))
        return cards
    except Exception as exc:
        logger.error("Scorecard generation failed.", exc_info=True)
        raise RuntimeError("Failed generating scorecards.") from exc


# GAUGE GENERATOR
#__________________________________________________________
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


# MINI STAT GENERATOR
#__________________________________________________________
def generate_stat_minis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate mini statistics cards."""
    logger.debug("Generating stat mini cards.")
    if df.empty:
        logger.warning("Prediction dataframe empty.")
        return list()
    stats: List[Dict[str, Any]] = list()
    try:
        candidate_columns = {"Users"      : "CustomerID",
                             "Products"   : "ProductID",
                             "Categories" : "CategoryID"}
        for label, column in candidate_columns.items():
            logger.debug("Evaluating stat column=%s", column)
            if column not in df.columns:
                logger.warning("Column=%s missing from dataframe.", column)
                continue
            stats.append({"label"   : label,
                          "value"   : str(df[column].nunique()),
                          "percent" : 100})
        stats.append({"label"   : "Rows",
                      "value"   : str(len(df)),
                      "percent" : 100})
        logger.info("Generated %d mini stat cards.", len(stats))
        return stats
    except Exception as exc:
        logger.error("Mini stat generation failed.", exc_info=True)
        raise RuntimeError("Failed generating mini stats.") from exc


# CHART NORMALIZER
#__________________________________________________________
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
                "image": chart.get("image"),
                "type" : chart.get("type"),
                "data" : chart.get("data"),
                "full" : "importance" in chart.get("title", "").lower()})
        logger.info("Normalized %d charts.", len(normalized))
        return normalized
    except Exception as exc:
        logger.error("Chart normalization failed.", exc_info=True)
        raise RuntimeError("Failed normalizing charts.") from exc
