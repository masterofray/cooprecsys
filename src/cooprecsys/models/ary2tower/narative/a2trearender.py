#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
a2trearender.py
_________________________________________
ary2tower's inference dashboard renderer. Same context-building shape
as arycolbring/narative/rearender.py, reusing the SAME shared helpers
(src/assets/dashboard_utils.py, src/assets/vizdata.py) -- only real
production-inference metrics are shown here (latency/qps/throughput),
never ranking-quality metrics (precision/recall/ndcg/auc/mrr) or a
fabricated "coverage" number; see the Task 1 dashboard fix this
mirrors for why those two categories must stay separate.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from ....configs import logger
from .rensupport import get_env, copy_static, OUTPUT_DIR

try:
    from ....assets import (generate_scorecards as _gen_scorecards,
                            generate_gauges, normalize_charts, overall_score_percent,
                            score_distribution, embedding_projection_2d,
                            similarity_heatmap, top_k_similar_items)
except ImportError:  # pragma: no cover - fallback for standalone/test use
    from src.assets import (generate_scorecards as _gen_scorecards,
                            generate_gauges, normalize_charts, overall_score_percent,
                            score_distribution, embedding_projection_2d,
                            similarity_heatmap, top_k_similar_items)


def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _gen_scorecards(metrics, sub_label="Inference metric")


def generate_stat_minis(inference_statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"label": "Predictions", "value": str(inference_statistics.get("n_predictions", 0))},
        {"label": "Users Served", "value": str(inference_statistics.get("n_users_served", 0))},
        {"label": "Avg Latency", "value": f"{inference_statistics.get('avg_latency_ms', 0):.2f} ms"},
        {"label": "Throughput", "value": f"{inference_statistics.get('throughput_preds_per_sec', 0):.1f}/s"},
    ]


def build_visualizations(context: Dict[str, Any]) -> Dict[str, Any]:
    """Same three widgets as arycolbring's Insights tab: score
    histogram, 2D embedding projection, item-item similarity heatmap --
    all optional/None-able, same graceful-empty-state behavior."""
    viz: Dict[str, Any] = {"score_distribution": None, "embedding_plot": None,
                           "similarity_heatmap": None}

    predictions = context.get("predictions", [])
    if predictions:
        scores = [p.get("score") for p in predictions if p.get("score") is not None]
        if scores:
            viz["score_distribution"] = score_distribution(scores)

    embeddings = context.get("embeddings")
    if embeddings and embeddings.get("vectors") is not None:
        vectors = np.asarray(embeddings["vectors"])
        if vectors.size:
            viz["embedding_plot"] = embedding_projection_2d(
                vectors, ids=embeddings.get("ids"), highlight_id=embeddings.get("highlight_id"))

    item_embeddings = context.get("item_embeddings")
    item_ids = context.get("item_ids")
    if item_embeddings is not None and item_ids:
        vectors = np.asarray(item_embeddings)
        if vectors.size:
            viz["similarity_heatmap"] = similarity_heatmap(vectors, ids=item_ids, top_n=20)
            context["_similar_items_map"] = top_k_similar_items(vectors, list(item_ids), k=3)

    return viz


def build_inference_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the full ary2tower inference-report context. Only real
    production metrics go in `metrics` here -- see the module
    docstring."""
    logger.debug("Entering ary2tower build_inference_context().")
    try:
        context = deepcopy(context_data)
        metrics = context.get("metrics", {})

        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"] = generate_gauges(metrics)
        context["stat_minis"] = generate_stat_minis(context.get("inference_statistics", {}))
        context["charts"] = normalize_charts(context.get("charts", []))

        context.update(build_visualizations(context))
        similar_items_map = context.pop("_similar_items_map", {})

        predictions = context.get("predictions", [])
        if predictions:
            pred_df = pd.DataFrame(predictions)
            if similar_items_map and "item_id" in pred_df.columns:
                pred_df["similar_items"] = pred_df["item_id"].map(
                    lambda iid: similar_items_map.get(iid, []))
            context["rankings"] = pred_df.to_dict(orient="records")
            context["total_rankings"] = int(pred_df.shape[0])
        else:
            context["rankings"] = list()
            context["total_rankings"] = 0

        context.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title", "ary2tower Inference Dashboard")
        context.setdefault("subtitle", "Two-Tower Model — Production Inference Monitoring")
        context.setdefault("experiment_name", "Default Experiment")
        context.setdefault("theme_css", "supportinfer.css")
        context.setdefault("theme_js", "supportinfer.js")

        context["overall_score_percent"] = overall_score_percent(metrics)

        logger.info("ary2tower inference context built: %d metric(s), %d prediction(s).",
                    len(metrics), context["total_rankings"])
        return context
    except Exception as exc:
        logger.error("ary2tower context building failed.", exc_info=True)
        raise RuntimeError("Failed building ary2tower inference context.") from exc


def render_inference_report(context: Dict[str, Any], output_path: Path) -> str:
    """Render + write the inference HTML report to `output_path`."""
    env = get_env()
    template = env.get_template("a2t_inference.html")
    html = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    copy_static(output_path.parent)
    return html


def generate_inference_report(context_data: Dict[str, Any],
                              output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Build the context and render the inference report to disk.
    Returns the written file's path."""
    context = build_inference_context(context_data)
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_ary2tower_Inference_Report.html"
    output_path = out_dir / filename
    render_inference_report(context, output_path)
    logger.info("ary2tower inference report written to %s", output_path)
    return output_path
