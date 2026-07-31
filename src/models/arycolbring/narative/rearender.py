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
rearender.py
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

import json
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional
from .rensupport import Tplatedir, readir, OUTPUT_DIR, get_env, runcopy

LocDir = Path(__file__).resolve()
#sys.path.append(str(LocDir.parents[3]))
from ....configs import logger, _cfg
from ....assets  import (generate_scorecards as _gen_scorecards,
                         generate_gauges, normalize_charts,
                         overall_score_percent, score_distribution,
                         embedding_projection_2d, similarity_heatmap,
                         top_k_similar_items)

# NOTE: generate_scorecards/generate_gauges/normalize_charts/the overall
# score computation used to be defined here AND (near-identically) in
# advirender.py. Both copies now live once in
# src/assets/dashboard_utils.py -- see that module + its unit tests.
#
# generate_gauges() only ever produces a gauge for a *ranking/quality*
# metric (precision/recall/ndcg/auc/mrr/f1 -- see
# dashboard_utils.detect_gauge_metric). An inference report's `metrics`
# dict should only ever contain real production-performance numbers
# (qps, latency, throughput, ...), so in the normal case `gauges` here
# is correctly empty -- evaluation-quality metrics belong on the
# *training* dashboard (advirender.py), not this one.


def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inference-dashboard scorecards (thin wrapper: keeps this report's
    original "Inference metric" sub-label)."""
    return _gen_scorecards(metrics, sub_label="Inference metric")


def generate_stat_minis(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mini stat cards for the overview page.

    BUGFIX: this used to always include a "Coverage" card sourced from
    `inference_statistics['coverage']` -- but nothing in inference.py
    ever computed a real catalog-coverage number, so every report
    rendered a hardcoded 0.75 (75%) placeholder as if it were measured.
    Coverage is dropped until a real computation is wired up; showing a
    fabricated number is worse than not showing one.
    """
    stats = list()
    inf = context.get("inference_statistics", {})
    stats.append({"label": "Predictions",  "value": str(inf.get("n_predictions", 0)),         "percent": 100})
    stats.append({"label": "Users Served", "value": str(inf.get("n_users_served", 0)),         "percent": 100})
    stats.append({"label": "Avg Latency",  "value": f"{inf.get('avg_latency_ms', 0):.2f} ms",  "percent": 100})
    stats.append({"label": "Throughput",   "value": f"{inf.get('throughput_preds_per_sec', 0):.1f}/s", "percent": 100})
    logger.info("Generated %d mini stat cards.", len(stats))
    return stats


def build_visualizations(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build the payloads for the new Insights tab: a prediction-score
    histogram (always, if predictions are present), and -- only when the
    caller supplied embeddings -- a 2D embedding scatter and an
    item-item similarity heatmap. Each is optional and independently
    None-able so the template can render an empty-state per widget
    instead of the whole tab failing.
    """
    viz: Dict[str, Any] = {"score_distribution": None,
                           "embedding_plot": None,
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
                vectors,
                ids=embeddings.get("ids"),
                highlight_id=embeddings.get("highlight_id"))

    item_embeddings = context.get("item_embeddings")
    item_ids = context.get("item_ids")
    if item_embeddings is not None and item_ids:
        vectors = np.asarray(item_embeddings)
        if vectors.size:
            viz["similarity_heatmap"] = similarity_heatmap(
                vectors, ids=item_ids, top_n=20)
            context["_similar_items_map"] = top_k_similar_items(
                vectors, list(item_ids), k=3)

    return viz


def build_inference_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build full inference dashboard rendering context.

    Only real production-inference data goes on this dashboard: latency,
    throughput, QPS, prediction/embedding visualizations. Ranking-quality
    metrics (Precision@K, Recall@K, NDCG, AUC, MRR, ...) belong on the
    *training* dashboard (see advirender.py) -- if a caller passes them
    in `metrics` here anyway, `generate_gauges` will still surface them
    as gauges (the underlying data isn't wrong, just misplaced), but
    nothing in this pipeline fabricates or renames them.
    """
    logger.debug("Entering build_inference_context().")
    try:
        context = deepcopy(context_data)
        metrics = context.get("metrics", {})

        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"]     = generate_gauges(metrics)
        context["stat_minis"] = generate_stat_minis(context)
        context["charts"]     = normalize_charts(context.get("charts", []))
        context["bar_labels"] = list(metrics.keys())  if metrics else list()
        context["bar_data"]   = list(metrics.values()) if metrics else list()

        # New visualizations (replaces the removed "coverage" metric
        # and any other display of eval-only numbers on this report).
        context.update(build_visualizations(context))
        similar_items_map = context.pop("_similar_items_map", {})

        # Predictions
        predictions = context.get("predictions", [])
        if predictions:
            pred_df = pd.DataFrame(predictions)
            if similar_items_map and "item_id" in pred_df.columns:
                pred_df["similar_items"] = pred_df["item_id"].map(
                    lambda iid: similar_items_map.get(iid, []))
            context["rankings"]       = pred_df.to_dict(orient="records")
            context["total_rankings"] = int(pred_df.shape[0])
        else:
            context["rankings"]       = list()
            context["total_rankings"] = 0

        context.setdefault("generated_at",    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title",      "AryColBring Inference Dashboard")
        context.setdefault("subtitle",        "Production Inference Monitoring")
        context.setdefault("experiment_name", context.get("experiment_name", "Default Experiment"))
        context.setdefault("model_version",   context.get("model_version", "1.0.0"))
        context.setdefault("batch_size",      context.get("batch_size", 100))
        context.setdefault("num_threads",     context.get("num_threads", 4))

        # Overall score: average of any ranking/quality gauge metrics
        # present. Correctly 0 for a "clean" inference report (no
        # eval-quality metrics), rather than a fabricated number.
        context["overall_score_percent"] = overall_score_percent(metrics)
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
        # BUGFIX: "base.html.j2" doesn't exist in templates/ -- the actual
        # inference template is "infrc_base.html.j2" (the training
        # counterpart in advirender.py correctly uses "train_base.html.j2").
        # This raised jinja2.exceptions.TemplateNotFound on every run.
        template = env.get_template("infrc_base.html.j2")
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
        static_paths = runcopy(advisor = False, dest = output_dir)

        # 4. Build context
        context = build_inference_context(context_data)
        context.update(static_paths or {})

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
            raise FileNotFoundError(f'Not Found {input_path}!')
        with open(input_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
        print(f"Loaded context from: {input_path}")
    else:
        # NOTE: `metrics` only carries real production-performance
        # numbers (latency, QPS) -- NOT ranking-quality metrics like
        # precision/recall/ndcg/auc/mrr, and NOT a fabricated
        # "coverage" placeholder. Those belong on the training
        # dashboard (see advirender.py's own sample data instead).
        rng = np.random.default_rng(0)
        n_items = 60
        item_ids = list(range(100, 100 + n_items))
        item_embeddings = rng.normal(size=(n_items, 16))

        context_data = {
            "metrics": {
                "avg_latency_ms": 12.5,
                "qps": 850,
                "throughput_preds_per_sec": 640.0,
            },
            "inference_statistics": {
                "n_predictions": 50000,
                "n_users_served": 5000,
                "avg_latency_ms": 12.5,
                "throughput_preds_per_sec": 640.0,
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
            "item_embeddings": item_embeddings,
            "item_ids": item_ids,
            "embeddings": {"vectors": item_embeddings, "ids": item_ids,
                          "highlight_id": item_ids[0]},
        }
        print("No input file specified, using sample data.")

    output = generate_inference_report(
        context_data=context_data,
        output_dir=args.output,
        report_name=args.name,
    )
    print(f"\nInference report generated: {output}")
