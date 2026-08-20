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
t05_dashboard_refactor.py
___________________________________________________________________
Regression tests for the arycolbring narative/ dashboard refactor:

  * src/assets/dashboard_utils.py  -- shared scorecard/gauge helpers
    (previously duplicated in advirender.py and rearender.py).
  * src/assets/vizdata.py          -- new visualization-data builders
    (score histogram, 2D embedding PCA projection, similarity heatmap,
    top-K similar items) that replace the inference dashboard's old
    eval-metric / fabricated-coverage display.
  * rearender.py build_inference_context() -- confirms the inference
    dashboard no longer fabricates a "coverage" number and no longer
    surfaces ranking-quality metrics as gauges when only real
    production metrics (latency/qps/throughput) are supplied.

NOTE: like the rest of this project, these tests require the full
project dependency set (tqdm, duckdb, jinja2, numpy, pandas, pytest)
to be installed -- they are not runnable in a bare/offline sandbox.
"""

import numpy as np
import pytest

from src.assets import (bealabel, detect_gauge_metric, generate_gauges,
                        generate_scorecards, overall_score_percent,
                        score_distribution, embedding_projection_2d,
                        similarity_heatmap, top_k_similar_items)
from src.models.arycolbring.narative.rearender import build_inference_context


# ================================================================
# dashboard_utils
# ================================================================

def test_bealabel_ndcg_at_k():
    assert bealabel("ndcg_at_10") == "NDCG@10"


def test_bealabel_precision_at_k():
    assert bealabel("precision_at_10") == "Precision@10"


@pytest.mark.parametrize("metric_name", [
    "precision_at_10", "recall_at_5", "ndcg_at_10", "auc", "mrr", "f1",
])
def test_detect_gauge_metric_flags_eval_metrics(metric_name):
    assert detect_gauge_metric(metric_name) is True


@pytest.mark.parametrize("metric_name", [
    "avg_latency_ms", "qps", "n_predictions",
    "throughput_preds_per_sec", "elapsed_time_sec", "n_users_served",
])
def test_detect_gauge_metric_ignores_inference_metrics(metric_name):
    assert detect_gauge_metric(metric_name) is False


def test_generate_gauges_skips_inference_only_metrics():
    """The core bugfix: an inference report's metrics dict (latency/qps)
    must not produce any quality gauges."""
    metrics = {"avg_latency_ms": 12.5, "qps": 850, "throughput_preds_per_sec": 640.0}
    assert generate_gauges(metrics) == []


def test_generate_scorecards_carries_both_icon_keys():
    """Needed so both advirender.py (icon_emoji) and rearender.py
    (icon_class) templates keep working against the shared function."""
    cards = generate_scorecards({"qps": 100})
    assert "icon_class" in cards[0]
    assert "icon_emoji" in cards[0]
    assert cards[0]["icon_class"] in cards[0]["icon_emoji"]


def test_overall_score_percent_zero_for_clean_inference_metrics():
    assert overall_score_percent({"qps": 100, "avg_latency_ms": 5}) == 0.0


def test_overall_score_percent_averages_real_gauges():
    assert overall_score_percent({"precision_at_10": 0.4, "recall_at_10": 0.6}) == 50.0


# ================================================================
# vizdata
# ================================================================

def test_score_distribution_basic():
    d = score_distribution([0.1, 0.2, 0.9, 0.95, 0.5], bins=5)
    assert d["n"] == 5
    assert sum(d["counts"]) == 5


def test_score_distribution_filters_nan():
    d = score_distribution([0.1, float("nan"), 0.3])
    assert d["n"] == 2


def test_score_distribution_empty_is_safe():
    d = score_distribution([])
    assert d["n"] == 0
    assert d["counts"] == []


def test_embedding_projection_2d_shape_and_highlight():
    emb = np.random.default_rng(0).normal(size=(50, 8))
    ids = list(range(50))
    p = embedding_projection_2d(emb, ids=ids, highlight_id=5)
    assert len(p["x"]) == len(p["y"]) == 50
    assert p["ids"][p["highlight_index"]] == 5


def test_embedding_projection_2d_subsamples_large_inputs():
    emb = np.random.default_rng(1).normal(size=(1000, 4))
    p = embedding_projection_2d(emb, max_points=100)
    assert len(p["x"]) == 100


def test_similarity_heatmap_diagonal_is_one():
    h = similarity_heatmap(np.eye(4), ids=["a", "b", "c", "d"], top_n=4)
    for i in range(4):
        assert h["z"][i][i] == pytest.approx(1.0, abs=1e-4)


def test_top_k_similar_items_excludes_self():
    emb = np.array([[1, 0], [1, 0.01], [0, 1], [0, 1.01]])
    sim = top_k_similar_items(emb, ["A", "B", "C", "D"], k=1)
    assert sim["A"] == ["B"]
    assert sim["C"] == ["D"]


# ================================================================
# rearender.build_inference_context (the actual bugfix, end-to-end)
# ================================================================

@pytest.fixture
def clean_inference_context_data():
    rng = np.random.default_rng(0)
    n_items = 20
    item_ids = list(range(100, 100 + n_items))
    item_embeddings = rng.normal(size=(n_items, 8))
    return {
        "metrics": {"avg_latency_ms": 12.5, "qps": 850,
                    "throughput_preds_per_sec": 640.0},
        "inference_statistics": {"n_predictions": 50000, "n_users_served": 5000,
                                 "avg_latency_ms": 12.5,
                                 "throughput_preds_per_sec": 640.0},
        "experiment_name": "Test Run",
        "predictions": [
            {"user_id": 1, "item_id": 100, "score": 0.95, "rank": 1},
            {"user_id": 1, "item_id": 105, "score": 0.87, "rank": 2},
            {"user_id": 2, "item_id": 150 % (100 + n_items), "score": 0.42, "rank": 1},
        ],
        "item_embeddings": item_embeddings,
        "item_ids": item_ids,
        "embeddings": {"vectors": item_embeddings, "ids": item_ids,
                      "highlight_id": item_ids[0]},
    }


def test_inference_context_has_no_fabricated_coverage(clean_inference_context_data):
    ctx = build_inference_context(clean_inference_context_data)
    assert "coverage" not in ctx.get("inference_statistics", {})


def test_inference_context_no_gauges_for_clean_metrics(clean_inference_context_data):
    ctx = build_inference_context(clean_inference_context_data)
    assert ctx["gauges"] == []
    assert ctx["overall_score_percent"] == 0.0


def test_inference_context_builds_new_visualizations(clean_inference_context_data):
    ctx = build_inference_context(clean_inference_context_data)
    assert ctx["score_distribution"]["n"] == 3
    assert ctx["embedding_plot"]["x"]
    assert ctx["similarity_heatmap"]["z"]


def test_inference_context_rankings_have_similar_items(clean_inference_context_data):
    ctx = build_inference_context(clean_inference_context_data)
    assert ctx["rankings"][0].get("similar_items")


def test_inference_context_gauges_still_appear_if_caller_passes_eval_metrics(
        clean_inference_context_data):
    """Not a recommendation (eval metrics belong on the training
    dashboard) -- just confirms generate_gauges() itself isn't broken
    for a caller who does pass one in."""
    clean_inference_context_data["metrics"]["precision_at_10"] = 0.3
    ctx = build_inference_context(clean_inference_context_data)
    assert len(ctx["gauges"]) == 1
    assert ctx["gauges"][0]["label"] == "Precision@10"
