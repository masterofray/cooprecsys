#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
t05_narative.py
___________________________________________________________________
Pytest suite for src/models/ary2tower/narative/ -- the native
dashboard renderers (a2trearender.py, a2tadvirender.py) and the
class-based wrappers (a2tinfer/a2tinfercore.py, a2ttrain/a2ttraincore.py).
"""

import numpy as np
import pytest

from src.models.ary2tower.narative import (build_inference_context, generate_inference_report,
                                            build_training_context, generate_training_report,
                                            StaticInferenceDashboard, StaticTrainingDashboard)


@pytest.fixture
def sample_inference_context():
    rng = np.random.default_rng(0)
    item_embeddings = rng.normal(size=(20, 6))
    item_ids = list(range(20))
    predictions = [{"user_id": 0, "item_id": i, "score": float(rng.normal()), "rank": r + 1}
                  for r, i in enumerate(range(8))]
    return {
        "metrics": {"avg_latency_ms": 5.2, "qps": 900, "throughput_preds_per_sec": 700.0},
        "inference_statistics": {"n_predictions": 8, "n_users_served": 1,
                                 "avg_latency_ms": 5.2, "throughput_preds_per_sec": 700.0},
        "experiment_name": "Pytest Inference Report",
        "predictions": predictions,
        "item_embeddings": item_embeddings,
        "item_ids": item_ids,
        "embeddings": {"vectors": item_embeddings, "ids": item_ids, "highlight_id": 0},
    }


@pytest.fixture
def sample_training_context():
    return {
        "metrics": {"precision_at_10": 0.3, "recall_at_10": 0.2, "auc": 0.75,
                   "training_time_sec": 120.0},
        "experiment_name": "Pytest Training Report",
        "loss_history": [0.69, 0.65, 0.60, float("nan"), 0.55],
    }


class TestInferenceReport:

    def test_build_inference_context_no_fabricated_metrics(self, sample_inference_context):
        context = build_inference_context(sample_inference_context)
        assert context["gauges"] == []  # clean inference metrics -> no ranking-quality gauges
        assert "coverage" not in context.get("inference_statistics", {})

    def test_build_inference_context_has_visualizations(self, sample_inference_context):
        context = build_inference_context(sample_inference_context)
        assert context["score_distribution"]["n"] == 8
        assert context["embedding_plot"]["x"]
        assert context["similarity_heatmap"]["z"]

    def test_generate_inference_report_renders(self, sample_inference_context, tmp_path):
        path = generate_inference_report(sample_inference_context, output_dir=tmp_path)
        html = path.read_text()
        assert "ary2tower" in html
        assert "Pytest Inference Report" in html
        assert "coverage" not in html.lower()

    def test_generate_inference_report_copies_static_assets(self, sample_inference_context, tmp_path):
        path = generate_inference_report(sample_inference_context, output_dir=tmp_path)
        assert (path.parent / "static" / "css" / "supportinfer.css").exists()
        assert (path.parent / "static" / "js" / "supportinfer.js").exists()


class TestTrainingReport:

    def test_build_training_context_has_gauges(self, sample_training_context):
        context = build_training_context(sample_training_context)
        # Ranking-quality metrics ARE expected here (unlike the inference report).
        labels = [g["label"] for g in context["gauges"]]
        assert "Precision@10" in labels

    def test_generate_training_report_renders(self, sample_training_context, tmp_path):
        path = generate_training_report(sample_training_context, output_dir=tmp_path)
        html = path.read_text()
        assert "ary2tower" in html
        assert "Precision@10" in html
        assert "js-loss-render" in html


class TestDashboardWrappers:

    def test_static_inference_dashboard(self, sample_inference_context, tmp_path):
        dashboard = StaticInferenceDashboard(output_dir=tmp_path)
        path = dashboard.generate(sample_inference_context)
        assert path.exists()

    def test_static_training_dashboard(self, sample_training_context, tmp_path):
        dashboard = StaticTrainingDashboard(output_dir=tmp_path)
        path = dashboard.generate(sample_training_context)
        assert path.exists()

    def test_static_dashboard_output_dir_override(self, sample_inference_context, tmp_path):
        dashboard = StaticInferenceDashboard(output_dir=tmp_path / "default")
        override_dir = tmp_path / "override"
        path = dashboard.generate(sample_inference_context, output_dir=override_dir)
        assert str(override_dir) in str(path)
