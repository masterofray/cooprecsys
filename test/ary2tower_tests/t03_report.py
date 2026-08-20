#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-31"

"""
t03_report.py
___________________________________________________________________
Pytest suite for report.py -- confirms ary2tower's inference dashboard
correctly reuses arycolbring's renderer (no duplicated templates) and
that TwoTowerInference.generate_inference_report() works end to end.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.ary2tower import TwoTowerConfig, TwoTowerTrainer, TwoTowerInference
from src.models.ary2tower.report import generate_two_tower_report


class TestGenerateTwoTowerReport:

    @pytest.fixture
    def sample_predictions_and_metrics(self):
        rng = np.random.default_rng(0)
        item_embeddings = rng.normal(size=(20, 6))
        item_ids = list(range(20))
        predictions = [{"user_id": 0, "item_id": i, "score": float(rng.normal()), "rank": r + 1}
                      for r, i in enumerate(range(8))]
        metrics = {"avg_latency_ms": 5.2, "qps": 900, "throughput_preds_per_sec": 700.0}
        return predictions, metrics, item_embeddings, item_ids

    def test_report_title_overridden(self, sample_predictions_and_metrics, tmp_path):
        predictions, metrics, item_embeddings, item_ids = sample_predictions_and_metrics
        path = generate_two_tower_report(predictions, metrics, item_embeddings, item_ids,
                                         experiment_name="Pytest Report Test",
                                         output_dir=tmp_path)
        html = path.read_text()
        assert "ary2tower" in html
        assert "Pytest Report Test" in html
        assert "AryColBring Inference Dashboard" not in html

    def test_report_has_no_fabricated_coverage(self, sample_predictions_and_metrics, tmp_path):
        predictions, metrics, item_embeddings, item_ids = sample_predictions_and_metrics
        path = generate_two_tower_report(predictions, metrics, item_embeddings, item_ids,
                                         output_dir=tmp_path)
        assert "coverage" not in path.read_text().lower()

    def test_report_includes_insights_tab(self, sample_predictions_and_metrics, tmp_path):
        predictions, metrics, item_embeddings, item_ids = sample_predictions_and_metrics
        path = generate_two_tower_report(predictions, metrics, item_embeddings, item_ids,
                                         output_dir=tmp_path)
        html = path.read_text()
        assert "js-scatter-render" in html and "js-heatmap-render" in html

    def test_report_without_embeddings_still_renders(self, sample_predictions_and_metrics, tmp_path):
        predictions, metrics, _, _ = sample_predictions_and_metrics
        path = generate_two_tower_report(predictions, metrics, output_dir=tmp_path)
        assert path.exists()


class TestInferenceGenerateReport:

    @pytest.fixture
    def trained_inference(self, tmp_path):
        rng = np.random.default_rng(0)
        n_users, n_items, true_dim = 25, 15, 4
        true_user = rng.normal(size=(n_users, true_dim))
        true_item = rng.normal(size=(n_items, true_dim))
        true_scores = true_user @ true_item.T
        rows, cols = [], []
        for u in range(n_users):
            liked = np.argsort(true_scores[u])[::-1][:5]
            rows += [u] * len(liked); cols += list(liked)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                        output_dim=4, n_epochs=5, random_state=0))
        trainer.fit(mat)
        path = tmp_path / "model.npz"
        trainer.save_model(path)
        return TwoTowerInference(path), tmp_path

    def test_generate_inference_report_end_to_end(self, trained_inference):
        infer, tmp_path = trained_inference
        report_path = infer.generate_inference_report(
            user_ids=[0, 1, 2], n_items=5,
            experiment_name="Fixture Report Test",
            output_dir=tmp_path / "report_output")
        html = report_path.read_text()
        assert "Fixture Report Test" in html
        assert "ary2tower" in html

    def test_generate_inference_report_without_embeddings(self, trained_inference):
        infer, tmp_path = trained_inference
        report_path = infer.generate_inference_report(
            user_ids=[0], n_items=3, include_embeddings=False,
            output_dir=tmp_path / "report_output_no_emb")
        assert report_path.exists()
