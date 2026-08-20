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
t02_viztower.py
___________________________________________________________________
Pytest suite for src/models/ary2tower/viztower/ and report.py.
CPU-only (matplotlib) -- no GPU/torch dependency, no compiled
extension needed (this whole module is pure Python/NumPy/matplotlib).
"""

import base64

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.models.ary2tower.viztower import (plot_loss_curve, plot_metric_bars,
                                            plot_embedding_projection, plot_similarity_heatmap,
                                            plot_score_distribution, precision_recall_at_k,
                                            plot_precision_recall_at_k, fig_to_base64_png,
                                            fig_to_data_uri, figs_to_html_gallery)


class TestMetricsVisualizer:

    def test_loss_curve_skips_nan(self):
        fig = plot_loss_curve([0.6, 0.5, float("nan"), 0.3])
        line = fig.axes[0].get_lines()[0]
        assert len(line.get_ydata()) == 3

    def test_loss_curve_empty_shows_placeholder(self):
        fig = plot_loss_curve([])
        assert isinstance(fig, plt.Figure)

    def test_metric_bars_excludes_non_numeric(self):
        fig = plot_metric_bars({"qps": 780, "avg_latency_ms": 9.4, "label": "not-a-number"})
        assert len(fig.axes[0].patches) == 2

    def test_metric_bars_empty_dict(self):
        fig = plot_metric_bars({})
        assert isinstance(fig, plt.Figure)


class TestEmbeddingVisualizer:

    @pytest.fixture
    def embeddings(self):
        return np.random.default_rng(0).normal(size=(40, 8))

    def test_projection_point_count(self, embeddings):
        fig = plot_embedding_projection(embeddings, ids=list(range(40)), highlight_id=5)
        scatter = fig.axes[0].collections[0]
        assert scatter.get_offsets().shape[0] == 40

    def test_projection_empty_embeddings(self):
        fig = plot_embedding_projection(np.array([]))
        assert isinstance(fig, plt.Figure)

    def test_heatmap_shape(self, embeddings):
        fig = plot_similarity_heatmap(embeddings, ids=list(range(40)), top_n=10)
        im = fig.axes[0].images[0]
        assert im.get_array().shape == (10, 10)

    def test_heatmap_empty_embeddings(self):
        fig = plot_similarity_heatmap(np.array([]))
        assert isinstance(fig, plt.Figure)


class TestPerformancePlots:

    def test_score_distribution_bin_count(self):
        scores = np.random.default_rng(0).normal(loc=0.5, scale=0.2, size=200)
        fig = plot_score_distribution(scores, bins=15)
        assert len(fig.axes[0].patches) <= 15

    def test_score_distribution_empty(self):
        fig = plot_score_distribution([])
        assert isinstance(fig, plt.Figure)

    def test_precision_recall_at_k_values(self):
        recommended = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
        relevant = [{2, 4, 99}, {11, 100}]
        result = precision_recall_at_k(recommended, relevant, k_values=[1, 3, 5])
        assert result["precision_at_k"][2] == pytest.approx(0.3)

    def test_precision_recall_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            precision_recall_at_k([[1, 2]], [{1}, {2}])

    def test_plot_precision_recall_returns_figure(self):
        fig = plot_precision_recall_at_k([[1, 2, 3]], [{2}], k_values=[1, 3])
        assert isinstance(fig, plt.Figure)


class TestDashboardComponents:

    def test_fig_to_base64_png_is_valid_png(self):
        fig = plot_metric_bars({"qps": 100})
        encoded = fig_to_base64_png(fig)
        decoded = base64.b64decode(encoded)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_fig_to_data_uri_prefix(self):
        fig = plot_metric_bars({"qps": 100})
        uri = fig_to_data_uri(fig)
        assert uri.startswith("data:image/png;base64,")

    def test_html_gallery_embeds_all_figures(self):
        fig1 = plot_metric_bars({"qps": 100})
        fig2 = plot_loss_curve([0.5, 0.4])
        html = figs_to_html_gallery({"Metrics": fig1, "Loss": fig2}, title="Test Gallery")
        assert html.count("<img") == 2
        assert "Test Gallery" in html
