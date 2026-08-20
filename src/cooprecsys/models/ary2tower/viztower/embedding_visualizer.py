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
embedding_visualizer.py
_________________________________________
High-dimensional embedding visualization for ary2tower tower outputs
(or raw id embeddings).

Deliberately does NOT reimplement the PCA/cosine-similarity math --
that already exists, tested, in src/assets/vizdata.py (built for the
Task 1 dashboard refactor). This module only adds a matplotlib
rendering layer on top of that same math, so the numbers on a static
PNG (this module) and the numbers on the interactive dashboard
(vizdata.py -> Plotly, see narative/infrc_insights.html.j2) always
agree -- there is exactly one implementation of the projection/
similarity math in this repo.
"""

from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ....configs import logger
from ....assets import embedding_projection_2d, similarity_heatmap

ACCENT = "#FF6B35"
ACCENT_TEAL = "#4ECDC4"


def plot_embedding_projection(embeddings: np.ndarray,
                               ids: Optional[Sequence[Any]] = None,
                               highlight_id: Optional[Any] = None,
                               max_points: int = 500,
                               title: str = "Embedding Projection (2D PCA)",
                               ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Scatter plot of `embeddings` projected to 2D via PCA (same
    computation as the dashboard's Insights tab -- see
    src.assets.vizdata.embedding_projection_2d)."""
    logger.debug("Plotting embedding projection for %d entities.",
                 0 if embeddings is None else len(embeddings))
    projection = embedding_projection_2d(embeddings, ids=ids,
                                         highlight_id=highlight_id,
                                         max_points=max_points)

    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 6))
    ax = ax or fig.gca()

    if not projection["x"]:
        ax.text(0.5, 0.5, "No embeddings to project", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    x = np.asarray(projection["x"])
    y = np.asarray(projection["y"])
    highlight_index = projection["highlight_index"]

    colors = np.full(len(x), ACCENT_TEAL, dtype=object)
    sizes = np.full(len(x), 24.0)
    if highlight_index is not None:
        colors[highlight_index] = ACCENT
        sizes[highlight_index] = 90.0

    ax.scatter(x, y, c=list(colors), s=sizes, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_similarity_heatmap(embeddings: np.ndarray,
                             ids: Optional[Sequence[Any]] = None,
                             top_n: int = 20,
                             title: str = "Item-Item Similarity",
                             ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Heatmap of cosine similarity among the first `top_n` entities in
    `embeddings` (same computation as the dashboard's Insights tab --
    see src.assets.vizdata.similarity_heatmap)."""
    logger.debug("Plotting similarity heatmap (top_n=%d).", top_n)
    heatmap = similarity_heatmap(embeddings, ids=ids, top_n=top_n)

    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 6))
    ax = ax or fig.gca()

    if not heatmap["z"]:
        ax.text(0.5, 0.5, "No embeddings to compare", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    matrix = np.asarray(heatmap["z"])
    im = ax.imshow(matrix, cmap="Oranges", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(heatmap["x"])))
    ax.set_xticklabels(heatmap["x"], rotation=90, fontsize=6)
    ax.set_yticks(range(len(heatmap["y"])))
    ax.set_yticklabels(heatmap["y"], fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
