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
performance_plots.py
_________________________________________
Model-performance plots for ary2tower: prediction score distribution
(reuses cooprecsys/assets/vizdata.py's histogram math -- same numbers as the
dashboard's Insights tab) and a precision/recall-at-K curve (new here;
no equivalent existed elsewhere in the repo to reuse).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from ....configs import logger
from ....assets import score_distribution

ACCENT = "#FF6B35"
ACCENT_TEAL = "#4ECDC4"


def plot_score_distribution(scores: Sequence[float],
                             bins: int = 20,
                             title: str = "Prediction Score Distribution",
                             ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Histogram of prediction scores, with a mean/median overlay --
    same computation as the dashboard's Insights tab (see
    cooprecsys.assets.vizdata.score_distribution)."""
    logger.debug("Plotting score distribution (n=%d).",
                 0 if scores is None else len(scores))
    dist = score_distribution(scores, bins=bins)

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 4))
    ax = ax or fig.gca()

    if dist["n"] == 0:
        ax.text(0.5, 0.5, "No prediction scores available", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    edges = np.asarray(dist["bin_edges"])
    counts = np.asarray(dist["counts"])
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    ax.bar(centers, counts, width=widths, color=ACCENT, edgecolor="white", align="center")
    ax.axvline(dist["mean"], color=ACCENT_TEAL, linestyle="--", linewidth=2,
               label=f"mean={dist['mean']:.3f}")
    ax.set_xlabel("Prediction score")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} (n={dist['n']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def precision_recall_at_k(recommended_ids: Sequence[Sequence],
                           relevant_ids: Sequence[set],
                           k_values: Sequence[int] = (1, 3, 5, 10, 20)) -> dict:
    """Precision@K and Recall@K averaged across a set of users.

    Parameters
    ----------
    recommended_ids : one ranked recommendation list (item ids, best
        first) per user -- e.g. `[iid for iid, _ in infer.recommend(u, n_items=max(k_values))]`.
    relevant_ids    : one ground-truth relevant-item set per user
        (same order/length as `recommended_ids`).
    k_values        : the K cutoffs to evaluate.
    """
    if len(recommended_ids) != len(relevant_ids):
        raise ValueError(f"recommended_ids ({len(recommended_ids)}) and relevant_ids "
                         f"({len(relevant_ids)}) must have the same length (one per user)")

    precisions, recalls = list(), list()
    for k in k_values:
        p_at_k, r_at_k = list(), list()
        for recs, relevant in zip(recommended_ids, relevant_ids):
            top_k = set(recs[:k])
            hits = len(top_k & relevant)
            p_at_k.append(hits / k if k > 0 else 0.0)
            r_at_k.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(float(np.mean(p_at_k)) if p_at_k else 0.0)
        recalls.append(float(np.mean(r_at_k)) if r_at_k else 0.0)

    return {"k_values": list(k_values), "precision_at_k": precisions, "recall_at_k": recalls}


def plot_precision_recall_at_k(recommended_ids: Sequence[Sequence],
                                relevant_ids: Sequence[set],
                                k_values: Sequence[int] = (1, 3, 5, 10, 20),
                                title: str = "Precision@K / Recall@K",
                                ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Precision@K and Recall@K curves (see precision_recall_at_k)."""
    result = precision_recall_at_k(recommended_ids, relevant_ids, k_values)
    logger.debug("Plotting precision/recall@K for %d user(s), K in %s.",
                 len(recommended_ids), list(k_values))

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 4))
    ax = ax or fig.gca()

    ax.plot(result["k_values"], result["precision_at_k"], color=ACCENT,
            marker="o", label="Precision@K")
    ax.plot(result["k_values"], result["recall_at_k"], color=ACCENT_TEAL,
            marker="s", label="Recall@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
