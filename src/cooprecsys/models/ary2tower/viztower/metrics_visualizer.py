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
metrics_visualizer.py
_________________________________________
Training/inference metrics plots for ary2tower: loss curves and metric
comparison bars. Matplotlib only (CPU, no GPU/torch dependency) --
consistent with the rest of this repo's plotting (see
arycolbring's notebooks, which use the same stack).

Every function returns a matplotlib Figure and does not call
plt.show()/plt.close() -- callers decide whether to display, save
(fig.savefig(...)), or embed it (see dashboard_components.py).
"""

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; caller can still display
                       # the returned Figure in a notebook via its own
                       # backend/inline renderer.
import matplotlib.pyplot as plt
import numpy as np
from ....configs import logger

ACCENT = "#FF6B35"     # matches the light+orange dashboard theme (Task 1)
ACCENT_TEAL = "#4ECDC4"


def plot_loss_curve(loss_history: Sequence[float],
                     title: str = "Training Loss",
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Line plot of per-epoch loss (e.g. TwoTowerTrainer.loss_history).

    NaN entries (the Cython backend's fit() doesn't return a real
    per-epoch loss -- see trainer.py) are skipped rather than plotted
    as gaps, with a note added to the title.
    """
    logger.debug("Plotting loss curve (%d epochs).", len(loss_history))
    values = np.asarray(loss_history, dtype=np.float64)
    epochs = np.arange(1, len(values) + 1)
    valid = ~np.isnan(values)

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 4))
    ax = ax or fig.gca()

    if valid.any():
        ax.plot(epochs[valid], values[valid], color=ACCENT, linewidth=2, marker="o", markersize=4)
    else:
        ax.text(0.5, 0.5, "No per-epoch loss available\n(Cython backend doesn't return one)",
                ha="center", va="center", transform=ax.transAxes, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_bars(metrics: Dict[str, float],
                      title: str = "Metrics",
                      ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Bar chart of a flat metrics dict (e.g. TwoTowerInference.get_metrics(),
    filtered to numeric values first)."""
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    logger.debug("Plotting %d metric bar(s).", len(numeric_metrics))

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 4))
    ax = ax or fig.gca()

    if not numeric_metrics:
        ax.text(0.5, 0.5, "No numeric metrics to display", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    labels = list(numeric_metrics.keys())
    values = list(numeric_metrics.values())
    colors = [ACCENT if i % 2 == 0 else ACCENT_TEAL for i in range(len(labels))]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
