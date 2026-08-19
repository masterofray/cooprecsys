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
report.py
_________________________________________
Interactive HTML dashboard for a trained ary2tower model.

Delegates to ary2tower's own native renderer (narative/a2trearender.py)
-- a light + orange, single-page dashboard (Overview / Rankings /
Insights), built specifically for this module rather than reusing
arycolbring's tabbed multi-template system. See
narative/README-equivalent docstrings in a2trearender.py and
rensupport.py for the rendering details.

(An earlier revision of this module reused
arycolbring.narative.rearender.generate_inference_report() directly,
to avoid duplicating a whole dashboard tree. That reuse is now
superseded by ary2tower's own narative/ package, added on explicit
request for a lighter-weight, single-page-per-mode dashboard design --
see CHANGELOG.md.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .narative.a2trearender import generate_inference_report as _generate_report


def generate_two_tower_report(
        predictions        : List[Dict[str, Any]],
        metrics             : Dict[str, float],
        item_embeddings     : Optional[np.ndarray]   = None,
        item_ids            : Optional[List[Any]]     = None,
        experiment_name      : str                     = "ary2tower Inference Run",
        output_dir           : Optional[Union[str, Path]] = None,
    ) -> Path:
    """Generate an interactive HTML inference dashboard for ary2tower.

    Parameters
    ----------
    predictions     : list of {"user_id", "item_id", "score", "rank"}
        dicts -- same shape TwoTowerInference.recommend() plus a
        user_id/rank annotation produces (see
        TwoTowerInference.generate_inference_report(), which builds
        this for you from a live model).
    metrics         : real production-inference metrics only
        (avg_latency_ms, qps, throughput_preds_per_sec, ...) --
        NOT ranking-quality metrics (precision/recall/ndcg/auc/mrr).
        See narative/a2trearender.py for why those two categories are
        kept separate on this report.
    item_embeddings : (n_items, output_dim) tower outputs, for the
        Insights section's PCA scatter + similarity heatmap. Optional
        -- the report renders fine without it, just without those two
        widgets.
    item_ids        : catalog ids for `item_embeddings`'s rows.

    Returns
    -------
    Path to the rendered HTML file.
    """
    context_data: Dict[str, Any] = {
        "experiment_name"     : experiment_name,
        "metrics"             : metrics,
        "inference_statistics": {
            "n_predictions"           : metrics.get("n_predictions", len(predictions)),
            "n_users_served"          : metrics.get("n_users_served", 0),
            "avg_latency_ms"          : metrics.get("avg_latency_ms", 0.0),
            "throughput_preds_per_sec": metrics.get("throughput_preds_per_sec", 0.0),
        },
        "predictions"         : predictions,
    }

    if item_embeddings is not None and item_ids is not None:
        context_data["item_embeddings"] = item_embeddings
        context_data["item_ids"] = item_ids
        context_data["embeddings"] = {"vectors": item_embeddings, "ids": item_ids}

    return _generate_report(context_data, output_dir=output_dir)
