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
vizdata.py
_________________________________________
Pure numpy/pandas helpers that turn raw inference artifacts (prediction
scores, user/item embeddings) into small, JSON-serializable payloads
ready to hand to a Plotly widget in a Jinja2 template.

Deliberately dependency-light (numpy + pandas only, no scikit-learn) --
report renderers are on the critical path of every inference run, so
this keeps them fast and avoids growing the model-serving dependency
footprint just to draw a chart. PCA is implemented directly via
`numpy.linalg.svd` rather than importing `sklearn.decomposition.PCA`.

Every function returns plain Python (lists/floats/ints) instead of numpy
types, so `json.dumps` / Jinja's `tojson` filter can consume the result
directly.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..configs import logger

ArrayLike = Union[Sequence[float], np.ndarray]


def score_distribution(scores: ArrayLike, bins: int = 20) -> Dict[str, Any]:
    """Histogram of prediction scores for the current report, plus
    summary statistics (mean/median) so the template can overlay them.

    Returns
    -------
    dict with keys: `counts`, `bin_edges`, `mean`, `median`, `n`.
    Empty-input-safe: returns zeroed-out payload rather than raising.
    """
    logger.debug("Building score distribution (bins=%d).", bins)
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[~np.isnan(arr)] if arr.size else arr
    if arr.size == 0:
        logger.warning("score_distribution() called with no valid scores.")
        return {"counts": [], "bin_edges": [], "mean": 0.0,
                "median": 0.0, "n": 0}

    counts, bin_edges = np.histogram(arr, bins=bins)
    payload = {"counts"    : counts.astype(int).tolist(),
              "bin_edges" : [round(float(edge), 6) for edge in bin_edges],
              "mean"      : round(float(np.mean(arr)), 6),
              "median"    : round(float(np.median(arr)), 6),
              "n"         : int(arr.size)}
    logger.info("Score distribution built: n=%d, mean=%.4f.",
                 payload["n"], payload["mean"])
    return payload


def embedding_projection_2d(embeddings: np.ndarray,
                             ids: Optional[Sequence[Any]] = None,
                             highlight_id: Optional[Any] = None,
                             max_points: int = 500,
                             random_state: int = 42,
                            ) -> Dict[str, Any]:
    """Project `embeddings` (n_entities x dim) down to 2D via PCA
    (top-2 singular components) for scatter-plot visualization.

    If there are more than `max_points` rows, a random (seeded)
    subsample is taken first so the resulting payload/plot stays
    small and fast to render.
    """
    logger.debug("Building 2D embedding projection (max_points=%d).", max_points)
    if embeddings is None or len(embeddings) == 0:
        logger.warning("embedding_projection_2d() called with no embeddings.")
        return {"x": [], "y": [], "ids": [], "highlight_index": None}

    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = embeddings.shape[0]
    ids = list(ids) if ids is not None else list(range(n))

    keep_idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(random_state)
        keep_idx = rng.choice(n, size=max_points, replace=False)
        # Always keep the highlighted point in the sample, if present.
        if highlight_id is not None and highlight_id in ids:
            h = ids.index(highlight_id)
            if h not in keep_idx:
                keep_idx[0] = h
        keep_idx.sort()

    sub = embeddings[keep_idx]
    sub_ids = [ids[i] for i in keep_idx]

    centered = sub - sub.mean(axis=0, keepdims=True)
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        # Not enough rank to project; fall back to zero-padded axes.
        coords = np.zeros((sub.shape[0], 2))
        if sub.shape[1] >= 1:
            coords[:, 0] = centered[:, 0]
    else:
        # PCA via SVD -- top 2 principal components.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T

    highlight_index = None
    if highlight_id is not None and highlight_id in sub_ids:
        highlight_index = sub_ids.index(highlight_id)

    payload = {"x"               : [round(float(v), 6) for v in coords[:, 0]],
              "y"                : [round(float(v), 6) for v in coords[:, 1]],
              "ids"              : sub_ids,
              "highlight_index"  : highlight_index}
    logger.info("Embedding projection built: %d point(s) (of %d total).",
                 len(sub_ids), n)
    return payload


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of `matrix` against itself."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def similarity_heatmap(embeddings: np.ndarray,
                        ids: Optional[Sequence[Any]] = None,
                        top_n: int = 20,
                       ) -> Dict[str, Any]:
    """Cosine-similarity heatmap payload for the first `top_n` entities
    in `embeddings` (e.g. the top-N recommended items for a user, or the
    top-N most active users)."""
    logger.debug("Building similarity heatmap (top_n=%d).", top_n)
    if embeddings is None or len(embeddings) == 0:
        logger.warning("similarity_heatmap() called with no embeddings.")
        return {"z": [], "x": [], "y": []}

    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = min(top_n, embeddings.shape[0])
    sub = embeddings[:n]
    labels = [str(v) for v in (list(ids)[:n] if ids is not None else range(n))]

    sim = _cosine_similarity_matrix(sub)
    payload = {"z": [[round(float(v), 4) for v in row] for row in sim],
              "x" : labels,
              "y" : labels}
    logger.info("Similarity heatmap built: %dx%d.", n, n)
    return payload


def top_k_similar_items(item_embeddings: np.ndarray,
                         item_ids: Sequence[Any],
                         k: int = 3,
                        ) -> Dict[Any, List[Any]]:
    """For every item in `item_ids`, find its `k` nearest neighbors
    (by cosine similarity) among the other items. Used to populate the
    "Similar Items" column of the recommendations table.

    Returns a dict mapping item_id -> list of up to `k` similar item_ids
    (nearest first, self excluded).
    """
    logger.debug("Computing top-%d similar items for %d item(s).",
                  k, len(item_ids) if item_ids is not None else 0)
    if item_embeddings is None or len(item_embeddings) == 0 or not len(item_ids):
        return dict()

    item_embeddings = np.asarray(item_embeddings, dtype=np.float64)
    sim = _cosine_similarity_matrix(item_embeddings)
    np.fill_diagonal(sim, -np.inf)

    result: Dict[Any, List[Any]] = dict()
    n = len(item_ids)
    k = max(0, min(k, n - 1))
    for i in range(n):
        if k == 0:
            result[item_ids[i]] = list()
            continue
        neighbor_idx = np.argpartition(-sim[i], k - 1)[:k]
        neighbor_idx = neighbor_idx[np.argsort(-sim[i][neighbor_idx])]
        result[item_ids[i]] = [item_ids[j] for j in neighbor_idx]
    logger.info("Top-%d similar items computed for %d item(s).", k, n)
    return result


def scores_to_predictions_frame(user_ids: ArrayLike,
                                 item_ids: ArrayLike,
                                 scores: ArrayLike,
                                ) -> pd.DataFrame:
    """Assemble a tidy (user_id, item_id, score, rank) DataFrame from
    three parallel arrays -- convenience for callers building the
    `predictions` context list from raw `predict()` output."""
    df = pd.DataFrame({"user_id": list(user_ids),
                       "item_id" : list(item_ids),
                       "score"   : [float(s) for s in scores]})
    df["rank"] = df.groupby("user_id")["score"] \
                   .rank(method="first", ascending=False).astype(int)
    return df.sort_values(["user_id", "rank"]).reset_index(drop=True)


if __name__ == "__main__":
    logger.info("vizdata self-test placeholder.")
