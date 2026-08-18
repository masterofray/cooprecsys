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
fallback_reasoner.py
_________________________________________
TwoTowerFallBack: purchase-aware recommendation cleanup + item-to-item
cold-start fallback, for a two-tower model's item tower outputs.
Mirrors AryInfFallBack's role and output contract
(arycolbring/inout/fallback_reasoner.py):

  1. Filter: drop candidates already in the user's purchase history.
  2. Item-to-item fallback: if filtering leaves the list short, backfill
     via cosine similarity over item embeddings, seeded by the user's
     purchase history (or an explicit seed list for a brand-new user
     with no purchase history at all -- the true cold-start case).

Intentionally simpler than AryInfFallBack in one respect: column
auto-detection there goes through `prepare.DetectReco_Identifier`
(a heavier dependency); this module takes explicit `user_col`/
`item_col` instead, keeping ary2tower's own dependency footprint to
just NumPy/pandas.
"""

import numpy as np
import pandas as pd
from ....configs import logger
from typing import Any, Dict, List, Optional, Set, Tuple


class TwoTowerFallBack:
    """Wraps a trained item tower's output embeddings with
    purchase-aware recommendation cleanup.

    Parameters
    ----------
    purchase_data   : DataFrame with (at least) a user-id and an
        item-id column.
    item_embeddings : (n_items, output_dim) tower outputs (e.g.
        `TwoTowerInference.weights.item_embeddings` after running every
        catalog item through the item tower once).
    user_col, item_col : column names in `purchase_data`.
    """

    SOURCE_MODEL = "model_topn"
    SOURCE_ITEM2ITEM = "item2item_fallback"

    def __init__(self, purchase_data: pd.DataFrame,
                 item_embeddings: np.ndarray,
                 user_col: str = "user_id", item_col: str = "item_id"):
        if user_col not in purchase_data.columns:
            raise ValueError(f"user_col '{user_col}' not found in purchase_data "
                             f"columns: {list(purchase_data.columns)}")
        if item_col not in purchase_data.columns:
            raise ValueError(f"item_col '{item_col}' not found in purchase_data "
                             f"columns: {list(purchase_data.columns)}")

        self.user_col = user_col
        self.item_col = item_col
        self.item_embeddings = np.asarray(item_embeddings, dtype=np.float64)
        self._purchase_map = self._build_purchase_map(purchase_data)
        logger.info("TwoTowerFallBack initialized: %d user(s) with purchase history, "
                   "%d item embedding(s).", len(self._purchase_map),
                   self.item_embeddings.shape[0])

    def _build_purchase_map(self, purchase_data: pd.DataFrame) -> Dict[Any, Set[Any]]:
        return purchase_data.groupby(self.user_col)[self.item_col].apply(set).to_dict()

    def purchased_items(self, user_id: Any) -> Set[Any]:
        """Every item `user_id` has already purchased (empty set for an
        unseen/new user)."""
        return self._purchase_map.get(user_id, set())

    def item_to_item_candidates(self, seed_items: List[int],
                                 exclude: Optional[Set[int]] = None,
                                 n: int = 5) -> List[Tuple[int, float]]:
        """Cosine-similarity neighbors of the (mean of the) `seed_items`
        embeddings -- the true cold-start path: works from any seed
        item list, purchase history or otherwise (e.g. items viewed in
        the current session), with no dependency on the model having
        seen the target user before.
        """
        exclude = set(exclude or set())
        n_catalog = self.item_embeddings.shape[0]
        valid_seeds = [i for i in seed_items if 0 <= i < n_catalog]
        if not valid_seeds:
            logger.warning("item_to_item_candidates(): no valid seed items in range.")
            return list()

        seed_vector = self.item_embeddings[valid_seeds].mean(axis=0)
        norms = np.linalg.norm(self.item_embeddings, axis=1)
        seed_norm = np.linalg.norm(seed_vector)
        similarities = (self.item_embeddings @ seed_vector) / (norms * seed_norm + 1e-8)

        order = np.argsort(similarities)[::-1]
        excluded_ids = exclude | set(valid_seeds)
        results: List[Tuple[int, float]] = list()
        for idx in order:
            idx = int(idx)
            if idx in excluded_ids:
                continue
            results.append((idx, float(similarities[idx])))
            if len(results) >= n:
                break
        return results

    def clean_recommendations(self, user_id: Any,
                               candidate_pool: List[Tuple[int, float]],
                               n_items: int = 10) -> pd.DataFrame:
        """Filter `candidate_pool` (a model's raw ranked (item_id, score)
        list) against `user_id`'s purchase history, then backfill any
        gap left by filtering via item-to-item fallback so the returned
        list is always `n_items` long (when the catalogue is large
        enough to support it).

        Returns a DataFrame with columns:
        user_id, rank, item_id, score, source, is_fallback
        """
        bought = self.purchased_items(user_id)
        filtered = [(item_id, score) for item_id, score in candidate_pool
                   if item_id not in bought]

        rows: List[Dict[str, Any]] = list()
        for item_id, score in filtered[:n_items]:
            rows.append({"user_id": user_id, "item_id": item_id, "score": score,
                        "source": self.SOURCE_MODEL, "is_fallback": False})

        if len(rows) < n_items:
            already_seen = bought | {r["item_id"] for r in rows} | {i for i, _ in candidate_pool}
            seeds = list(bought) if bought else [item_id for item_id, _ in candidate_pool[:3]]
            needed = n_items - len(rows)
            fallback_candidates = self.item_to_item_candidates(
                seeds, exclude=already_seen, n=needed)
            for item_id, score in fallback_candidates:
                rows.append({"user_id": user_id, "item_id": item_id, "score": score,
                            "source": self.SOURCE_ITEM2ITEM, "is_fallback": True})

        logger.info("clean_recommendations(user_id=%s): %d model, %d fallback slot(s).",
                    user_id, sum(1 for r in rows if not r["is_fallback"]),
                    sum(1 for r in rows if r["is_fallback"]))

        result = pd.DataFrame(rows, columns=["user_id", "item_id", "score",
                                             "source", "is_fallback"])
        if not result.empty:
            result.insert(1, "rank", range(1, len(result) + 1))
        return result
