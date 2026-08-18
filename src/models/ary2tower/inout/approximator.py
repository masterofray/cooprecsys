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
approximator.py
_________________________________________
TwoTowerPredictor: forward pass + scoring for a two-tower model, given
already-trained weights. Mirrors AryColBringPredictor's role
(arycolbring/inout/approximator.py) -- same build_pairs()
pairwise/broadcast/cross-join semantics, same predict()/predict_rank()
shape -- built on top of ..towers.UserTower/ItemTower rather than
duplicating the forward-pass math (that lives in exactly one place:
towers.py, dispatching to the compiled Cython kernel or the NumPy
fallback).
"""

import numpy as np
from typing import Any, List, Optional, Tuple, Union

from ....configs import logger
from .scaffold import TwoTowerBase
from ..config import TwoTowerConfig
from ..towers import (TwoTowerWeights, UserTower, ItemTower,
                      dot_product_similarity, cosine_similarity)

IdsLike = Union[int, List[int], np.ndarray]


class TwoTowerPredictor(TwoTowerBase):
    """Scores (user, item) pairs from a two-tower model's weights.

    Parameters
    ----------
    weights : an existing TwoTowerWeights (e.g. loaded from disk). If
        omitted, a fresh (untrained) TwoTowerWeights is created from
        `config` -- matching TwoTowerArchitect's own construction, so
        the two classes stay interchangeable.
    """

    def __init__(self, n_users: int, n_items: int,
                 config: Optional[TwoTowerConfig] = None,
                 weights: Optional[TwoTowerWeights] = None, **kwargs):
        super().__init__(n_users, n_items, config, **kwargs)
        self.weights = weights if weights is not None else TwoTowerWeights(
            n_users, n_items, embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim, output_dim=self.config.output_dim,
            random_state=self.config.random_state)
        self._is_fitted = weights is not None
        self.user_tower = UserTower(self.weights)
        self.item_tower = ItemTower(self.weights)
        logger.info("TwoTowerPredictor initialized: n_users=%d n_items=%d",
                    n_users, n_items)

    @staticmethod
    def build_pairs(user_ids: IdsLike, item_ids: IdsLike,
                     cross_join: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Turn `user_ids`/`item_ids` into two parallel (user_id, item_id)
        arrays, same semantics as AryColBringPredictor.build_pairs():

        - cross_join=False (default): strict pairwise -- lengths must
          match, or one side may be a length-1 scalar broadcast against
          the other.
        - cross_join=True: every user against every item (row-major:
          user 0 vs every item, then user 1 vs every item, ...).
        """
        user_arr = np.atleast_1d(np.asarray(user_ids, dtype=np.int32))
        item_arr = np.atleast_1d(np.asarray(item_ids, dtype=np.int32))

        if cross_join:
            u = np.repeat(user_arr, len(item_arr))
            i = np.tile(item_arr, len(user_arr))
            return u, i

        if len(user_arr) == len(item_arr):
            return user_arr, item_arr
        if len(user_arr) == 1:
            return np.repeat(user_arr, len(item_arr)), item_arr
        if len(item_arr) == 1:
            return user_arr, np.repeat(item_arr, len(user_arr))
        raise ValueError(f"user_ids (len={len(user_arr)}) and item_ids "
                         f"(len={len(item_arr)}) have incompatible lengths "
                         "for pairwise scoring (neither matches, and neither "
                         "is a length-1 scalar to broadcast). Pass "
                         "cross_join=True for every-user-vs-every-item.")

    def predict(self, user_ids: IdsLike, item_ids: IdsLike,
                cross_join: bool = False, similarity: str = "dot") -> np.ndarray:
        """Score (user, item) pairs. See build_pairs() for how
        `user_ids`/`item_ids` are paired up."""
        u, i = self.build_pairs(user_ids, item_ids, cross_join=cross_join)
        user_out = self.user_tower.forward(u)
        item_out = self.item_tower.forward(i)
        if similarity == "cosine":
            return cosine_similarity(user_out, item_out)
        if similarity == "dot":
            return dot_product_similarity(user_out, item_out)
        raise ValueError(f"Unknown similarity '{similarity}', expected 'dot' or 'cosine'")

    def predict_rank(self, user_id: int, item_ids: Optional[IdsLike] = None,
                      n_items: int = 10, exclude_items: Optional[List[Any]] = None,
                      similarity: str = "dot") -> List[Tuple[int, float]]:
        """Top-N items for `user_id`, sorted by score descending.
        Defaults to scoring the whole catalogue if `item_ids` isn't given."""
        exclude = set(exclude_items or [])
        if item_ids is None:
            item_ids = np.array([i for i in range(self.n_items) if i not in exclude],
                                dtype=np.int32)
        else:
            item_ids = np.array([i for i in np.atleast_1d(item_ids) if i not in exclude],
                                dtype=np.int32)
        if item_ids.size == 0:
            return []

        scores = self.predict(user_id, item_ids, cross_join=False, similarity=similarity) \
            if len(item_ids) == 1 else \
            self.predict([user_id], item_ids, cross_join=True, similarity=similarity)
        top_n_idx = np.argsort(scores)[::-1][:n_items]
        return [(int(item_ids[idx]), float(scores[idx])) for idx in top_n_idx]
