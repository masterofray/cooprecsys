#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-30"

"""
inference.py
_________________________________________
TwoTowerInference: predict()/recommend() for a trained two-tower model.

API shape deliberately mirrors AryColBringInference
(src/models/arycolbring/inference.py) -- same method names/semantics
(predict, recommend, get_metrics) -- so callers already familiar with
the arycolbring inference path don't need to learn a second API.
"""

import logging
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from ...configs import logger
except ImportError:  # pragma: no cover - fallback for standalone/test use
    logger = logging.getLogger(__name__)

from .config import TwoTowerConfig
from .towers import TwoTowerWeights, UserTower, ItemTower, dot_product_similarity


class _LRUCache:
    """Minimal LRU cache for tower-output vectors, keyed by entity id.
    Avoids recomputing a user's/item's forward pass on every request
    when the same ids recur across calls (e.g. a popular item, or a
    user browsing multiple pages)."""

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self._store: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: int, value: np.ndarray) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class TwoTowerInference:
    """Serves recommendations from a trained two-tower model.

    Parameters
    ----------
    model_path   : path to a .npz file written by TwoTowerTrainer.save_model().
    num_threads  : OpenMP thread count for the compiled kernels (ignored
                   by the NumPy fallback path).
    cache_enabled: enable the per-id tower-output LRU cache.
    """

    def __init__(self, model_path: Union[str, Path],
                 num_threads: int = 4, cache_enabled: bool = True,
                 cache_capacity: int = 4096):
        # Local import to avoid a hard circular dependency at module
        # load time (trainer.py also imports from towers.py/config.py).
        from .trainer import TwoTowerTrainer

        self._trainer = TwoTowerTrainer.load_model(model_path)
        self.weights = self._trainer.weights
        self.user_tower = UserTower(self.weights)
        self.item_tower = ItemTower(self.weights)
        self.num_threads = num_threads
        self.cache_enabled = cache_enabled
        self._user_cache = _LRUCache(cache_capacity) if cache_enabled else None
        self._item_cache = _LRUCache(cache_capacity) if cache_enabled else None

        self.inference_stats = {"n_predictions": 0, "n_users_served": 0,
                                "total_latency_ms": 0.0,
                                "start_time": datetime.now()}
        logger.info("TwoTowerInference loaded from %s (n_users=%d, n_items=%d)",
                    model_path, self._trainer.n_users, self._trainer.n_items)

    def _user_output(self, user_id: int) -> np.ndarray:
        if self.cache_enabled:
            cached = self._user_cache.get(user_id)
            if cached is not None:
                return cached
        out = self.user_tower.forward(np.array([user_id]))[0]
        if self.cache_enabled:
            self._user_cache.put(user_id, out)
        return out

    def _item_outputs(self, item_ids: np.ndarray) -> np.ndarray:
        if not self.cache_enabled:
            return self.item_tower.forward(item_ids)

        outputs = np.zeros((len(item_ids), self.weights.output_dim), dtype=np.float32)
        missing_mask = np.zeros(len(item_ids), dtype=bool)
        for i, iid in enumerate(item_ids):
            cached = self._item_cache.get(int(iid))
            if cached is None:
                missing_mask[i] = True
            else:
                outputs[i] = cached

        if missing_mask.any():
            fresh = self.item_tower.forward(item_ids[missing_mask])
            outputs[missing_mask] = fresh
            for iid, out in zip(item_ids[missing_mask], fresh):
                self._item_cache.put(int(iid), out)
        return outputs

    def predict(self, user_ids: Union[int, List[int], np.ndarray],
                item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """Dot-product similarity score for each (user_id, item_id) pair
        (paired, not cross-joined -- same convention as
        AryColBringPredictor.predict() with cross_join=False)."""
        start = time.perf_counter()
        user_ids = np.atleast_1d(np.asarray(user_ids, dtype=np.int32))
        item_ids = np.atleast_1d(np.asarray(item_ids, dtype=np.int32))
        if user_ids.shape[0] != item_ids.shape[0]:
            raise ValueError(f"user_ids ({user_ids.shape[0]}) and item_ids "
                             f"({item_ids.shape[0]}) must have equal length")

        user_out = np.stack([self._user_output(int(u)) for u in user_ids])
        item_out = self._item_outputs(item_ids)
        scores = dot_product_similarity(user_out, item_out)

        latency_ms = (time.perf_counter() - start) * 1000
        self.inference_stats["n_predictions"] += len(user_ids)
        self.inference_stats["total_latency_ms"] += latency_ms
        return scores

    def recommend(self, user_id: int, n_items: int = 10,
                  exclude_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Top-N items for `user_id`, sorted by score descending."""
        start = time.perf_counter()
        exclude = set(exclude_items or [])
        n_catalog_items = self.weights.item_embeddings.shape[0]
        candidate_ids = np.array([i for i in range(n_catalog_items) if i not in exclude],
                                 dtype=np.int32)
        if candidate_ids.size == 0:
            return []

        user_out = self._user_output(user_id)
        item_out = self._item_outputs(candidate_ids)
        scores = item_out @ user_out

        top_n_idx = np.argsort(scores)[::-1][:n_items]
        recommendations = [(int(candidate_ids[i]), float(scores[i])) for i in top_n_idx]

        latency_ms = (time.perf_counter() - start) * 1000
        self.inference_stats["n_predictions"] += len(recommendations)
        self.inference_stats["n_users_served"] += 1
        self.inference_stats["total_latency_ms"] += latency_ms
        return recommendations

    def batch_recommend(self, user_ids: List[int], n_items: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """recommend() for multiple users at once."""
        return {uid: self.recommend(uid, n_items=n_items) for uid in user_ids}

    def get_metrics(self) -> Dict[str, float]:
        """Real production-inference metrics only (latency/throughput/
        volume) -- deliberately NOT ranking-quality metrics; see the
        Task-1 dashboard fix (rearender.py) for why those two categories
        must stay separate."""
        n_preds = self.inference_stats["n_predictions"]
        total_latency = self.inference_stats["total_latency_ms"]
        elapsed = (datetime.now() - self.inference_stats["start_time"]).total_seconds()
        return {
            "n_predictions": n_preds,
            "n_users_served": self.inference_stats["n_users_served"],
            "avg_latency_ms": (total_latency / n_preds) if n_preds else 0.0,
            "elapsed_time_sec": elapsed,
            "throughput_preds_per_sec": (n_preds / elapsed) if elapsed > 0 else 0.0,
            "qps": (self.inference_stats["n_users_served"] / elapsed) if elapsed > 0 else 0.0,
            "cache_size_users": len(self._user_cache) if self.cache_enabled else 0,
            "cache_size_items": len(self._item_cache) if self.cache_enabled else 0,
        }
