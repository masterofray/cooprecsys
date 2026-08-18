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

The core scoring (build_pairs + predict) now delegates to
inout/approximator.py's TwoTowerPredictor -- this class adds the
serving-layer concerns on top of it: an LRU cache for tower outputs,
latency/throughput stats, top-N recommend()/batch_recommend(), and
report generation. Same relationship arycolbring/inference.py's
AryColBringInference has to TheReasoner (inout/approximator.py).

API shape deliberately mirrors AryColBringInference
(src/models/arycolbring/inference.py) -- same method names/semantics
(predict, recommend, get_metrics) -- so callers already familiar with
the arycolbring inference path don't need to learn a second API.

FALLBACK FOR SHORT RESULTS (exclude_purchased): when a caller asks for
`n_items` recommendations with already-purchased items excluded, the
raw top-N candidate pool can come up short for some users (a heavy
repeat buyer may have already purchased most of what the model would
otherwise rank highly for them). `recommend()`/`batch_recommend()`
optionally backfill any such shortfall via
inout/fallback_reasoner.py's TwoTowerFallBack -- pure item-to-item
cosine similarity over the item tower's own output embeddings, seeded
by the user's purchase history. Deliberately NOT similar-user/
"favorite user" modeling and NOT query/keyword-based retrieval (both
would need infrastructure this module doesn't have and would answer a
different question), and never re-suggests an item the user already
bought -- see TwoTowerFallBack's own docstring.
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
from .inout.approximator import TwoTowerPredictor
from .inout.fallback_reasoner import TwoTowerFallBack


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
                 cache_capacity: int = 4096,
                 purchase_data=None, user_col: str = "user_id",
                 item_col: str = "item_id"):
        # Local import to avoid a hard circular dependency at module
        # load time (trainer.py also imports from towers.py/config.py).
        from .trainer import TwoTowerTrainer

        self._trainer = TwoTowerTrainer.load_model(model_path)
        self.weights = self._trainer.weights
        self.predictor = TwoTowerPredictor(self._trainer.n_users, self._trainer.n_items,
                                           config=self._trainer.config, weights=self.weights)
        self.user_tower = self.predictor.user_tower
        self.item_tower = self.predictor.item_tower
        self.num_threads = num_threads
        self.cache_enabled = cache_enabled
        self._user_cache = _LRUCache(cache_capacity) if cache_enabled else None
        self._item_cache = _LRUCache(cache_capacity) if cache_enabled else None

        self._fallback: Optional[TwoTowerFallBack] = None
        if purchase_data is not None:
            self.set_purchase_data(purchase_data, user_col=user_col, item_col=item_col)

        self.inference_stats = {"n_predictions": 0, "n_users_served": 0,
                                "total_latency_ms": 0.0,
                                "start_time": datetime.now()}
        logger.info("TwoTowerInference loaded from %s (n_users=%d, n_items=%d)",
                    model_path, self._trainer.n_users, self._trainer.n_items)

    def set_purchase_data(self, purchase_data, user_col: str = "user_id",
                          item_col: str = "item_id") -> None:
        """Enable `exclude_purchased=True` on recommend()/batch_recommend()
        by supplying purchase history. Can be called after construction
        instead of passing `purchase_data` to __init__ (e.g. once fresh
        purchase data becomes available in a long-lived serving process).

        Builds the fallback's item-similarity space from the item
        tower's own OUTPUT embeddings (running the whole catalogue
        through `item_tower.forward()` once) -- the same space
        recommend() scores in -- not the raw pre-tower lookup table.
        """
        n_items = self.weights.item_embeddings.shape[0]
        item_tower_outputs = self.item_tower.forward(np.arange(n_items))
        self._fallback = TwoTowerFallBack(purchase_data, item_tower_outputs,
                                          user_col=user_col, item_col=item_col)
        logger.info("Purchase data set: exclude_purchased=True is now available.")

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
                  exclude_items: Optional[List[int]] = None,
                  exclude_purchased: bool = False) -> List[Tuple[int, float]]:
        """Top-N items for `user_id`, sorted by score descending.

        Parameters
        ----------
        exclude_purchased : if True, filter out items `user_id` has
            already purchased (leaving ONLY items never bought before),
            backfilling any resulting shortfall via item-to-item
            cosine-similarity fallback seeded by the user's purchase
            history -- so the result still has `n_items` entries
            (unless the catalogue itself is smaller than that) instead
            of silently coming up short for heavy repeat buyers.
            Requires purchase data -- see set_purchase_data() /
            the `purchase_data` constructor argument.
        """
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

        if exclude_purchased:
            if self._fallback is None:
                raise ValueError(
                    "recommend(exclude_purchased=True) requires purchase data -- pass "
                    "purchase_data=... to TwoTowerInference(...) or call "
                    "set_purchase_data(...) first.")
            # Score the WHOLE (exclude_items-filtered) candidate pool,
            # not just a naive top-N slice, and hand it to
            # clean_recommendations() to filter-then-truncate (instead
            # of truncate-then-filter). This is what actually fixes the
            # "some customers get fewer than n_items" bug: with the old
            # top-N-first ordering, filtering purchased items out
            # *afterward* could shrink an already-truncated list. With
            # the whole pool scored first, filtered[:n_items] already
            # reaches n_items whenever that many non-purchased items
            # exist anywhere in the catalogue. The item-to-item fallback
            # inside clean_recommendations() only matters as a genuine
            # last resort when a user has purchased so much of the
            # catalogue that fewer than n_items non-purchased items
            # exist at all -- at which point nothing (fallback included)
            # can manufacture more real items to recommend.
            ranked_idx = np.argsort(scores)[::-1]
            candidate_pool = [(int(candidate_ids[i]), float(scores[i])) for i in ranked_idx]
            cleaned = self._fallback.clean_recommendations(user_id, candidate_pool, n_items=n_items)
            recommendations = list(zip(cleaned["item_id"].tolist(), cleaned["score"].tolist()))
            if len(recommendations) < n_items:
                logger.warning("recommend(user_id=%s, exclude_purchased=True): only %d/%d "
                               "items available after excluding purchases -- catalogue is "
                               "too small to fill the request even with fallback.",
                               user_id, len(recommendations), n_items)
        else:
            top_n_idx = np.argsort(scores)[::-1][:n_items]
            recommendations = [(int(candidate_ids[i]), float(scores[i])) for i in top_n_idx]

        latency_ms = (time.perf_counter() - start) * 1000
        self.inference_stats["n_predictions"] += len(recommendations)
        self.inference_stats["n_users_served"] += 1
        self.inference_stats["total_latency_ms"] += latency_ms
        return recommendations

    def batch_recommend(self, user_ids: List[int], n_items: int = 10,
                        exclude_purchased: bool = False) -> Dict[int, List[Tuple[int, float]]]:
        """recommend() for multiple users at once."""
        return {uid: self.recommend(uid, n_items=n_items, exclude_purchased=exclude_purchased)
                for uid in user_ids}

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

    def generate_inference_report(self, user_ids: List[int], n_items: int = 10,
                                   experiment_name: str = "ary2tower Inference Run",
                                   output_dir: Optional[Union[str, Path]] = None,
                                   include_embeddings: bool = True,
                                   exclude_purchased: bool = False) -> Path:
        """Generate the interactive HTML inference dashboard for this
        model, covering `recommend()` output for each id in `user_ids`.

        Mirrors AryColBringInference.generate_inference_report()'s API
        shape -- see report.py / narative/a2trearender.py for the
        renderer this delegates to.
        """
        predictions: List[Dict[str, Any]] = list()
        for user_id in user_ids:
            recs = self.recommend(user_id, n_items=n_items, exclude_purchased=exclude_purchased)
            predictions.extend({"user_id": user_id, "item_id": item_id,
                               "score": score, "rank": rank + 1}
                              for rank, (item_id, score) in enumerate(recs))

        item_embeddings = self.weights.item_embeddings if include_embeddings else None
        item_ids = list(range(item_embeddings.shape[0])) if item_embeddings is not None else None

        # Local import to avoid importing arycolbring at module load time
        # for callers who never generate a report.
        from .report import generate_two_tower_report
        return generate_two_tower_report(
            predictions=predictions, metrics=self.get_metrics(),
            item_embeddings=item_embeddings, item_ids=item_ids,
            experiment_name=experiment_name, output_dir=output_dir)
