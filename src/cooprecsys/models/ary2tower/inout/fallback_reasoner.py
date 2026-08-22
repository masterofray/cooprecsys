#!/usr/bin/env python3
"""Residual candidate generation for ary2tower inference.

The fallback is deliberately NOT item-to-item filtering. Its job is to fill
remaining recommendation slots with candidates that are globally relevant,
recent and statistically reliable when the learned ranker/candidate pool is
short. The primary path should normally score the full catalogue with the
compiled two-tower kernel; this class is the final backstop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from ....configs import logger


class TwoTowerFallBack:
    SOURCE_MODEL = "model_topn"
    SOURCE_POPULARITY = "bayesian_popularity_fallback"

    def __init__(
        self,
        purchase_data: pd.DataFrame,
        n_items: int,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: Optional[str] = None,
        half_life_days: float = 30.0,
        prior_strength: float = 10.0,
    ):
        if user_col not in purchase_data.columns:
            raise ValueError(f"user_col '{user_col}' not found in purchase_data")
        if item_col not in purchase_data.columns:
            raise ValueError(f"item_col '{item_col}' not found in purchase_data")
        if n_items <= 0:
            raise ValueError("n_items must be > 0")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be > 0")

        self.user_col = user_col
        self.item_col = item_col
        self.n_items = int(n_items)
        self.half_life_days = float(half_life_days)
        self.prior_strength = float(prior_strength)
        self._purchase_map = self._build_purchase_map(purchase_data)
        self._fallback_scores = self._build_fallback_scores(
            purchase_data, timestamp_col=timestamp_col
        )

        logger.info(
            "TwoTowerFallBack initialized: %d users, %d catalogue items, strategy=%s",
            len(self._purchase_map), self.n_items, self.SOURCE_POPULARITY,
        )

    def _build_purchase_map(self, data: pd.DataFrame) -> Dict[Any, Set[int]]:
        out: Dict[Any, Set[int]] = {}
        for uid, group in data.groupby(self.user_col, sort=False):
            out[uid] = set(int(x) for x in group[self.item_col].dropna().tolist())
        return out

    def purchased_items(self, user_id: Any) -> Set[int]:
        return self._purchase_map.get(user_id, set())

    def _build_fallback_scores(
        self, data: pd.DataFrame, timestamp_col: Optional[str]
    ) -> np.ndarray:
        """Bayesian-smoothed global prior, optionally time-decayed.

        Score = log1p(weighted interaction count) shrunk toward the global
        mean. This is substantially more robust than raw popularity and does
        not use item-item similarity.
        """
        valid = pd.to_numeric(data[self.item_col], errors="coerce")
        mask = valid.notna() & (valid >= 0) & (valid < self.n_items)
        items = valid[mask].astype(np.int64).to_numpy()
        weights = np.ones(items.shape[0], dtype=np.float64)

        time_col = timestamp_col
        if time_col is None:
            for candidate in ("timestamp", "event_time", "created_at", "datetime", "date"):
                if candidate in data.columns:
                    time_col = candidate
                    break
        if time_col is not None and time_col in data.columns and items.size:
            ts = pd.to_datetime(data.loc[mask, time_col], errors="coerce", utc=True)
            if ts.notna().any():
                age_days = (ts.max() - ts).dt.total_seconds().to_numpy() / 86400.0
                age_days = np.maximum(np.nan_to_num(age_days, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
                weights = np.exp(-np.log(2.0) * age_days / max(self.half_life_days, 1e-6))

        weighted_counts = np.bincount(items, weights=weights, minlength=self.n_items).astype(np.float64)
        positive = weighted_counts[weighted_counts > 0]
        global_mean = float(positive.mean()) if positive.size else 0.0
        smoothed = (weighted_counts + self.prior_strength * global_mean) / (1.0 + self.prior_strength)
        scores = np.log1p(smoothed)

        # Deterministic tie-breaker: lower item ids come later only when scores tie.
        scores += np.linspace(0.0, -1e-9, self.n_items)
        return scores.astype(np.float64, copy=False)

    def popularity_candidates(
        self,
        exclude: Optional[Set[int]] = None,
        n: int = 10,
    ) -> List[Tuple[int, float]]:
        if n <= 0:
            return []
        excluded = {int(x) for x in (exclude or set())}
        order = np.argsort(self._fallback_scores)[::-1]
        out: List[Tuple[int, float]] = []
        for iid in order:
            iid = int(iid)
            if iid in excluded:
                continue
            out.append((iid, float(self._fallback_scores[iid])))
            if len(out) >= n:
                break
        return out

    def clean_recommendations(
        self,
        user_id: Any,
        candidate_pool: List[Tuple[int, float]],
        n_items: int = 10,
    ) -> pd.DataFrame:
        """Filter purchased/excluded candidates and fill gaps from the prior.

        Contract: returns exactly ``n_items`` unique, unseen catalogue items
        whenever that many eligible items exist. No item-to-item method is used.
        """
        if n_items <= 0:
            raise ValueError("n_items must be > 0")

        bought = self.purchased_items(user_id)
        seen_model: Set[int] = set()
        rows: List[Dict[str, Any]] = []

        for item_id, score in candidate_pool:
            iid = int(item_id)
            if iid in bought or iid in seen_model or iid < 0 or iid >= self.n_items:
                continue
            seen_model.add(iid)
            rows.append({
                "user_id": user_id, "item_id": iid, "score": float(score),
                "source": self.SOURCE_MODEL, "is_fallback": False,
            })
            if len(rows) >= n_items:
                break

        if len(rows) < n_items:
            excluded = bought | seen_model
            for iid, prior_score in self.popularity_candidates(exclude=excluded, n=n_items - len(rows)):
                rows.append({
                    "user_id": user_id, "item_id": iid, "score": prior_score,
                    "source": self.SOURCE_POPULARITY, "is_fallback": True,
                })

        if len(rows) < n_items:
            logger.warning(
                "user_id=%s: unable to fill %d requested slots; only %d eligible unique items exist",
                user_id, n_items, len(rows),
            )

        result = pd.DataFrame(rows, columns=[
            "user_id", "item_id", "score", "source", "is_fallback"
        ])
        if not result.empty:
            result.insert(1, "rank", np.arange(1, len(result) + 1))
        return result
