#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-30"


"""
inference.py
============
Dedicated inference pipeline for the LightGBM LTR framework.

Provides score generation and per-query Top-K ranking logic.  Designed
to be used both as a standalone batch scorer and as an importable class
inside a serving layer (REST / gRPC / Spark UDF).

Classes
-------
LTRInference — loads a trained booster and scores / ranks query groups.
"""

import logging
import os
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from ltr_framework.config import LTRConfig

logger = logging.getLogger(__name__)


class LTRInference:
    """Score and rank items per query using a trained LightGBM booster.

    Parameters
    ----------
    config:
        :class:`~ltr_framework.config.LTRConfig` master config.
    model:
        An already-trained :class:`lgb.Booster`.  If ``None``, the booster
        is loaded from ``config.model.model_path`` on first use.

    Attributes (populated after :meth:`predict` or :meth:`rank_top_k`)
    -------------------------------------------------------------------
    scores_ : np.ndarray
        Raw relevance scores for the last scored DataFrame.
    ranked_df_ : pd.DataFrame
        Top-K ranked output from the last :meth:`rank_top_k` call.
    """

    def __init__(
        self,
        config: LTRConfig,
        model:  Optional[lgb.Booster] = None,
    ) -> None:
        self._config = config
        self._model  = model

        self.scores_:    Optional[np.ndarray]    = None
        self.ranked_df_: Optional[pd.DataFrame]  = None

        logger.debug("LTRInference initialised.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> LTRConfig:
        return self._config

    @property
    def model(self) -> lgb.Booster:
        """Lazy-load the booster on first access."""
        if self._model is None:
            path = self._config.model.model_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No model file found at '{path}'. "
                    "Train the model first or provide a model instance."
                )
            logger.info("Lazy-loading booster from: %s", path)
            self._model = lgb.Booster(model_file=path)
        return self._model

    @model.setter
    def model(self, booster: lgb.Booster) -> None:
        """Inject a trained booster from outside (e.g. from LTRTrainer)."""
        self._model  = booster
        self.scores_ = None       # invalidate stale scores
        logger.debug("LTRInference.model updated via setter.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Assert all feature and query_id columns exist in *df*."""
        required = set(self._config.feature.features) | {
            self._config.feature.query_id
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Inference input is missing columns: {sorted(missing)}"
            )
        logger.debug("Inference input validated. Shape: %s", df.shape)

    # ------------------------------------------------------------------

    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        """Run booster.predict on a feature matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        scores: np.ndarray = self.model.predict(
            X,
            num_iteration = self.model.best_iteration or 0,
        )
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> None:
        """Generate relevance scores for every row in *df*.

        Stores result in ``self.scores_``.

        Parameters
        ----------
        df:
            DataFrame containing at least all feature columns and the
            query_id column.
        """
        logger.info("LTRInference.predict() — %d rows", len(df))
        self._validate_input(df)

        feature_cols = self._config.feature.features
        X = df[feature_cols].to_numpy(dtype=np.float32)

        self.scores_ = self._score_batch(X)

        logger.info(
            "Scoring complete — mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
            float(np.mean(self.scores_)),
            float(np.std(self.scores_)),
            float(np.min(self.scores_)),
            float(np.max(self.scores_)),
        )

    # ------------------------------------------------------------------

    def rank_top_k(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
        extra_cols: Optional[List[str]] = None,
    ) -> None:
        """Score *df* and return the Top-K items per query group.

        Stores result in ``self.ranked_df_``.

        Parameters
        ----------
        df:
            DataFrame containing feature columns and ``query_id``.
        top_k:
            Number of top items to keep per query.  Defaults to
            ``config.inference.top_k``.
        extra_cols:
            Additional columns from *df* to carry through to the output
            (e.g. item_id, item_name).
        """
        k      = top_k if top_k is not None else self._config.inference.top_k
        score_col = self._config.inference.score_col
        q_col  = self._config.feature.query_id

        logger.info(
            "LTRInference.rank_top_k() — top_k=%d | %d rows | %d unique queries",
            k, len(df), df[q_col].nunique(),
        )

        self._validate_input(df)

        # Score all rows
        self.predict(df)

        # Attach scores to a working copy
        out_cols = [q_col] + (extra_cols or []) + self._config.feature.features
        work_df  = df[[c for c in out_cols if c in df.columns]].copy()
        work_df[score_col] = self.scores_

        # Rank within each query group
        query_groups = df[q_col].unique()
        ranked_parts: List[pd.DataFrame] = []

        for qid in tqdm(query_groups, desc="Ranking queries", unit="query"):
            mask      = work_df[q_col] == qid
            group_df  = work_df.loc[mask].copy()

            # Sort descending by score, keep top-k
            group_df  = (
                group_df
                .sort_values(score_col, ascending=False)
                .head(k)
                .reset_index(drop=True)
            )
            group_df["rank"] = range(1, len(group_df) + 1)
            ranked_parts.append(group_df)

        self.ranked_df_ = pd.concat(ranked_parts, ignore_index=True)

        logger.info(
            "rank_top_k complete — output shape: %s", self.ranked_df_.shape
        )

    # ------------------------------------------------------------------

    def save_rankings(self, output_path: Optional[str] = None) -> None:
        """Persist ``self.ranked_df_`` to a CSV file.

        Parameters
        ----------
        output_path:
            Destination path.  Defaults to
            ``<output_dir>/rankings_top{k}.csv``.
        """
        if self.ranked_df_ is None:
            raise RuntimeError(
                "No rankings available. Call rank_top_k() first."
            )

        k    = self._config.inference.top_k
        path = output_path or os.path.join(
            self._config.path.output_dir,
            f"rankings_top{k}.csv",
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.ranked_df_.to_csv(path, index=False)
        logger.info("Rankings saved to: %s", path)

    # ------------------------------------------------------------------

    def score_single_query(
        self,
        query_df: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Convenience method: score a single query's candidate items.

        Parameters
        ----------
        query_df:
            DataFrame for a single query (all rows share the same
            query_id value).
        top_k:
            Items to return.  Defaults to ``config.inference.top_k``.

        Returns
        -------
        pd.DataFrame
            Scored and ranked rows, top-k only.
        """
        k         = top_k if top_k is not None else self._config.inference.top_k
        score_col = self._config.inference.score_col

        self._validate_input(query_df)
        feature_cols = self._config.feature.features

        X      = query_df[feature_cols].to_numpy(dtype=np.float32)
        scores = self._score_batch(X)

        result = query_df.copy()
        result[score_col] = scores
        result = (
            result
            .sort_values(score_col, ascending=False)
            .head(k)
            .reset_index(drop=True)
        )
        result["rank"] = range(1, len(result) + 1)

        logger.debug(
            "score_single_query — returned %d items (top-%d)", len(result), k
        )
        return result
