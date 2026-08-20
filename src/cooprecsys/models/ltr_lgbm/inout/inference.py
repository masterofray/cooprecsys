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
________________________________________
Dedicated inference pipeline for the LightGBM LTR framework.
Provides score generation and per-query Top-K ranking logic.  Designed
to be used both as a standalone batch scorer and as an importable class
inside a serving layer (REST / gRPC / Spark UDF).
The product Classes is LTRInference 
That class does to load a trained booster and scores / ranks query groups.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from typing import List, Optional
from ....configs import LTRConfig, logger, _cfg


class LTRInference:
    """Score and rank items per query using a trained LightGBM booster.
    config : `LTRConfig` master config.
    model  : An already-trained :class:`lgb.Booster`.  If ``None``, the booster
             is loaded from ``config.model.model_path`` on first use.
    _________________________________________________________________
    Attributes (populated after :meth:`predict` or :meth:`rank_top_k`)
    scores_    : np.ndarray, Raw relevance scores for the last scored DataFrame.
    ranked_df_ : pd.DataFrame, Top-K ranked output from the 
                 last :meth:`rank_top_k` call.
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
                    "Train the model first or provide a model instance.")
            self._model = lgb.Booster(model_file=path)
            logger.debug("Lazy-loading booster from: %s", path)
        return self._model

    @model.setter
    def model(self, booster: lgb.Booster) -> None:
        """Inject a trained booster from outside (e.g. from LTRTrainer)."""
        self._model  = booster
        self.scores_ = None
        logger.debug("LTRInference.model updated via setter.")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Assert all feature and query_id columns exist in *df*."""
        required = set(self._config.feature.features) | {
                   self._config.feature.query_id}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Inference input is missing columns: {sorted(missing)}")
        logger.debug("Inference input validated. Shape: %s", df.shape)

    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        """Run booster.predict on a feature matrix.
        X : np.ndarray of shape (n_samples, n_features)
        """
        scores = self.model.predict(X,
                 num_iteration = self.model.best_iteration or 0)
        return scores


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, 
                predictdata: pd.DataFrame,
        ) -> None:
        """Generate relevance scores for every row in *predictdata*.
        Stores result in ``self.scores_``.
        predictdata:
            DataFrame containing at least all feature columns and the
            query_id column.
        """
        logger.info("LTRInference.predict() — %d rows", len(predictdata))
        self._validate_input(predictdata)
        feature_cols = self._config.feature.features
        X = predictdata[feature_cols].to_numpy(dtype=np.float32)
        self.scores_ = self._score_batch(X)
        logger.info(
            "Scoring complete — mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
            float(np.mean(self.scores_)),
            float(np.std(self.scores_)),
            float(np.min(self.scores_)),
            float(np.max(self.scores_)),
            )


    def rank_top_k(self,
                   predictdata : pd.DataFrame,
                   top_k       : Optional[int] = None,
                   extra_cols  : Optional[List[str]] = None,
        ) -> None:
        """Score *predictdata* and return the Top-K items per query group.
        Stores result in ``self.ranked_df_``.
        predictdata : DataFrame containing feature columns and ``query_id``.
        top_k       : Number of top items to keep per query.  Defaults to
                      ``config.inference.top_k``.
        extra_cols  : Additional columns from *predictdata* to carry through to the output
                      (e.g. item_id, item_name).
        """
        k           = top_k if top_k is not None else self._config.inference.top_k
        score_col   = self._config.inference.score_col
        q_col       = self._config.feature.query_id
        logger.info("LTRInference.rank_top_k() — top_k=%d | %d rows | %d unique queries",
                     k, len(predictdata), predictdata[q_col].nunique())
        self._validate_input(predictdata)
        self.predict(predictdata)

        # Attach scores to a working copy
        out_cols = [q_col] + (extra_cols or []) + self._config.feature.features
        work_df  = predictdata[[c for c in out_cols if c in predictdata.columns]].copy()
        work_df[score_col] = self.scores_

        # Rank within each query group
        query_groups = predictdata[q_col].unique()
        ranked_parts: List[pd.DataFrame] = list()
        for qid in tqdm(query_groups,
                        desc        = "Ranking queries", 
                        unit        = "query",
                        colour      = _cfg.get('tqdm', 'colour'),
                        ncols       = _cfg.getint('tqdm', 'ncols'),
                        bar_format  = _cfg.get('tqdm', 'BarFormats'),
                        mininterval = 0.1):
            mask      = work_df[q_col] == qid
            group_df  = deepcopy(work_df.loc[mask])

            # Sort descending by score, keep top-k
            group_df  = (group_df
                        .sort_values(score_col, ascending=False)
                        .head(k)
                        .reset_index(drop=True))
            group_df["rank"] = range(1, len(group_df) + 1)
            ranked_parts.append(group_df)
        self.ranked_df_ = pd.concat(ranked_parts, ignore_index=True)
        logger.info("rank_top_k complete — output shape: %s", self.ranked_df_.shape)


    def save_rankings(self, 
                      output_path: Optional[str] = None,
                     ) -> None:
        """Persist ``self.ranked_df_`` to a CSV file.
        output_path: Destination path. Defaults to
                    ``<output_dir>/rankings_top{k}.csv``.
        """
        if self.ranked_df_ is None:
            raise RuntimeError("No rankings available. Call rank_top_k() first.")
        k    = self._config.inference.top_k
        parquet = _cfg.getboolean('INFERENCE', 'parquet')
        ext  = '.csv' if not parquet else '.parquet'
        path = output_path or os.path.join(
            self._config.path.output_dir,
            f"rankings_top{k}{ext}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
        if parquet:
            self.ranked_df_.to_parquet(
                path, 
                index      = False, 
                engine     = 'pyarrow', 
                compression= 'gzip')
        else:
            self.ranked_df_.to_csv(path, index = False)
        logger.info("Rankings saved to: %s", path)
        return path


    def score_single_query(self,
                           query_df: pd.DataFrame,
                           top_k: Optional[int] = None,
        ) -> pd.DataFrame:
        """Convenience method: score a single query's candidate items.
        query_df: DataFrame for a single query (all rows share the same
                  query_id value).
        top_k   : Items to return. Defaults to `config.inference.top_k`.
        Returns : Dataframe that be scored and ranked rows, top-k only.
        """
        k         = top_k if top_k is not None else self._config.inference.top_k
        score_col = self._config.inference.score_col
        self._validate_input(query_df)
        feature_cols = self._config.feature.features
        X      = query_df[feature_cols].to_numpy(dtype=np.float32)
        scores = self._score_batch(X)
        result = deepcopy(query_df)
        result[score_col] = scores
        result = (result
                  .sort_values(score_col, ascending=False)
                  .head(k)
                  .reset_index(drop=True))
        result["rank"] = range(1, len(result) + 1)
        logger.debug("score_single_query -- returned %d items (top-%d)", len(result), k)
        return result


    # ------------------------------------------------------------------
    # Call everything
    # ------------------------------------------------------------------
    def __call__(self,
                 predictdata : pd.DataFrame,
                 top_k       : Optional[int] = None,
                 extra_cols  : Optional[List[str]] = None,
                 save_output : bool = False,
                 output_path : Optional[str] = None,
                 parquet     : bool = True,
        ) -> pd.DataFrame:
        """
        Main callable interface for production inference pipeline.
        predictdata : pd.DataFrame, Input candidate dataset 
                      containing query_id + feature columns.
        top_k       : Optional[int], Number of top rows per query group.
                      Defaults to config.inference.top_k.
        extra_cols  : Optional[List[str]], Additional columns to preserve in output.
        save_output : bool, If True, persist result using save_rankings().
        output_path : Optional[str], Custom path for saved output.
        parquet     : bool, Save as parquet if True, else CSV.
        """
        logger.info("LTRInference.__call__() invoked.")
        try:
            if predictdata is None:
                raise ValueError("Input dataframe is None.")
            if not isinstance(predictdata, pd.DataFrame):
                raise TypeError(
                    f"Expected pd.DataFrame, got {type(predictdata).__name__}")
            if predictdata.empty:
                raise ValueError("Input dataframe is empty.")
            logger.debug("Input dataframe validated.")
            logger.debug("Input shape: %s", predictdata.shape)
            logger.debug("Input columns: %s", list(predictdata.columns))

            q_col = self._config.feature.query_id
            if q_col not in predictdata.columns:
                raise ValueError(f"Query column '{q_col}' not found in dataframe.")
            logger.debug("Unique queries detected: %d",
                predictdata[q_col].nunique())

            # ---------------------------------------------------------
            # Determine top_k
            # ---------------------------------------------------------
            k = top_k if top_k is not None else self._config.inference.top_k
            if not isinstance(k, int):
                raise TypeError("top_k must be integer.")
            if k <= 0:
                raise ValueError("top_k must be > 0.")
            logger.debug("Using top_k = %d", k)

            # ---------------------------------------------------------
            # Check model readiness
            # ---------------------------------------------------------
            _ = self.model
            logger.debug("Model loaded and ready.")

            # ---------------------------------------------------------
            # Run ranking pipeline
            # ---------------------------------------------------------
            logger.info("Starting rank_top_k pipeline.")
            self.rank_top_k(predictdata = predictdata,
                            top_k       = k,
                            extra_cols  = extra_cols)
            if self.ranked_df_ is None:
                raise RuntimeError("Ranking finished but ranked_df_ is None.")
            logger.debug("Ranking success. Output shape: %s",
                self.ranked_df_.shape)

            # ---------------------------------------------------------
            # Save output if requested
            # ---------------------------------------------------------
            if save_output:
                logger.debug("Saving ranking output.")
                self.save_rankings(
                    output_path = output_path,
                    parquet     = parquet)
                logger.debug("Output successfully saved.")
            logger.info("LTRInference.__call__() completed successfully.")
            return self.ranked_df_

        except FileNotFoundError as e:
            logger.exception("Model file not found: %s", str(e))
            raise
        except ValueError as e:
            logger.exception("Validation error in __call__: %s", str(e))
            raise
        except TypeError as e:
            logger.exception("Type error in __call__: %s", str(e))
            raise
        except RuntimeError as e:
            logger.exception("Runtime error in __call__: %s", str(e))
            raise
        except Exception as arc:
            logger.exception(
                "Unexpected error during inference pipeline: %s",str(arc))
            raise


if __name__ == '__main__':
    pass