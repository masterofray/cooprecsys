#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-10"

"""
model_inference.py
__________________________________________________________________
LightGBM LambdaRank inference for product recommendations.
Handles model loading, scoring, and ranking of items per customer.
Uses centralized LTRConfig from lgbm_config.py.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from pathlib import Path
from copy import deepcopy
from typing import List, Optional, Dict, Any, Tuple


LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import LTRConfig, logger, _cfg
from prepare import latest_found
from features import (LabelEncoderManager,
                      load_feature_columns,
                      load_group_sizes,
                      load_encoders)


class LTRModelInference:
    """LightGBM LambdaRank inference for scoring and ranking items.
    Uses centralized LTRConfig from lgbm_config.py for all configuration.
    """
    def __init__(self,
            config       : LTRConfig,
            encoder_path : Optional[Path] = None,
        ) -> None:
        """Initialize inference engine with centralized config.
           config       : Master config from lgbm_config.py. If None, loads from INI.
           encoder_path : Override for label encoder pickle file
        """
        self.ltr_config = config
        self.encman     = LabelEncoderManager()
        self._model     : Optional[lgb.Booster]  = None
        self._data      : Optional[pd.DataFrame] = None
        self.scores_    : Optional[np.ndarray]   = None
        self.ranked_df_ : Optional[pd.DataFrame] = None
        self._base             = Path(self.ltr_config.path.output_dir).resolve().parents[2]
        self.encoder_path      = encoder_path or latest_found(self._base, 'encoder')
        self._feature          = config.feature.features
        self._label            = config.feature.label
        self._query_id         = config.feature.query_id
        
        # Load encoders and feature columns if they exist
        if self.encoder_path.exists():
            try:
                self.encman.load(self.encoder_path)
                logger.info(f"Encoders loaded from: {self.encoder_path}")
            except Exception as e:
                logger.warning(f"Could not load encoders: {e}")


    # ------------------------------------------------------------------
    # Properties for Config values
    # ------------------------------------------------------------------
    @property
    def query_id_col(self) -> str:
        """Query ID column name from config."""
        return self.ltr_config.feature.query_id
    
    @property
    def label_col(self) -> str:
        """Label column name from config."""
        return self.ltr_config.feature.label
    
    @property
    def score_col(self) -> str:
        """Score column name from config."""
        return self.ltr_config.inference.score_col
    
    @property
    def rank_col(self) -> str:
        """Rank column name (fixed, not in LTRConfig)."""
        return "rank"
    
    @property
    def top_k(self) -> int:
        """Default top-k value from config."""
        return self.ltr_config.inference.top_k
    
    @property
    def model_path(self) -> Path:
        """Model file path from config."""
        return Path(self.ltr_config.model.model_path)
    
    @property
    def model(self) -> lgb.Booster:
        """Lazy-load the booster on first access."""
        if self._model is None:
            path = Path(self.model_path)
            if not path.exists():
                path = latest_found(self._base, 'model')
                if not path.exists():
                    raise FileNotFoundError(
                        f"No model file found at '{path}'. "
                        "Train the model first or provide a model instance."
                    )
            self._model = lgb.Booster(model_file = str(path))
            logger.info(f"Model loaded from: {path}")
        return self._model
    
    @model.setter
    def model(self, booster: lgb.Booster) -> None:
        """Inject a trained booster."""
        self._model = booster
        self.scores_ = None

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    @data.setter
    def data(self, value) -> None:
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"data must be a pandas DataFrame, got {type(value).__name__}")
            if value.empty:
                raise ValueError("DataFrame is empty")
            if len(value) < 2:
                raise ValueError(
                    f"DataFrame must have at least 2 rows, got {len(value)}")
        self._data = value

    def load_model(self) -> None:
        self._model = lgb.Booster(model_file = str(self.model_path))
        logger.info(f"Model loaded from: {path}")
    
    def load_encoders(self) -> None:
        self.encman.load(self.encoder_path)
        logger.info(f"Encoders loaded from: {path}")
    
    def load(self) -> None:
        self.load_model()
        self.load_encoders()

    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, data : pd.DataFrame = pd.DataFrame([])) -> None:
        """Validate input dataframe has required columns."""
        if data.empty:
            datac = deepcopy(self._data.columns)
        else:
            datac = data.columns.tolist()
        required  = set(self._feature) | {self._query_id}
        missing   = required - set(datac)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
    
    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        """Run model prediction. The returns is Array of relevance scores"""
        scores = self.model.predict(X,
                 num_iteration = self.model.best_iteration or 0)
        return scores
    
    def _prepare_features(self, data = pd.DataFrame([])) -> np.ndarray:
        """Prepare feature matrix from dataframe."""
        if data.empty:
            data = deepcopy(self._data)
        for item in self._feature:
            if item not in data.columns:
                data[item] = 0
        X = data[self._feature].to_numpy(dtype = np.float32)
        return X



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self) -> None:
        logger.info(f"Predicting scores for {len(self._data)} rows.")
        self._validate_input()
        X = self._prepare_features()
        self.scores_ = self._score_batch(X)
        logger.info(
            f"Scoring complete - mean = {np.mean(self.scores_):.4f}, "
            f"std={np.std(self.scores_):.4f}, "
            f"min={np.min(self.scores_):.4f}, max={np.max(self.scores_):.4f}")

    
    def rank_top_k(self,
                   top_k      : Optional[int] = 20,
                   extra_cols : Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Score and rank Top-K items per query. With parameter:
        - top_k: Number of items per query (uses config.top_k if None)
        - extra_cols: Additional columns to include in output
        It will return for DataFrame with top-k ranked items per customer"""
        k         = top_k if top_k is not None else self.top_k
        q_col     = self._query_id
        score_col = self.score_col
        rank_col  = self.rank_col
        self._validate_input()
        self.predict()
        logger.info(f"Ranking top-{k} items for {self._data[q_col].nunique()} customers.")
        
        # Build working dataframe
        out_cols = [q_col] + (extra_cols or list())
        work_df  = self._data[[c for c in out_cols if c in self._data.columns]].copy()
        work_df[score_col] = self.scores_
        for col in self._feature:
            if col in self._data.columns:
                work_df[col] = self._data[col].values
        
        # Rank within each query group
        ranked_parts: List[pd.DataFrame] = list()
        query_groups = self._data[q_col].unique()
        for qid in tqdm(query_groups,
                         desc   = 'Ranking Process',
                         colour = _cfg.get('tqdm', 'colour'),
                         ncols  = _cfg.getint('tqdm', 'ncols'),
                         unit   = 'Customer',
                         mininterval = 0.1):
            mask     = work_df[q_col] == qid
            group_df = deepcopy(work_df.loc[mask])
            
            # Sort descending by score, keep top-k
            group_df = (group_df
                        .sort_values(score_col, ascending = False)
                        .head(k).reset_index(drop = True))
            group_df[rank_col] = range(1, len(group_df) + 1)
            ranked_parts.append(group_df)
        
        self.ranked_df_ = pd.concat(ranked_parts, ignore_index=True)
        logger.info(f"Ranking complete — output shape: {self.ranked_df_.shape}")
        return self.ranked_df_
    
    def save_rankings(self,
                      output_path: Optional[str] = None,
                      as_parquet: bool = True,
                     ) -> str:
        if self.ranked_df_ is None:
            raise RuntimeError("No rankings available. Call rank_top_k() first.")
        ext = '.parquet' if as_parquet else '.csv'
        if output_path is None:
            output_path = f"output/rankings_top{self.top_k}{ext}"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok = True)
        if as_parquet:
            self.ranked_df_.to_parquet(output_path,
                                       index  = False,
                                       engine = 'pyarrow',
                                       compression = 'gzip')
        else:
            self.ranked_df_.to_csv(output_path, index = False)
        logger.info(f"Rankings saved to: {output_path}")
        return output_path
    
    def score_single_query(self,
                           singledata : pd.DataFrame,
                           top_k      : Optional[int] = 5,
                          ) -> pd.DataFrame:
        """Score and rank a single customer's items."""
        k         = top_k if top_k is not None else self.top_k
        score_col = self.score_col
        rank_col  = self.rank_col
        self._validate_input(singledata)
        X       = self._prepare_features(singledata)
        scores  = self._score_batch(X)
        result  = deepcopy(singledata)
        result[score_col] = scores
        result  = (result.sort_values(score_col, ascending = False)
            .head(k).reset_index(drop=True))
        result[rank_col] = range(1, len(result) + 1)
        return result
    

    def __call__(self,
                 top_k       : Optional[int] = 25,
                 extra_cols  : Optional[List[str]] = None,
                 save_output : bool = False,
                 output_path : Optional[str] = None,
                 parquet     : bool = True,
                ) -> pd.DataFrame:
        logger.info("LTRModelInference.__call__() invoked.")
        try:
            if self._data is None or self._data.empty:
                raise ValueError("Input dataframe is empty or None.")
            q_col = self._query_id
            if q_col not in self._data.columns:
                raise ValueError(f"Query column '{q_col}' not found.")
            result = self.rank_top_k(top_k      = top_k,
                                     extra_cols = extra_cols)
            if save_output:
                self.save_rankings(output_path, parquet)
            logger.info("Inference complete.")
            return result
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise ValueError()



# UTILITY FUNCTIONS
#____________________________________________________________________________
def Inference_Engine(Data       : pd.DataFrame
                     TheConfig  : LTRConfig,
                     encdpath   : Path,
                     model_path : Path = None,
                    ) -> LTRModelInference:
    assert model_path, 'Model_path must not None'
    TheConfig.model.model_path = str(model_path)
    logger.info(f"Model path overridden: {model_path}")
    engine = LTRModelInference(config       = TheConfig,
                               encoder_path = encdpath)
    engine.data = Data
    internalmp = Path(TheConfig.model.model_path)
    DoNotLoad  = False
    if not internalmp.exist():
        try:
            engine.model = lgb.Booster(model_file = str(model_path))
            DoNotLoad    = True
        except Exception as Arc:
            logger.error(f'Check the model_path is exist or not: {model_path}.')
            raise ValueError('Model can load the binary lightGBM') from Arc
    try:
        if no DoNotLoad:
            engine.load()
            logger.debug("Inference engine successfully initialized")
        else:
            engine.load_encoders()
            logger.debug("Encoder engine successfully initialized")
        engine._validate_input()
    except Exception as e:
        logger.warning(f"Could not load all components: {e}")
        logger.info("Engine created but some components may need manual loading")
    return engine


def predict_with_fallback(predictdata      : pd.DataFrame,
                          engine           : LTRModelInference,
                          customer_history : Dict[int, pd.DataFrame],
                          top_k            : Optional[int] = None,
                         ) -> pd.DataFrame:
    """Predict with fallback to similar items for cold-start users.
    Args:
        predictdata: Candidate items
        engine: LTRModelInference instance
        customer_history: Dict mapping customer_id to their purchase history
        top_k: Number of items per customer
    Returns:
        DataFrame with predictions
    """
    q_col     = engine.query_id_col
    score_col = engine.score_col
    rank_col  = engine.rank_col
    k         = top_k if top_k is not None else engine.top_k
    try:
        ranked = engine.rank_top_k(predictdata, top_k = k)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return pd.DataFrame()
    
    # Check if any customer has less than top_k items
    item_counts = ranked.groupby(q_col).size()
    deficient_customers = item_counts[item_counts < k].index.tolist()
    
    if not deficient_customers:
        return ranked
    
    logger.info(f"Found {len(deficient_customers)} customers with < {k} items, using fallback...")
    
    # For deficient customers, find similar items from their history
    for customer_id in deficient_customers:
        current_items = ranked[ranked[q_col] == customer_id]
        needed = k - len(current_items)
        
        if customer_id not in customer_history:
            logger.debug(f"Customer {customer_id}: No history available")
            continue
        
        history = customer_history[customer_id]
        
        # Get characteristics from history
        category_ids = history['CategoryID'].unique().tolist() if 'CategoryID' in history.columns else []
        avg_price = history['TotalPrice'].mean() if 'TotalPrice' in history.columns else 0
        city_name = history['CityName'].iloc[0] if 'CityName' in history.columns else None
        
        # Find similar items from predictdata
        similar_items = predictdata[
            (predictdata['CategoryID'].isin(category_ids)) &
            (~predictdata.index.isin(current_items.index))
        ] if category_ids else pd.DataFrame()
        
        # Sort by price similarity and city match
        if len(similar_items) > 0 and 'TotalPrice' in similar_items.columns:
            similar_items = similar_items.copy()
            similar_items['price_diff'] = abs(similar_items['TotalPrice'] - avg_price)
            similar_items = similar_items.sort_values('price_diff')
            
            # Filter by city if available
            if city_name and 'CityName' in similar_items.columns:
                same_city = similar_items[similar_items['CityName'] == city_name]
                if len(same_city) >= needed:
                    similar_items = same_city
                elif len(same_city) > 0:
                    similar_items = pd.concat([same_city, similar_items], ignore_index=True)
        
        # Take top needed items
        if len(similar_items) > 0:
            fallback_items = similar_items.head(needed).copy()
            fallback_items[q_col] = customer_id
            
            # Add dummy score for fallback items
            if score_col not in fallback_items.columns:
                fallback_items[score_col] = 0.0
            
            # Add to ranked results
            ranked = pd.concat([ranked, fallback_items], ignore_index=True)
    
    # Re-rank for customers with added items
    final_parts = []
    for customer_id in ranked[q_col].unique():
        customer_df = ranked[ranked[q_col] == customer_id].copy()
        if score_col in customer_df.columns:
            customer_df = customer_df.sort_values(score_col, ascending=False)
        customer_df[rank_col] = range(1, len(customer_df) + 1)
        final_parts.append(customer_df)
    
    return pd.concat(final_parts, ignore_index=True)


# ============================================================================
# MAIN (TEST)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Inference Module with LTRConfig Integration")
    print("# Basic usage with auto-configuration")
    from model_inference import create_inference_engine
    engine = create_inference_engine()
    ranked = engine(predictdata)

    print("# With explicit LTRConfig")
    from configs import LTRConfig
    config = LTRConfig.from_ini(features=feature_list)
    engine = create_inference_engine(TheConfig=config)

    print("# Save results")
    ranked = engine(predictdata, save_output=True, 
            output_path='output/my_rankings.parquet')
    print("=" * 60)