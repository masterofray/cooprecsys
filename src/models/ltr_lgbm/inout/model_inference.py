#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-09"


"""
model_inference.py
__________________________________________________________________
LightGBM LambdaRank inference for product recommendations.
Handles model loading, scoring, and ranking of items per customer.
Author: MiniMax Agent
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

# Import from data_preprocessing module
from data_preprocessing import (
    LabelEncoderManager,
    FeatureProcessor,
    load_encoders,
    load_feature_columns,
    load_group_sizes,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    """Configuration for model inference."""

    # Paths
    MODEL_PATH = Path("output/ltr_model.txt")
    ENCODER_PATH = Path("output/encoders/encoders.pkl")
    FEATURE_COLS_PATH = Path("output/feature_columns.json")
    GROUP_SIZES_PATH = Path("output/group_sizes.json")

    # Model settings
    QUERY_ID_COLUMN = "CustomerID"
    SCORE_COLUMN = "score"
    RANK_COLUMN = "rank"

    # Top-K for recommendations
    TOP_K = 20

    # String columns to decode
    STRING_COLUMNS = [
        "ProductName", "Class", "Resistant", "IsAllergic",
        "CityName", "EmployeeGender", "Employee_City"
    ]


# ============================================================================
# MODEL INFERENCE CLASS
# ============================================================================

class LTRModelInference:
    """LightGBM LambdaRank inference for scoring and ranking items.

    Attributes:
        config: Inference configuration
        model: Trained LightGBM booster
        encoder_manager: Label encoder manager
        feature_processor: Feature processor
        feature_columns: List of feature column names
        scores_: Last computed scores
        ranked_df_: Last ranked output
    """

    def __init__(
        self,
        config: LTRConfig = None,
        model: Optional[lgb.Booster] = None,
    ) -> None:
        """Initialize inference engine.

        Args:
            config: Inference configuration
            model: Pre-loaded booster (loads from file if None)
        """
        self._config = config or InferenceConfig()
        self._model = model
        self.encoder_manager = LabelEncoderManager()
        self.feature_processor = FeatureProcessor()
        self.feature_columns: List[str] = []
        self.scores_: Optional[np.ndarray] = None
        self.ranked_df_: Optional[pd.DataFrame] = None

        # Try to load from defaults
        if self._config.ENCODER_PATH.exists():
            self.encoder_manager.load(self._config.ENCODER_PATH)

        if self._config.FEATURE_COLS_PATH.exists():
            with open(self._config.FEATURE_COLS_PATH, 'r') as f:
                self.feature_columns = json.load(f)

    @property
    def config(self) -> InferenceConfig:
        return self._config

    @property
    def model(self) -> lgb.Booster:
        """Lazy-load the booster on first access."""
        if self._model is None:
            path = self._config.MODEL_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"No model file found at '{path}'. "
                    "Train the model first or provide a model instance."
                )
            self._model = lgb.Booster(model_file=str(path))
            print(f"Model loaded from: {path}")
        return self._model

    @model.setter
    def model(self, booster: lgb.Booster) -> None:
        """Inject a trained booster."""
        self._model = booster
        self.scores_ = None

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load model from file.

        Args:
            path: Path to model file
        """
        if path is None:
            path = self._config.MODEL_PATH

        self._model = lgb.Booster(model_file=str(path))
        print(f"Model loaded from: {path}")

    def load_encoders(self, path: Optional[Path] = None) -> None:
        """Load label encoders.

        Args:
            path: Path to encoder file
        """
        self.encoder_manager.load(path or self._config.ENCODER_PATH)

    def load_features(self, path: Optional[Path] = None) -> None:
        """Load feature column names.

        Args:
            path: Path to feature columns JSON
        """
        with open(path or self._config.FEATURE_COLS_PATH, 'r') as f:
            self.feature_columns = json.load(f)

    def load_all(self) -> None:
        """Load all required components."""
        self.load_model()
        self.load_encoders()
        self.load_features()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has required columns.

        Args:
            df: Input dataframe
        """
        required = set(self.feature_columns) | {self._config.QUERY_ID_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        """Run model prediction.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of relevance scores
        """
        scores = self.model.predict(
            X,
            num_iteration=self.model.best_iteration or 0
        )
        return scores

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from dataframe.

        Args:
            df: Input dataframe

        Returns:
            Feature matrix as numpy array
        """
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Fill missing columns with 0

        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, predictdata: pd.DataFrame) -> None:
        """Generate relevance scores for all rows.

        Args:
            predictdata: DataFrame with feature columns and query_id
        """
        print(f"Predicting scores for {len(predictdata)} rows...")
        self._validate_input(predictdata)

        X = self._prepare_features(predictdata)
        self.scores_ = self._score_batch(X)

        print(
            f"Scoring complete — mean={np.mean(self.scores_):.4f}, "
            f"std={np.std(self.scores_):.4f}, "
            f"min={np.min(self.scores_):.4f}, max={np.max(self.scores_):.4f}"
        )

    def rank_top_k(
        self,
        predictdata: pd.DataFrame,
        top_k: Optional[int] = None,
        extra_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Score and rank Top-K items per query.

        Args:
            predictdata: DataFrame with features and query_id
            top_k: Number of items per query
            extra_cols: Additional columns to include in output

        Returns:
            DataFrame with top-k ranked items per customer
        """
        k = top_k if top_k is not None else self._config.TOP_K
        q_col = self._config.QUERY_ID_COLUMN
        score_col = self._config.SCORE_COLUMN

        print(f"Ranking top-{k} items for {predictdata[q_col].nunique()} customers...")
        self._validate_input(predictdata)
        self.predict(predictdata)

        # Build working dataframe
        out_cols = [q_col] + (extra_cols or [])
        work_df = predictdata[[c for c in out_cols if c in predictdata.columns]].copy()
        work_df[score_col] = self.scores_

        # Add feature columns to output
        for col in self.feature_columns:
            if col in predictdata.columns:
                work_df[col] = predictdata[col].values

        # Rank within each query group
        ranked_parts: List[pd.DataFrame] = []
        query_groups = predictdata[q_col].unique()

        for qid in tqdm(query_groups, desc="Ranking", unit="customer"):
            mask = work_df[q_col] == qid
            group_df = deepcopy(work_df.loc[mask])

            # Sort descending by score, keep top-k
            group_df = (
                group_df
                .sort_values(score_col, ascending=False)
                .head(k)
                .reset_index(drop=True)
            )
            group_df[self._config.RANK_COLUMN] = range(1, len(group_df) + 1)
            ranked_parts.append(group_df)

        self.ranked_df_ = pd.concat(ranked_parts, ignore_index=True)
        print(f"Ranking complete — output shape: {self.ranked_df_.shape}")

        return self.ranked_df_

    def save_rankings(
        self,
        output_path: Optional[str] = None,
        as_parquet: bool = True,
    ) -> str:
        """Save rankings to file.

        Args:
            output_path: Custom output path
            as_parquet: Save as parquet (else CSV)

        Returns:
            Path where file was saved
        """
        if self.ranked_df_ is None:
            raise RuntimeError("No rankings available. Call rank_top_k() first.")

        ext = '.parquet' if as_parquet else '.csv'
        if output_path is None:
            k = self._config.TOP_K
            output_path = f"output/rankings_top{k}{ext}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if as_parquet:
            self.ranked_df_.to_parquet(
                output_path,
                index=False,
                engine='pyarrow',
                compression='gzip'
            )
        else:
            self.ranked_df_.to_csv(output_path, index=False)

        print(f"Rankings saved to: {output_path}")
        return output_path

    def score_single_query(
        self,
        query_df: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Score and rank a single customer's items.

        Args:
            query_df: DataFrame with one customer's items
            top_k: Number of items to return

        Returns:
            Top-k ranked items
        """
        k = top_k if top_k is not None else self._config.TOP_K
        self._validate_input(query_df)

        X = self._prepare_features(query_df)
        scores = self._score_batch(X)

        result = deepcopy(query_df)
        result[self._config.SCORE_COLUMN] = scores
        result = (
            result
            .sort_values(self._config.SCORE_COLUMN, ascending=False)
            .head(k)
            .reset_index(drop=True)
        )
        result[self._config.RANK_COLUMN] = range(1, len(result) + 1)

        return result

    # ------------------------------------------------------------------
    # Main callable interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        predictdata: pd.DataFrame,
        top_k: Optional[int] = None,
        extra_cols: Optional[List[str]] = None,
        save_output: bool = False,
        output_path: Optional[str] = None,
        parquet: bool = True,
    ) -> pd.DataFrame:
        """Main callable interface for production inference.

        Args:
            predictdata: Input candidate dataset
            top_k: Number of items per query
            extra_cols: Additional columns to preserve
            save_output: Whether to save output
            output_path: Custom output path
            parquet: Save as parquet if True

        Returns:
            Ranked DataFrame
        """
        print("LTRModelInference.__call__() invoked.")
        try:
            if predictdata is None or predictdata.empty:
                raise ValueError("Input dataframe is empty or None.")

            q_col = self._config.QUERY_ID_COLUMN
            k = top_k if top_k is not None else self._config.TOP_K

            if q_col not in predictdata.columns:
                raise ValueError(f"Query column '{q_col}' not found.")

            # Run ranking pipeline
            result = self.rank_top_k(
                predictdata=predictdata,
                top_k=k,
                extra_cols=extra_cols,
            )

            # Save if requested
            if save_output:
                self.save_rankings(output_path, parquet)

            print("Inference complete.")
            return result

        except Exception as e:
            print(f"Error during inference: {e}")
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_inference_engine(
    model_path: Optional[Path] = None,
    encoder_path: Optional[Path] = None,
    feature_cols_path: Optional[Path] = None,
    top_k: int = 20,
) -> LTRModelInference:
    """Create and initialize inference engine.

    Args:
        model_path: Path to model file
        encoder_path: Path to encoder file
        feature_cols_path: Path to feature columns JSON
        top_k: Default top-k

    Returns:
        Initialized LTRModelInference instance
    """
    config = InferenceConfig()

    if model_path:
        config.MODEL_PATH = Path(model_path)
    if encoder_path:
        config.ENCODER_PATH = Path(encoder_path)
    if feature_cols_path:
        config.FEATURE_COLS_PATH = Path(feature_cols_path)
    config.TOP_K = top_k

    engine = LTRModelInference(config=config)
    engine.load_all()

    return engine


def predict_with_fallback(
    predictdata: pd.DataFrame,
    engine: LTRModelInference,
    customer_history: Dict[int, pd.DataFrame],
    top_k: int = 20,
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
    q_col = engine.config.QUERY_ID_COLUMN

    # First, run normal inference
    try:
        ranked = engine.rank_top_k(predictdata, top_k=top_k)
    except Exception as e:
        print(f"Inference failed: {e}")
        return pd.DataFrame()

    # Check if any customer has less than top_k items
    item_counts = ranked.groupby(q_col).size()
    deficient_customers = item_counts[item_counts < top_k].index.tolist()

    if not deficient_customers:
        return ranked

    print(f"Found {len(deficient_customers)} customers with < {top_k} items, using fallback...")

    # For deficient customers, find similar items from their history
    for customer_id in deficient_customers:
        current_items = ranked[ranked[q_col] == customer_id]
        needed = top_k - len(current_items)

        if customer_id not in customer_history:
            print(f"  Customer {customer_id}: No history available")
            continue

        history = customer_history[customer_id]

        # Get characteristics from history
        category_ids = history['CategoryID'].unique().tolist()
        avg_price = history['TotalPrice'].mean() if 'TotalPrice' in history.columns else 0
        city_name = history['CityName'].iloc[0] if 'CityName' in history.columns else None

        # Find similar items from predictdata
        similar_items = predictdata[
            (predictdata['CategoryID'].isin(category_ids)) &
            (~predictdata.index.isin(current_items.index))
        ]

        # Sort by price similarity and city match
        if 'TotalPrice' in similar_items.columns:
            similar_items = similar_items.copy()
            similar_items['price_diff'] = abs(similar_items['TotalPrice'] - avg_price)
            similar_items = similar_items.sort_values('price_diff')

            # Filter by city if available
            if city_name and 'CityName' in similar_items.columns:
                same_city = similar_items[similar_items['CityName'] == city_name]
                if len(same_city) >= needed:
                    similar_items = same_city
                else:
                    similar_items = pd.concat([same_city, similar_items], ignore_index=True)

        # Take top needed items
        fallback_items = similar_items.head(needed).copy()
        fallback_items[q_col] = customer_id

        if len(fallback_items) > 0:
            # Add to ranked results
            ranked = pd.concat([ranked, fallback_items], ignore_index=True)

    # Re-rank for customers with added items
    final_parts = []
    for customer_id in ranked[q_col].unique():
        customer_df = ranked[ranked[q_col] == customer_id].copy()
        customer_df = customer_df.sort_values(engine.config.SCORE_COLUMN, ascending=False)
        customer_df[engine.config.RANK_COLUMN] = range(1, len(customer_df) + 1)
        final_parts.append(customer_df)

    return pd.concat(final_parts, ignore_index=True)


# ============================================================================
# MAIN (TEST)
# ============================================================================

if __name__ == "__main__":
    print("Model Inference Module")
    print("Usage:")
    print("  from model_inference import LTRModelInference, create_inference_engine")
    print("  engine = create_inference_engine()")
    print("  ranked = engine(predictdata)")
    print("  ranked.to_parquet('output/rankings.parquet')")