#!/usr/bin/env python3


"""
recommendation_handler.py
__________________________________________________________________
Recommendation handler with top-k predictions and fallback logic.
For each CustomerID, generates k=20 item recommendations. If LGBM
prediction doesn't provide enough items, falls back to similar items
based on CategoryID, TotalPrice, and CityName from purchase history.
Author: MiniMax Agent
"""


import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from model_inference
from infcore import LTRModelInference, create_inference_engine


# ============================================================================
# CONFIGURATION
# ============================================================================

class RecommendationConfig:
    """Configuration for recommendation handler."""

    # Paths
    MODEL_PATH = Path("output/ltr_model.txt")
    ENCODER_PATH = Path("output/encoders/encoders.pkl")
    FEATURE_COLS_PATH = Path("output/feature_columns.json")
    HISTORY_PATH = Path("output/customer_history.parquet")

    # Output
    OUTPUT_DIR = Path("output/recommendations")
    OUTPUT_FILE = OUTPUT_DIR / "recommendations_top20.parquet"

    # Settings
    QUERY_ID_COLUMN = "CustomerID"
    TOP_K = 20

    # Similarity fields for fallback
    FALLBACK_SIMILARITY_FIELDS = ["CategoryID", "TotalPrice", "CityName"]

    # String columns (for decoding)
    STRING_COLUMNS = [
        "ProductName", "Class", "Resistant", "IsAllergic",
        "CityName", "EmployeeGender", "Employee_City"
    ]


# ============================================================================
# CUSTOMER HISTORY HANDLER
# ============================================================================

class CustomerHistoryHandler:
    """Handle customer purchase history for fallback recommendations."""

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """Initialize with optional history dataframe.

        Args:
            history_df: DataFrame with customer purchase history
        """
        self.history_df = history_df
        self.history_indexed: Dict[int, pd.DataFrame] = {}

        if history_df is not None:
            self._build_index()

    def _build_index(self) -> None:
        """Build customer history index."""
        if self.history_df is None:
            return

        for customer_id in self.history_df[self.history_df.columns[0]].unique():
            if 'CustomerID' in self.history_df.columns:
                self.history_indexed[customer_id] = self.history_df[
                    self.history_df['CustomerID'] == customer_id
                ].copy()

    def load_history(self, path: Path) -> None:
        """Load customer history from file.

        Args:
            path: Path to parquet file
        """
        self.history_df = pd.read_parquet(path)
        self._build_index()
        print(f"Loaded customer history: {len(self.history_df)} rows")

    def get_history(self, customer_id: int) -> Optional[pd.DataFrame]:
        """Get purchase history for a customer.

        Args:
            customer_id: Customer ID

        Returns:
            DataFrame with customer's purchase history or None
        """
        return self.history_indexed.get(customer_id)

    def get_customer_profile(self, customer_id: int) -> Dict[str, Any]:
        """Get customer profile from history.

        Args:
            customer_id: Customer ID

        Returns:
            Dict with customer characteristics
        """
        history = self.get_history(customer_id)
        if history is None or history.empty:
            return {}

        profile = {}
        if 'CategoryID' in history.columns:
            profile['favorite_categories'] = history['CategoryID'].value_counts().head(5).index.tolist()

        if 'TotalPrice' in history.columns:
            profile['avg_total_price'] = history['TotalPrice'].mean()
            profile['min_total_price'] = history['TotalPrice'].min()
            profile['max_total_price'] = history['TotalPrice'].max()

        if 'CityName' in history.columns:
            profile['city'] = history['CityName'].iloc[0] if not history['CityName'].empty else None

        if 'ProductName' in history.columns:
            profile['purchased_products'] = history['ProductName'].unique().tolist()

        return profile


# ============================================================================
# SIMILARITY FINDER
# ============================================================================

class SimilarityFinder:
    """Find similar items based on category, price, and location."""

    def __init__(self, catalog_df: pd.DataFrame):
        """Initialize with product catalog.

        Args:
            catalog_df: DataFrame with all available products
        """
        self.catalog_df = catalog_df
        self._precompute_similarity_vectors()

    def _precompute_similarity_vectors(self) -> None:
        """Precompute similarity vectors for faster lookup."""
        if 'TotalPrice' in self.catalog_df.columns:
            if self.catalog_df['TotalPrice'].dtype == object:
                self.catalog_df['TotalPrice_numeric'] = pd.to_numeric(
                    self.catalog_df['TotalPrice'], errors='coerce'
                ).fillna(0)
            else:
                self.catalog_df['TotalPrice_numeric'] = self.catalog_df['TotalPrice']

        # Normalize price for similarity
        if 'TotalPrice_numeric' in self.catalog_df.columns:
            price_min = self.catalog_df['TotalPrice_numeric'].min()
            price_max = self.catalog_df['TotalPrice_numeric'].max()
            if price_max > price_min:
                self.catalog_df['price_normalized'] = (
                    (self.catalog_df['TotalPrice_numeric'] - price_min) /
                    (price_max - price_min)
                )
            else:
                self.catalog_df['price_normalized'] = 0.5

    def find_similar_items(
        self,
        customer_profile: Dict[str, Any],
        exclude_ids: List[Any] = None,
        n_items: int = 10,
    ) -> pd.DataFrame:
        """Find similar items based on customer profile.

        Args:
            customer_profile: Customer profile dict
            exclude_ids: Product IDs to exclude
            n_items: Number of items to return

        Returns:
            DataFrame with similar items
        """
        similar_df = self.catalog_df.copy()

        # Exclude already recommended items
        if exclude_ids:
            if 'ProductName' in similar_df.columns:
                similar_df = similar_df[~similar_df['ProductName'].isin(exclude_ids)]

        # Filter by favorite categories
        if 'favorite_categories' in customer_profile:
            similar_df = similar_df[
                similar_df['CategoryID'].isin(customer_profile['favorite_categories'])
            ]

        # Sort by price similarity
        if 'avg_total_price' in customer_profile and 'TotalPrice_numeric' in similar_df.columns:
            avg_price = customer_profile['avg_total_price']
            similar_df['price_diff'] = abs(similar_df['TotalPrice_numeric'] - avg_price)
            similar_df = similar_df.sort_values('price_diff')

        # Filter by city preference
        if 'city' in customer_profile and customer_profile['city']:
            city = customer_profile['city']
            if 'CityName' in similar_df.columns:
                city_matches = similar_df[similar_df['CityName'] == city]
                if len(city_matches) >= n_items:
                    return city_matches.head(n_items)
                else:
                    # Combine city matches with others
                    other_matches = similar_df[similar_df['CityName'] != city]
                    combined = pd.concat([city_matches, other_matches], ignore_index=True)
                    similar_df = combined

        return similar_df.head(n_items)


# ============================================================================
# RECOMMENDATION HANDLER
# ============================================================================

class RecommendationHandler:
    """Main handler for generating top-k recommendations per customer."""

    def __init__(
        self,
        config: Optional[RecommendationConfig] = None,
        inference_engine: Optional[LTRModelInference] = None,
        customer_history: Optional[CustomerHistoryHandler] = None,
    ) -> None:
        """Initialize recommendation handler.

        Args:
            config: Recommendation configuration
            inference_engine: Pre-configured inference engine
            customer_history: Customer history handler
        """
        self.config = config or RecommendationConfig()
        self.inference_engine = inference_engine
        self.customer_history = customer_history or CustomerHistoryHandler()
        self.similarity_finder: Optional[SimilarityFinder] = None

        self.recommendations_df: Optional[pd.DataFrame] = None

    def initialize_inference(self) -> None:
        """Initialize inference engine if not provided."""
        if self.inference_engine is None:
            self.inference_engine = create_inference_engine(
                model_path=self.config.MODEL_PATH,
                encoder_path=self.config.ENCODER_PATH,
                feature_cols_path=self.config.FEATURE_COLS_PATH,
                top_k=self.config.TOP_K,
            )

    def load_customer_history(self, path: Optional[Path] = None) -> None:
        """Load customer history.

        Args:
            path: Path to history file
        """
        self.customer_history.load_history(path or self.config.HISTORY_PATH)

    def generate_recommendations(
        self,
        candidate_data: pd.DataFrame,
        top_k: int = 20,
        use_fallback: bool = True,
    ) -> pd.DataFrame:
        """Generate top-k recommendations for each customer.

        Args:
            candidate_data: DataFrame with candidate products
            top_k: Number of items per customer
            use_fallback: Whether to use fallback for deficient users

        Returns:
            DataFrame with recommendations for each customer
        """
        if self.inference_engine is None:
            self.initialize_inference()

        q_col = self.config.QUERY_ID_COLUMN

        print(f"Generating top-{top_k} recommendations for {candidate_data[q_col].nunique()} customers...")

        # Step 1: Run LGBM inference
        try:
            ranked = self.inference_engine.rank_top_k(
                candidate_data,
                top_k=top_k,
                extra_cols=['ProductName', 'CategoryID', 'TotalPrice', 'CityName'],
            )
        except Exception as e:
            print(f"LGBM inference failed: {e}")
            ranked = pd.DataFrame()

        if ranked.empty:
            print("No predictions from LGBM model")
            return ranked

        # Step 2: Check for deficient customers
        item_counts = ranked.groupby(q_col).size()
        deficient_customers = item_counts[item_counts < top_k].index.tolist()

        print(f"Customers with < {top_k} predictions: {len(deficient_customers)}")

        if not deficient_customers or not use_fallback:
            self.recommendations_df = ranked
            return ranked

        # Step 3: Apply fallback for deficient customers
        print("Applying fallback logic for deficient customers...")
        self.recommendations_df = self._apply_fallback(
            ranked,
            candidate_data,
            deficient_customers,
            top_k,
        )

        return self.recommendations_df

    def _apply_fallback(
        self,
        ranked_df: pd.DataFrame,
        candidate_data: pd.DataFrame,
        deficient_customers: List[int],
        top_k: int,
    ) -> pd.DataFrame:
        """Apply fallback logic for customers with insufficient predictions.

        Args:
            ranked_df: Current ranked predictions
            candidate_data: All candidate products
            deficient_customers: List of customer IDs needing fallback
            top_k: Target number of items

        Returns:
            DataFrame with complete recommendations
        """
        q_col = self.config.QUERY_ID_COLUMN

        # Build similarity finder if not exists
        if self.similarity_finder is None:
            self.similarity_finder = SimilarityFinder(candidate_data)

        final_recommendations = []

        for customer_id in tqdm(deficient_customers, desc="Applying fallback"):
            # Get current predictions for this customer
            current = ranked_df[ranked_df[q_col] == customer_id].copy()
            current_items = current['ProductName'].tolist() if 'ProductName' in current.columns else []
            needed = top_k - len(current)

            if needed <= 0:
                final_recommendations.append(current)
                continue

            # Get customer profile from history
            profile = self.customer_history.get_customer_profile(customer_id)

            if not profile:
                # If no history, just use current predictions
                final_recommendations.append(current)
                continue

            # Find similar items
            similar_items = self.similarity_finder.find_similar_items(
                customer_profile=profile,
                exclude_ids=current_items,
                n_items=needed,
            )

            # Create fallback entries
            if not similar_items.empty:
                fallback_df = similar_items.copy()
                fallback_df[q_col] = customer_id

                # Copy scores from current (lowest) or use a default
                fallback_df[self.inference_engine.config.SCORE_COLUMN] = (
                    current[self.inference_engine.config.SCORE_COLUMN].min()
                    if not current.empty else -1.0
                )

                # Mark as fallback
                fallback_df['is_fallback'] = True

                # Add to current predictions
                combined = pd.concat([current, fallback_df], ignore_index=True)
            else:
                combined = current

            # Re-rank and trim to top_k
            combined = combined.sort_values(
                self.inference_engine.config.SCORE_COLUMN,
                ascending=False
            ).head(top_k)

            combined['rank'] = range(1, len(combined) + 1)
            final_recommendations.append(combined)

        # Get all customers that had full predictions
        full_customers = ranked_df.groupby(q_col).size()
        full_customers = full_customers[full_customers >= top_k].index.tolist()

        for customer_id in full_customers:
            customer_df = ranked_df[ranked_df[q_col] == customer_id].copy()
            if 'is_fallback' not in customer_df.columns:
                customer_df['is_fallback'] = False
            final_recommendations.append(customer_df)

        result = pd.concat(final_recommendations, ignore_index=True)
        return result

    def decode_recommendations(self) -> pd.DataFrame:
        """Decode encoded columns back to human-readable format.

        Returns:
            DataFrame with decoded string columns
        """
        if self.recommendations_df is None:
            print("No recommendations to decode")
            return pd.DataFrame()

        df = self.recommendations_df.copy()

        # Decode string columns
        encoder = self.inference_engine.encoder_manager

        for col in self.config.STRING_COLUMNS:
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns:
                # Decode using inverse transform
                classes = encoder.get_classes(col)
                if classes:
                    df[col] = df[encoded_col].apply(
                        lambda x: classes[int(x)] if int(x) < len(classes) else "Unknown"
                    )

        return df

    def save_recommendations(
        self,
        output_path: Optional[Path] = None,
        decode: bool = True,
        as_parquet: bool = True,
    ) -> str:
        """Save recommendations to file.

        Args:
            output_path: Custom output path
            decode: Whether to decode before saving
            as_parquet: Save as parquet (else CSV)

        Returns:
            Path where file was saved
        """
        if self.recommendations_df is None:
            raise RuntimeError("No recommendations to save. Run generate_recommendations() first.")

        df = self.recommendations_df if not decode else self.decode_recommendations()

        if output_path is None:
            output_path = self.config.OUTPUT_FILE

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if as_parquet:
            df.to_parquet(
                output_path,
                index=False,
                engine='pyarrow',
                compression='gzip'
            )
        else:
            df.to_csv(output_path, index=False)

        print(f"Recommendations saved to: {output_path}")
        return str(output_path)

    def get_customer_recommendations(
        self,
        customer_id: int,
        n_items: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get recommendations for a specific customer.

        Args:
            customer_id: Customer ID
            n_items: Number of items to return (default: config.TOP_K)

        Returns:
            DataFrame with customer's recommendations
        """
        if self.recommendations_df is None:
            return pd.DataFrame()

        n_items = n_items or self.config.TOP_K
        q_col = self.config.QUERY_ID_COLUMN

        customer_recs = self.recommendations_df[
            self.recommendations_df[q_col] == customer_id
        ].head(n_items)

        return customer_recs

    def __call__(
        self,
        candidate_data: pd.DataFrame,
        top_k: int = 20,
        use_fallback: bool = True,
        save_output: bool = True,
        decode: bool = True,
    ) -> pd.DataFrame:
        """Main callable interface.

        Args:
            candidate_data: Candidate products DataFrame
            top_k: Number of items per customer
            use_fallback: Use fallback for deficient customers
            save_output: Save recommendations to file
            decode: Decode before saving

        Returns:
            DataFrame with recommendations
        """
        print("=" * 60)
        print("RECOMMENDATION HANDLER")
        print("=" * 60)

        recommendations = self.generate_recommendations(
            candidate_data=candidate_data,
            top_k=top_k,
            use_fallback=use_fallback,
        )

        if save_output and not recommendations.empty:
            self.save_recommendations(decode=decode)

        print("=" * 60)
        print(f"Generated {len(recommendations)} recommendations for "
              f"{recommendations[self.config.QUERY_ID_COLUMN].nunique()} customers")
        print("=" * 60)

        return recommendations


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def generate_recommendations_pipeline(
    candidate_data: pd.DataFrame,
    model_path: Path = Path("output/ltr_model.txt"),
    encoder_path: Path = Path("output/encoders/encoders.pkl"),
    feature_cols_path: Path = Path("output/feature_columns.json"),
    history_path: Optional[Path] = None,
    top_k: int = 20,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run complete recommendation pipeline.

    Args:
        candidate_data: Candidate products
        model_path: Path to model
        encoder_path: Path to encoders
        feature_cols_path: Path to feature columns
        history_path: Optional path to customer history
        top_k: Number of items per customer
        output_path: Output path

    Returns:
        DataFrame with recommendations
    """
    config = RecommendationConfig()
    config.MODEL_PATH = model_path
    config.ENCODER_PATH = encoder_path
    config.FEATURE_COLS_PATH = feature_cols_path
    config.TOP_K = top_k

    handler = RecommendationHandler(config=config)
    handler.initialize_inference()

    if history_path and history_path.exists():
        handler.load_customer_history(history_path)

    recommendations = handler(
        candidate_data=candidate_data,
        top_k=top_k,
        use_fallback=True,
        save_output=True,
        decode=True,
    )

    return recommendations


# ============================================================================
# MAIN (DEMO)
# ============================================================================

if __name__ == "__main__":
    print("Recommendation Handler Module")
    print("\nAvailable classes:")
    print("  - RecommendationHandler: Main recommendation engine")
    print("  - CustomerHistoryHandler: Customer history management")
    print("  - SimilarityFinder: Find similar items")
    print("\nUsage:")
    print("  from recommendation_handler import RecommendationHandler")
    print("  handler = RecommendationHandler()")
    print("  recommendations = handler(candidate_data)")
    print("\nOr use standalone function:")
    print("  from recommendation_handler import generate_recommendations_pipeline")
    print("  recs = generate_recommendations_pipeline(candidate_data)")