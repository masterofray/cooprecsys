#!/usr/bin/env python3
"""
main.py
__________________________________________________________________
Comprehensive preprocessing pipeline for LightGBM Learning to Rank.
Processes columns with 'date' as datetime, 'hours' as time, encodes
strings with LabelEncoder, and performs feature engineering.
Author: MiniMax Agent
"""

import os
import sys
import json
import pickle
from datetime import datetime, date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    # Paths
    INPUT_PATH = Path("user_input_files/sampledata.parquet")
    OUTPUT_DIR = Path("output")
    ENCODER_DIR = OUTPUT_DIR / "encoders"
    PROCESSED_DIR = OUTPUT_DIR / "processed"

    # Columns to handle
    DATE_COLUMNS = ["SalesDate", "Product_ModifyDate"]
    TIME_COLUMNS = ["SalesHours"]

    # Columns to drop (redundant or not needed for ML)
    DROP_COLUMNS = ["FirstName", "SalesID", "EmployeeFirstName", "CountryName"]

    # Target column for ranking
    TARGET_COLUMN = "Quantity"  # Using Quantity as relevance signal

    # Query ID column
    QUERY_ID_COLUMN = "CustomerID"

    # String columns to encode
    STRING_COLUMNS = [
        "ProductName", "Class", "Resistant", "IsAllergic",
        "CityName", "EmployeeGender", "Employee_City"
    ]

    # Numerical columns
    NUMERICAL_COLUMNS = [
        "ProductPrice", "Discount", "TotalPrice", "CategoryID",
        "VitalityDays", "EmployeeID", "EmployeeAge", "YearsWorking"
    ]

    # k for top-k recommendations
    TOP_K = 20

    # Feature engineering settings
    FEATURE_ENGINEERING = {
        "date_features": True,
        "time_features": True,
        "interaction_features": True,
    }


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

class DateFeatureEngineer:
    """Engineer features from date columns."""

    @staticmethod
    def extract_features(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Extract date-based features from a datetime column.

        Args:
            df: Input dataframe
            col_name: Name of the date column

        Returns:
            DataFrame with new features added
        """
        # Ensure column is datetime
        if df[col_name].dtype == 'object':
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

        # Extract components
        df[f"{col_name}_year"] = df[col_name].dt.year
        df[f"{col_name}_month"] = df[col_name].dt.month
        df[f"{col_name}_day"] = df[col_name].dt.day
        df[f"{col_name}_dayofweek"] = df[col_name].dt.dayofweek  # 0=Monday, 6=Sunday
        df[f"{col_name}_quarter"] = df[col_name].dt.quarter
        df[f"{col_name}_is_weekend"] = (df[col_name].dt.dayofweek >= 5).astype(int)

        # Days since reference date (e.g., 2018-01-01)
        reference_date = pd.Timestamp("2018-01-01")
        df[f"{col_name}_days_since_ref"] = (df[col_name] - reference_date).dt.days

        return df


class TimeFeatureEngineer:
    """Engineer features from time columns."""

    @staticmethod
    def extract_features(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Extract time-based features from a time column.

        Args:
            df: Input dataframe
            col_name: Name of the time column

        Returns:
            DataFrame with new features added
        """
        # Parse time if string
        if df[col_name].dtype == 'object':
            df[col_name] = pd.to_datetime(df[col_name], format='%H:%M:%S.%f', errors='coerce').dt.time
        elif hasattr(df[col_name], 'dt'):
            df[col_name] = df[col_name].dt.time

        # Extract components using apply (since time objects)
        df[f"{col_name}_hour"] = df[col_name].apply(lambda x: x.hour if isinstance(x, time) else 0)
        df[f"{col_name}_minute"] = df[col_name].apply(lambda x: x.minute if isinstance(x, time) else 0)
        df[f"{col_name}_second"] = df[col_name].apply(lambda x: x.second if isinstance(x, time) else 0)

        # Time period categories
        df[f"{col_name}_period"] = pd.cut(
            df[f"{col_name}_hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=[0, 1, 2, 3]  # 0=Night(0-6), 1=Morning(6-12), 2=Afternoon(12-18), 3=Evening(18-24)
        ).astype(int)

        # Is business hours (9-17)
        df[f"{col_name}_is_business_hours"] = (
            (df[f"{col_name}_hour"] >= 9) & (df[f"{col_name}_hour"] <= 17)
        ).astype(int)

        # Minute of day (0-1439)
        df[f"{col_name}_minute_of_day"] = (
            df[f"{col_name}_hour"] * 60 + df[f"{col_name}_minute"]
        )

        return df


class InteractionFeatureEngineer:
    """Create interaction features between columns."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with new features added
        """
        # Price-related interactions
        df["price_per_unit"] = df["ProductPrice"] * (1 - df["Discount"])
        df["total_value"] = df["ProductPrice"] * df["Quantity"]

        # Age-related interactions
        if "EmployeeAge" in df.columns:
            df["employee_experience_ratio"] = df["YearsWorking"] / (df["EmployeeAge"] + 1)

        # Customer preferences (if TotalPrice is numeric after conversion)
        if "TotalPrice" in df.columns:
            if df["TotalPrice"].dtype == object:
                # Handle Decimal objects
                df["TotalPrice_numeric"] = pd.to_numeric(
                    df["TotalPrice"].apply(lambda x: float(x) if not pd.isna(x) else 0)
                )
            else:
                df["TotalPrice_numeric"] = df["TotalPrice"].astype(float)

            df["avg_price_per_item"] = df["TotalPrice_numeric"] / (df["Quantity"] + 1)

        return df


# ============================================================================
# LABEL ENCODER MANAGER
# ============================================================================

class LabelEncoderManager:
    """Manages label encoders for string columns."""

    def __init__(self, encoder_dir: Path):
        self.encoder_dir = Path(encoder_dir)
        self.encoders: Dict[str, LabelEncoder] = {}
        self.encoder_classes: Dict[str, List[Any]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Fit label encoders on specified columns.

        Args:
            df: Input dataframe
            columns: List of column names to encode
        """
        self.encoder_dir.mkdir(parents=True, exist_ok=True)

        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in dataframe")
                continue

            # Create encoder
            encoder = LabelEncoder()

            # Handle missing values
            values = df[col].fillna("__MISSING__").astype(str)
            encoder.fit(values)

            # Store encoder and classes
            self.encoders[col] = encoder
            self.encoder_classes[col] = encoder.classes_.tolist()

            print(f"Fitted encoder for '{col}': {len(encoder.classes_)} unique values")

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform columns using fitted encoders.

        Args:
            df: Input dataframe
            columns: List of column names to transform

        Returns:
            DataFrame with encoded columns
        """
        df = df.copy()

        for col in columns:
            if col not in self.encoders:
                print(f"Warning: No encoder found for '{col}'")
                continue

            encoder = self.encoders[col]

            # Handle missing values
            values = df[col].fillna("__MISSING__").astype(str)

            # Handle unseen values
            known_classes = set(encoder.classes_)
            values = values.apply(
                lambda x: x if x in known_classes else "__MISSING__"
            )

            df[f"{col}_encoded"] = encoder.transform(values)

        return df

    def save(self, path: Optional[Path] = None) -> None:
        """Save encoders to file.

        Args:
            path: Path to save encoder file
        """
        if path is None:
            path = self.encoder_dir / "encoders.pkl"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        encoder_data = {
            "encoders": self.encoders,
            "encoder_classes": self.encoder_classes,
        }

        with open(path, 'wb') as f:
            pickle.dump(encoder_data, f)

        print(f"Encoders saved to: {path}")

    def load(self, path: Path) -> None:
        """Load encoders from file.

        Args:
            path: Path to encoder file
        """
        with open(path, 'rb') as f:
            encoder_data = pickle.load(f)

        self.encoders = encoder_data["encoders"]
        self.encoder_classes = encoder_data["encoder_classes"]
        print(f"Encoders loaded from: {path}")


# ============================================================================
# DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """Main data preprocessing class."""

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        self.config = config
        self.encoder_manager = LabelEncoderManager(config.ENCODER_DIR)

    def load_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load data from parquet file.

        Args:
            path: Path to parquet file

        Returns:
            DataFrame with loaded data
        """
        if path is None:
            path = self.config.INPUT_PATH

        df = pd.read_parquet(path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column types.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with converted types
        """
        df = df.copy()

        # Convert date columns
        for col in self.config.DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted '{col}' to datetime")

        # Convert time columns
        for col in self.config.TIME_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S.%f', errors='coerce').dt.time
                print(f"Converted '{col}' to time")

        # Convert TotalPrice to numeric
        if "TotalPrice" in df.columns:
            df["TotalPrice"] = pd.to_numeric(
                df["TotalPrice"].apply(
                    lambda x: float(x) if not pd.isna(x) and str(x).replace('.', '').replace('-', '').isdigit() else 0
                )
            )
            print("Converted 'TotalPrice' to numeric")

        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop redundant columns.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with dropped columns
        """
        df = df.copy()

        cols_to_drop = [c for c in self.config.DROP_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns: {cols_to_drop}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        settings = self.config.FEATURE_ENGINEERING

        # Date features
        if settings["date_features"]:
            for col in self.config.DATE_COLUMNS:
                if col in df.columns:
                    df = DateFeatureEngineer.extract_features(df, col)

        # Time features
        if settings["time_features"]:
            for col in self.config.TIME_COLUMNS:
                if col in df.columns:
                    df = TimeFeatureEngineer.extract_features(df, col)

        # Interaction features
        if settings["interaction_features"]:
            df = InteractionFeatureEngineer.extract_features(df)

        return df

    def encode_strings(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode string columns.

        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            DataFrame with encoded columns
        """
        df = df.copy()

        # Get string columns that exist in dataframe
        string_cols = [c for c in self.config.STRING_COLUMNS if c in df.columns]

        if fit:
            self.encoder_manager.fit(df, string_cols)

        df = self.encoder_manager.transform(df, string_cols)

        return df

    def prepare_for_ltr(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare dataframe for LightGBM LTR.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (DataFrame ready for LTR, list of feature column names)
        """
        df = df.copy()

        # Drop raw datetime/time columns (keep engineered features only)
        raw_date_cols = [c for c in self.config.DATE_COLUMNS if c in df.columns]
        raw_time_cols = [c for c in self.config.TIME_COLUMNS if c in df.columns]
        drop_raw = raw_date_cols + raw_time_cols
        if drop_raw:
            print(f"Dropping raw date/time columns (keeping engineered features): {drop_raw}")
            df = df.drop(columns=drop_raw)

        # Also drop original string columns (keep encoded versions)
        drop_strings = [c for c in self.config.STRING_COLUMNS if c in df.columns]
        if drop_strings:
            print(f"Dropping original string columns (keeping encoded): {drop_strings}")
            df = df.drop(columns=drop_strings)

        # Get all feature columns (exclude non-features)
        exclude_cols = [
            self.config.QUERY_ID_COLUMN,
            self.config.TARGET_COLUMN,
            "CustomerID",
        ]

        # Get numeric and encoded columns as features
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']
        ]

        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        return df, feature_cols

    def create_group_sizes(self, df: pd.DataFrame) -> List[int]:
        """Create group sizes for LightGBM ranking.

        Args:
            df: Input dataframe

        Returns:
            List of group sizes (number of items per query)
        """
        group_sizes = df.groupby(self.config.QUERY_ID_COLUMN).size().tolist()
        return group_sizes


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline."""

    print("=" * 60)
    print("LightGBM LTR Preprocessing Pipeline")
    print("=" * 60)

    # Initialize config
    config = PreprocessingConfig()

    # Create directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n[STEP 1] Loading data...")
    df = preprocessor.load_data()

    # =========================================================================
    # STEP 2: Convert Types
    # =========================================================================
    print("\n[STEP 2] Converting column types...")
    df = preprocessor.convert_types(df)

    # =========================================================================
    # STEP 3: Drop Redundant Columns
    # =========================================================================
    print("\n[STEP 3] Dropping redundant columns...")
    df = preprocessor.drop_columns(df)

    # =========================================================================
    # STEP 4: Feature Engineering
    # =========================================================================
    print("\n[STEP 4] Engineering features...")
    df = preprocessor.engineer_features(df)

    # =========================================================================
    # STEP 5: Encode String Columns
    # =========================================================================
    print("\n[STEP 5] Encoding string columns...")
    df = preprocessor.encode_strings(df, fit=True)

    # Save encoders
    preprocessor.encoder_manager.save()

    # =========================================================================
    # STEP 6: Prepare Final Dataset
    # =========================================================================
    print("\n[STEP 6] Preparing final dataset...")
    df_final, feature_cols = preprocessor.prepare_for_ltr(df)

    print(f"\nFinal columns ({len(df_final.columns)}):")
    for col in df_final.columns:
        print(f"  - {col} ({df_final[col].dtype})")

    # =========================================================================
    # STEP 7: Create Train/Test Split
    # =========================================================================
    print("\n[STEP 7] Creating train/test split...")

    # Sort by query_id for proper grouping
    df_final = df_final.sort_values(config.QUERY_ID_COLUMN).reset_index(drop=True)

    # Get unique customers for splitting
    unique_customers = df_final[config.QUERY_ID_COLUMN].unique()
    train_customers, test_customers = train_test_split(
        unique_customers,
        test_size=0.2,
        random_state=42
    )

    train_mask = df_final[config.QUERY_ID_COLUMN].isin(train_customers)
    test_mask = df_final[config.QUERY_ID_COLUMN].isin(test_customers)

    df_train = df_final[train_mask].copy()
    df_test = df_final[test_mask].copy()

    print(f"Training set: {df_train.shape[0]} rows, {df_train[config.QUERY_ID_COLUMN].nunique()} unique customers")
    print(f"Test set: {df_test.shape[0]} rows, {df_test[config.QUERY_ID_COLUMN].nunique()} unique customers")

    # Create group sizes
    train_groups = preprocessor.create_group_sizes(df_train)
    test_groups = preprocessor.create_group_sizes(df_test)

    print(f"Training groups: {len(train_groups)}, Test groups: {len(test_groups)}")

    # =========================================================================
    # STEP 8: Save Processed Data
    # =========================================================================
    print("\n[STEP 8] Saving processed data...")

    # Save train/test data
    train_path = config.PROCESSED_DIR / "train.parquet"
    test_path = config.PROCESSED_DIR / "test.parquet"

    df_train.to_parquet(train_path, index=False, engine='pyarrow', compression='gzip')
    df_test.to_parquet(test_path, index=False, engine='pyarrow', compression='gzip')

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    # Save feature columns
    feature_cols_path = config.OUTPUT_DIR / "feature_columns.json"
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to: {feature_cols_path}")

    # Save group sizes
    groups_path = config.OUTPUT_DIR / "group_sizes.json"
    groups_data = {
        "train_groups": train_groups,
        "test_groups": test_groups,
    }
    with open(groups_path, 'w') as f:
        json.dump(groups_data, f)
    print(f"Group sizes saved to: {groups_path}")

    # =========================================================================
    # STEP 9: Training LightGBM Ranker (Optional)
    # =========================================================================
    print("\n[STEP 9] Training LightGBM Ranker (optional)...")

    # Prepare training data
    X_train = df_train[feature_cols].values
    y_train = df_train[config.TARGET_COLUMN].values

    X_test = df_test[feature_cols].values
    y_test = df_test[config.TARGET_COLUMN].values

    # Create dataset
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_groups,
        feature_name=feature_cols
    )

    test_data = lgb.Dataset(
        X_test,
        label=y_test,
        group=test_groups,
        reference=train_data
    )

    # Training parameters for LambdaRank
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    print("Training LightGBM LambdaRank model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50)
        ]
    )

    # Save model
    model_path = config.OUTPUT_DIR / "ltr_model.txt"
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")

    # Feature importance
    print("\nTop 10 Feature Importance:")
    importance = model.feature_importance(importance_type='gain')
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.2f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Train data: {train_path}")
    print(f"  - Test data: {test_path}")
    print(f"  - Encoders: {config.ENCODER_DIR / 'encoders.pkl'}")
    print(f"  - Model: {model_path}")
    print(f"  - Feature columns: {feature_cols_path}")
    print(f"  - Group sizes: {groups_path}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
    print(f"\nTotal features engineered: {len(feature_cols)}")
    print(f"Top-K recommendation: {config.TOP_K} items per customer")


if __name__ == "__main__":
    main()