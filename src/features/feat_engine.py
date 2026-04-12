'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

from load import load_data
import pandas as pd
import numpy as np
import duckdb as dc
import polars as pl
import argparse

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(data)}")

        # --- Robustness: Normalize Headers ---
        # Strips whitespace and lowers casing to handle " User_ID " or "product_id"
        self.Data = data.copy()
        self.Data.columns = [col.strip().lower() for col in self.Data.columns]
        required_cols     = ['user_id', 'product_id', 'order_number', 'reordered']
        missing           = [col for col in required_cols if col not in self.Data.columns]
        if missing:
            raise KeyError(f"Critical columns missing: {missing}")

        self.user_stats    = None
        self.product_stats = None
        self.df_feat       = None
        print(f"--- FeatureEngineer Initialized: {len(self.Data)} rows | {len(self.Data.columns)} cols ---")

    def create_user_features(self):
        print("Tracing: Calculating user features...")
        agg_map = {
            'order_number'           : 'max',
            'days_since_prior_order' : 'mean',
            'add_to_cart_order'      : 'mean',
            'reordered'              : 'sum',
        }
        # Robustness: Only aggregate columns that actually exist
        available_aggs = {k: v for k, v in agg_map.items() if k in self.Data.columns}

        try:
            self.user_stats = self.Data.groupby('user_id').agg(available_aggs)
            rename_map = {'order_number'          : 'user_total_orders',
                          'days_since_prior_order': 'user_avg_days_between',
                          'add_to_cart_order'     : 'user_avg_cart_pos',
                          'reordered'             : 'user_total_reorders'}
            self.user_stats.rename(columns=rename_map, inplace=True)
            if self.user_stats.empty:
                raise ValueError("Grouping resulted in empty user stats.")
        except Exception as e:
            print(f"Error in user features: {e}")
            raise

    def create_product_features(self):
        print("Tracing: Calculating product features...")
        try:
            # Multi-level index handling is safer with reset_index immediately
            stats = self.Data.groupby('product_id').agg({
                'reordered': ['sum', 'mean'],
                'order_id': 'count' if 'order_id' in self.Data.columns else 'size',
                'add_to_cart_order': 'mean'
            })

            # Flattening multi-index columns robustly
            stats.columns = ['prod_total_reorders', 'prod_reorder_rate', 'prod_order_count', 'prod_avg_cart_pos']
            self.product_stats = stats.reset_index()

            # Analytics: Log data quality issues
            nan_count = self.product_stats['prod_reorder_rate'].isnull().sum()
            if nan_count > 0:
                print(f"Analytics Warning: {nan_count} products have NaN reorder rates (likely single-order products).")

        except Exception as e:
            print(f"Error in product features: {e}")
            raise

    def merge_features_pandas(self):
        self.df_feat = self.Data.merge(self.user_stats, on='user_id', how='left')
        self.df_feat = self.df_feat.merge(self.product_stats, on='product_id', how='left')

    def merge_features_duckdb(self):
        con = dc.connect(database=':memory:')
        self_Data = self.Data
        self_user_stats = self.user_stats.reset_index()
        self_product_stats = self.product_stats.reset_index()

        # We use a SQL JOIN which is often more memory-efficient
        Dquery = """
        SELECT
            d.*,
            u.* EXCLUDE (user_id),
            p.* EXCLUDE (product_id)
        FROM self_Data as d
        LEFT JOIN self_user_stats as u ON d.user_id = u.user_id
        LEFT JOIN self_product_stats as p ON d.product_id = p.product_id
        """
        self.df_feat = con.execute(Dquery).df()

    def merge_features_polars(self):
        ldf = pl.from_pandas(self.Data)
        u_stats = pl.from_pandas(self.user_stats)
        p_stats = pl.from_pandas(self.product_stats)

        # Polars allows 'validate' to prevent row explosion automatically
        try:
            self.df_feat = (
                ldf.join(u_stats, on="user_id", how="left", validate="m:1")
                   .join(p_stats, on="product_id", how="left", validate="m:1")
            ).to_pandas()
        except Exception as e:
            print(f"Merge failed validation: {e}")

    def merge_features(self):
        if self.user_stats is None or self.product_stats is None:
            raise ValueError("Feature calculation steps must be run before merging.")

        # Analytics: Track shape before merge
        original_count = len(self.Data)
        if original_count <= 50_000:
            self.merge_features_pandas()
        elif 50_000 < original_count <= 700_000:
            self.merge_features_polars()
        else:
            self.merge_features_duckdb()

        # Robustness: Check for Cartesian product issues (row explosion)
        if len(self.df_feat) > original_count:
            print(f"CRITICAL WARNING: Row count increased from {original_count} to {len(self.df_feat)}. Check for duplicate IDs.")

    def fit_transform(self):
        try:
            self.create_user_features()
            self.create_product_features()
            self.merge_features()

            # Analytics: Dynamic feature selection (only keep what we actually have)
            potential_features = [
                'order_number', 'order_dow', 'order_hour_of_day',
                'days_since_prior_order', 'add_to_cart_order',
                'user_total_orders', 'user_avg_days_between', 'user_avg_cart_pos',
                'user_total_reorders', 'prod_total_reorders', 'prod_reorder_rate',
                'prod_order_count', 'prod_avg_cart_pos'
            ]
            self.feature_cols = [c for c in potential_features if c in self.df_feat.columns]

            # Robustness: Fast NaN filling using a median map
            numeric_feats = self.df_feat[self.feature_cols].select_dtypes(include=[np.number]).columns

            # This avoids the SettingWithCopyWarning and is faster than a loop
            medians = self.df_feat[numeric_feats].median()
            self.df_feat.fillna(medians, inplace=True)

            print(f"--- fit_transform Completed: {len(numeric_feats)} features generated ---")
            return self.df_feat, numeric_feats, self.feature_cols

        except Exception as e:
            print(f"CRITICAL ERROR in pipeline: {e}")
            return None, None


import argparse
import pandas as pd
import duckdb
import os

def load_production_data(file_path):
    """
    Detects file extension and returns a pandas DataFrame.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.db':
        # Assumes the table name in DuckDB is 'data' or similar
        # Adjust the query as needed for your specific DB schema
        conn = duckdb.connect(file_path)
        df = conn.execute("SELECT * FROM train_table").df()
        conn.close()
        return df
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Feature Engineering Script")
    
    # Adding the --data argument
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to the input data file (.csv, .parquet, or .db)"
    )
    
    args = parser.parse_args()

    # 2. Load the data into FTrain
    try:
        FTrain = load_production_data(args.data)
        print(f"Successfully loaded data from {args.data} with shape {FTrain.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Your existing Logic
    # FEs = FeatureEngineer(data = FTrain)
    # DataFeatureTrain, feature_cols, others = FEs.fit_transform()
    print("Feature engineering complete.")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Feature Engineering Pipeline for Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feat_engine.py --data /path/to/data.csv
  python feat_engine.py --data /path/to/data.parquet
  python feat_engine.py --data /path/to/database.db
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file (supported formats: .csv, .parquet, .db for DuckDB)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    try:
        # Load data based on the provided path
        FTrain = load_data(args.data)
        
        # Initialize and run feature engineering
        FEs = FeatureEngineer(data=FTrain)
        DataFeatureTrain, feature_cols, others = FEs.fit_transform()
        
        print("\nFeature Engineering Completed Successfully!")
        print(f"Transformed data shape: {DataFeatureTrain.shape}")
        print(f"Feature columns: {feature_cols}")
        print(f"Other columns: {others}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)