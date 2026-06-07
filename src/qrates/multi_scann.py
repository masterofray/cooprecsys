#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-07"

import sys
import scann
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import _cfg, logger


class MultiGroupScaNN:
    """
    A scalable nearest neighbor search system utilizing ScaNN for parallel 
    Quasi-Rating extraction based on heterogeneous feature groupings.
    """
    def __init__(self, 
                 feature_groups: Dict[str, List[str]], 
                 scann_config  : Dict[str, Any] = None,
                ):
        """
        Initializes the index architecture.
        feature_groups: A dictionary where keys are group names and values are lists 
                        of feature column names.
                        Example: {"financial": ["price", "discount"], 
                                  "demographic": ["age"]}
        scann_config  : Custom hyperparameters for ScaNN (optional).
        """
        self.feature_groups = feature_groups
        self.group_names    = list(feature_groups.keys())
        self.scann_config = scann_config or {
            "num_leaves_ratio"     : 0.15,
            "num_leaves_to_search" : 10,
            "anisotropic_quantization_threshold": 0.2}
        self.scalers   : Dict[str, StandardScaler] = dict()
        self.searchers : Dict[str, Any]            = dict()
        self._is_fitted    = False
        self._reference_df = None


    def _l2_normalize(self, 
        vectors: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to vectors to ensure 
        Dot Product functions as Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
        norms[norms == 0] = 1e-10
        return vectors / norms


    def fit(self, Data: pd.DataFrame) -> 'MultiGroupScaNN':
        """Constructs isolated ScaNN indices for each feature group."""
        logger.debug(f"Starting index construction for "
                     f"{len(self.group_names)} feature groups.")
        require = [feat for group in self.feature_groups.values() for feat in group]
        miss    = [col for col in require if col not in Data.columns]
        if miss:
            logger.error(f"The following columns are missing from the dataset: {miss}")
            raise ValueError()
        self._reference_df = Data.reset_index(drop = True)
        num_rows           = len(self._reference_df)
        if num_rows == 0:
            logger.error("The input DataFrame is empty."
                         "Cannot build an index with zero records.")
            raise ValueError()

        # Dynamically adjust num_leaves if the dataset 
        # is too small to prevent ScaNN crashes
        num_leaves = max(10, int(num_rows * 
                         self.scann_config["num_leaves_ratio"]))
        if num_rows < num_leaves:
            num_leaves = max(1, num_rows//2)
            logger.warning(
            f"Dataset size ({num_rows}) is smaller than configured "
            f"num_leaves. Adjusted num_leaves to {num_leaves}.")
        for group_name, features in tqdm(
                self.feature_groups.items(), 
                desc        = "Building ScaNN Indices",
                colour      = _cfg.get('tqdm', 'colour'),
                ncols       = _cfg.getint('tqdm', 'ncols'),
                bar_format  = _cfg.get('tqdm', 'BarFormats'),
                unit        = 'Group',
                mininterval = 0.1):
            logger.info(f"Processing group: '{group_name}' | Features: {features}")
            total_dims = len(features)
            if total_dims == 0:
                raise ValueError(f"Feature group '{group_name}' has no features.")
                
            # Feature extraction and standardization with robust NaN handling
            raw_df = self._reference_df[features].copy()
            if raw_df.isnull().values.any():
                logger.warning(f"NaN values detected in group '{group_name}'. Filling with column medians.")
                raw_df = raw_df.fillna(raw_df.median())
            raw_data = raw_df.values.astype(np.float32)
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(raw_data)
            self.scalers[group_name] = scaler
            
            # L2 normalization
            normalized_data = self._l2_normalize(scaled_data)
            
            # Construct ScaNN Index
            # CRITICAL FIX: dimensions_per_block must evenly divide the total number of dimensions
            dim_block = 2 if (total_dims >= 2 and total_dims % 2 == 0) else 1
            
            searcher = scann.scann_ops_pybind.builder(
                normalized_data, num_neighbors=10, distance_measure="dot_product"
            ).tree(
                num_leaves=num_leaves,
                num_leaves_to_search=self.scann_config["num_leaves_to_search"],
                training_sample_size=min(num_rows, 100000) # Cap training sample to optimize memory and build time
            ).score_ah(
                dimensions_per_block=dim_block, 
                anisotropic_quantization_threshold=self.scann_config["anisotropic_quantization_threshold"]
            ).reorder(100).build()
            
            self.searchers[group_name] = searcher
            
        self._is_fitted = True
        logger.info("Index construction completed successfully.")
        return self

    def search(self, query_dict: Dict[str, float], k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Executes queries against all existing group indices and returns Quasi-Ratings.
        """
        if not self._is_fitted:
            raise RuntimeError("The model has not been fitted yet. Please invoke the .fit(Data) method first.")

        results = dict()
        
        # Iterate over groups with a progress bar
        for group_name, features in tqdm(
                self.feature_groups.items(), 
                desc        = 'Executing Search Queries',
                colour      = _cfg.get('tqdm', 'colour'),
                ncols       = _cfg.getint('tqdm', 'ncols'),
                bar_format  = _cfg.get('tqdm', 'BarFormats'),
                unit        = 'Group',
                mininterval = 0.1):
            # Validate query input
            missing_query_keys = [f for f in features if f not in query_dict]
            if missing_query_keys:
                logger.warning(f"Query for group '{group_name}' is skipped. Missing keys: {missing_query_keys}")
                continue

            # Construct query vector
            q_vector = np.array([[query_dict[f] for f in features]], dtype=np.float32)
            
            # Transformation (Scaling + L2 Normalization)
            q_scaled = self.scalers[group_name].transform(q_vector)
            q_scaled = np.nan_to_num(q_scaled, nan=0.0) # Robustly handle potential NaNs in query
            q_norm = self._l2_normalize(q_scaled)
            
            # Execute low-latency search
            neighbors, distances = self.searchers[group_name].search(q_norm[0], final_num_neighbors=k)
            
            # Format output
            res_df = self._reference_df.iloc[neighbors].copy()
            res_df[f'Quasi_Rating_{group_name}'] = distances
            results[group_name] = res_df
            
        return results

    def search_batch(self, queries_df: pd.DataFrame, k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Executes batched queries against all existing group indices for optimal throughput.
        Highly recommended for Quasi-Rating generation involving large user/item populations.
        
        Args:
            queries_df: A DataFrame where each row represents a query containing feature values.
            k: The number of nearest neighbors to retrieve.
            
        Returns:
            A dictionary mapping group names to DataFrames containing the batched search results.
        """
        if not self._is_fitted:
            raise RuntimeError("The model has not been fitted yet. Please invoke the .fit(Data) method first.")

        batch_results = dict()
        
        # Iterate over groups with a progress bar
        for group_name, features in tqdm(
                self.feature_groups.items(), 
                desc        = 'Executing Batched Search',
                colour      = _cfg.get('tqdm', 'colour'),
                ncols       = _cfg.getint('tqdm', 'ncols'),
                bar_format  = _cfg.get('tqdm', 'BarFormats'),
                unit        = 'Group',
                mininterval = 0.1):
            # Validate query input
            missing_query_keys = [f for f in features if f not in queries_df.columns]
            if missing_query_keys:
                logger.warning(f"Batch query for group '{group_name}' is skipped. Missing columns: {missing_query_keys}")
                continue

            # Construct batch query vectors
            q_vectors = queries_df[features].values.astype(np.float32)
            
            # Transformation (Scaling + L2 Normalization)
            q_scaled = self.scalers[group_name].transform(q_vectors)
            q_scaled = np.nan_to_num(q_scaled, nan=0.0) # Robustly handle potential NaNs
            q_norm = self._l2_normalize(q_scaled)
            
            # Execute batched low-latency search (Vectorized operation)
            neighbors, distances = self.searchers[group_name].search_batched(q_norm, final_num_neighbors=k)
            
            # Format output
            retrieved_items = list()
            for i in range(len(queries_df)):
                row_neighbors = neighbors[i]
                row_distances = distances[i]
                
                res_df = self._reference_df.iloc[row_neighbors].copy()
                res_df[f'Quasi_Rating_{group_name}'] = row_distances
                res_df['query_index'] = i
                retrieved_items.append(res_df)
                
            if retrieved_items:
                batch_results[group_name] = pd.concat(retrieved_items, ignore_index=True)
            else:
                batch_results[group_name] = pd.DataFrame()
            
        return batch_results


if __name__ == '__main__':
    datapath = LocDir / 'scr' / 'sampledata.parquet'
    df = pd.read_parquet(datapath)
    # Convert necessary columns to numeric types for ScaNN processing
    numeric_cols = ['SalesID', 'CustomerID', 'ProductPrice', 'Quantity', 'Discount', 
                    'TotalPrice', 'CategoryID', 'VitalityDays', 'EmployeeID', 
                    'EmployeeAge', 'YearsWorking']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = pd.concat([df] * 5, ignore_index=True)
    print(f"Dataset loaded and expanded to {len(df)} rows for ScaNN compatibility.\n")

    # 2. Define heterogeneous feature groups for Quasi-Rating extraction
    feature_groups = {
        "transaction_metrics": ["ProductPrice", "Quantity", "Discount", "TotalPrice"],
        "product_traits": ["CategoryID", "VitalityDays"],
        "employee_demographics": ["EmployeeAge", "YearsWorking"]
    }

    # 3. Initialize and fit the MultiGroupScaNN model
    model = MultiGroupScaNN(feature_groups=feature_groups)
    model.fit(df)

    # 4. Execute a single search query
    print("\n--- Executing Single Search Query ---")
    query = {
        "ProductPrice": 50.0,
        "Quantity": 15,
        "Discount": 0.1,
        "TotalPrice": 700.0,
        "CategoryID": 4,
        "VitalityDays": 60,
        "EmployeeAge": 50,
        "YearsWorking": 12
    }
    
    single_results = model.search(query, k=3)
    for group_name, res_df in single_results.items():
        print(f"\nTop 3 matches for '{group_name}':")
        display_cols = ['ProductName', 'EmployeeFirstName', f'Quasi_Rating_{group_name}']
        print(res_df[display_cols].head(3))

    # 5. Execute a batched search query (Optimized for high throughput)
    print("\n\n--- Executing Batched Search Queries ---")
    # Use the first 5 rows of our numeric features as batch queries
    batch_queries = df[numeric_cols].head(5)
    
    batch_results = model.search_batch(batch_queries, k=2)
    for group_name, res_df in batch_results.items():
        print(f"\nBatch results for '{group_name}' (showing first 5 retrieved items):")
        display_cols = ['query_index', 'ProductName', f'Quasi_Rating_{group_name}']
        print(res_df[display_cols].head(5))