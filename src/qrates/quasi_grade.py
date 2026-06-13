#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-06"


import gc
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Optional, Tuple, List


LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg
from db import duckdb_connection


def GenQuasi_Grade(data              : pd.DataFrame,
                   user_col          : str = "CustomerID",
                   item_col          : str = "CategoryID",
                   quantity_col      : str = "Quantity",
                   total_col         : str = "TotalPrice",
                   output_rating_col : str = "pseudo_rating",
                  ) -> pd.DataFrame:
    """
    Generate pseudo-ratings from implicit transaction data using DuckDB.
    Formula: ln(1 + sum(Quantity)) weighted by interaction frequency.
    """
    logger.debug("Starting pseudo-rating generation. Input shape: %s", data.shape)
    req     = {user_col, item_col, quantity_col, total_col}
    missing = req - set(data.columns)
    if missing:
        logger.error(f"Missing required columns for pseudo-rating: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    if (data[quantity_col] <= 0).any():
        logger.warning(f'''
        Found negative or zero values in '{quantity_col}'. 
        These will skew implicit feedback!''')
    if (data[total_col] <= 0).any():
        logger.warning(
        f"Found negative or zero values in '{total_col}'.")
    logger.debug("Registering raw data into DuckDB for aggregation...")

    # Query Compression of Transaction per user-item
    with duckdb_connection() as con:
        con.register_dataframe(name = "RAW_DATA", df = data)
        logger.debug('Formula Pseudo Rating - Log-scaling!')
        query = f'''
        SELECT 
            "{user_col}",
            "{item_col}",
            SUM("{quantity_col}") as total_quantity,
            COUNT(*) as transaction_count,
            SUM("{total_col}") as total_spend,
            LN(1 + SUM("{quantity_col}")) as "{output_rating_col}"
        FROM
            RAW_DATA
        GROUP BY
            "{user_col}", "{item_col}"
        ORDER BY
            "{user_col}" ASC, "{item_col}" DESC
        '''
        logger.debug(f"Executing aggregation: {query}")
        Aggs = con.query(query)
    logger.info(f"Pseudo-ratings Unique pairs: {Aggs.shape[0]}")
    logger.debug("Rating distribution ::"
        f"Min  : {Aggs[output_rating_col].min():.4f}, "
        f"Max  : {Aggs[output_rating_col].max():.4f}, "
        f"Mean : {Aggs[output_rating_col].mean():.4f}.")
    
    # Checking for Extreme Sparsity
    n_users  = Aggs[user_col].nunique()
    n_items  = Aggs[item_col].nunique()
    sparsity = 1.0 - (len(Aggs) / (n_users * n_items))
    if sparsity > 0.99:
        logger.warning("Extreme matrix sparsity detected"
        f": {sparsity * 100:.2f}%. Collaborative filtering might struggle.")
    return Aggs

def build_collaborative_features_v2(
    data: pd.DataFrame,
    user_col: str = "CustomerID",
    item_col: str = "CategoryID",
    rating_col: str = "pseudo_rating",
    user_feature_cols: Optional[List[str]] = None,
    item_feature_cols: Optional[List[str]] = None,
    weight_col: Optional[str] = "total_quantity",
    dtype: str = "float32"
) -> Tuple[sp.coo_matrix, sp.spmatrix, sp.spmatrix, sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Builds interaction matrix, user features, item features, and sample weights 
    using DuckDB and Scipy Sparse.
    
    Returns:
        interactions : sp.coo_matrix (n_users x n_items)
        user_features : sp.csr_matrix (n_users x n_user_features)
        item_features : sp.csr_matrix (n_items x n_item_features)
        sample_weight : sp.coo_matrix (n_users x n_items)
        user_ids      : np.ndarray (mapping index -> original user ID)
        item_ids      : np.ndarray (mapping index -> original item ID)
    """
    logger.info("Initializing build_collaborative_features_v2...")
    user_feature_cols = user_feature_cols or list()
    item_feature_cols = item_feature_cols or list()

    # 1. Enkodasi ID User & Item Menggunakan DuckDB
    logger.debug("Encoding user and item IDs with DENSE_RANK in DuckDB...")
    existing_columns = data.columns.tolist()
    if weight_col in existing_columns:
        weight_sql = '"total_quantity"'
    else:
        weight_sql = "1.0"
    with duckdb_connection() as con:
        con.register_dataframe("ENCODED_RAW", data)
        
        # Buat mapping index sequential dari 0
        encoded_df = con.query(f"""
            SELECT
                DENSE_RANK() OVER (ORDER BY "{user_col}") - 1 AS user_idx,
                DENSE_RANK() OVER (ORDER BY "{item_col}") - 1 AS item_idx,
                "{rating_col}",
                {weight_sql} AS weight_val,
                "{user_col}" AS user_id,
                "{item_col}" AS item_id
            FROM ENCODED_RAW
        """)
        con.register_dataframe("encoded_df", encoded_df)
        # Ambil list mapping unik untuk index NumPy array
        user_map = con.query("SELECT DISTINCT user_idx, user_id FROM encoded_df ORDER BY user_idx")
        item_map = con.query("SELECT DISTINCT item_idx, item_id FROM encoded_df ORDER BY item_idx")

    n_users = len(user_map)
    n_items = len(item_map)
    logger.info(f"Dimensions resolved: Unique Users = {n_users}, Unique Items = {n_items}")

    # 2. Bangun Matriks Interaksi & Sample Weights
    logger.debug("Building interaction and sample_weight matrices...")
    row_arr = encoded_df["user_idx"].values.astype(np.int32)
    col_arr = encoded_df["item_idx"].values.astype(np.int32)
    rating_arr = encoded_df[rating_col].values.astype(dtype)
    weight_arr = encoded_df["weight_val"].values.astype(dtype)

    interactions = sp.coo_matrix((rating_arr, (row_arr, col_arr)), shape=(n_users, n_items), dtype=dtype)
    sample_weight = sp.coo_matrix((weight_arr, (row_arr, col_arr)), shape=(n_users, n_items), dtype=dtype)

    # 3. Bangun User Features (CSR Matrix)
    logger.info("Processing User Features...")
    if user_feature_cols:
        # Cari representasi unik fitur per user. Jika user punya multi-value, ambil nilai modus/terakhir
        with duckdb_connection() as con:
            con.register_dataframe("RAW_CONN", data)
            con.register_dataframe("USER_MAP_CONN", user_map)
            select_feats = ", ".join([f'MAX("{col}") AS "{col}"' for col in user_feature_cols])
            
            user_features_df = con.query(f"""
                SELECT m.user_idx, {select_feats}
                FROM USER_MAP_CONN m
                JOIN RAW_CONN r ON m.user_id = r."{user_col}"
                GROUP BY m.user_idx
                ORDER BY m.user_idx
            """)
        
        # One-hot encoding untuk fitur kategori
        logger.debug("One-hot encoding user features...")
        user_encoded_feats = pd.get_dummies(user_features_df[user_feature_cols], dtype=np.float32)
        # Sifat hybrid CF mewajibkan identity matrix ditambahkan ke dalam fitur asli
        user_features_sparse = sp.hstack([sp.eye(n_users, dtype=dtype).tocsr(), sp.csr_matrix(user_encoded_feats.values)])
    else:
        logger.warning("No user features provided. Defaulting to Identity Matrix.")
        user_features_sparse = sp.eye(n_users, dtype=dtype).tocsr()

    # 4. Bangun Item Features (CSR Matrix)
    logger.info("Processing Item Features...")
    if item_feature_cols:
        with duckdb_connection() as con:
            con.register_dataframe("RAW_CONN", data)
            con.register_dataframe("ITEM_MAP_CONN", item_map)
            select_feats = ", ".join([f'MAX("{col}") AS "{col}"' for col in item_feature_cols])
            
            item_features_df = con.query(f"""
                SELECT m.item_idx, {select_feats}
                FROM ITEM_MAP_CONN m
                JOIN RAW_CONN r ON m.item_id = r."{item_col}"
                GROUP BY m.item_idx
                ORDER BY m.item_idx
            """)
            
        logger.debug("One-hot encoding item features...")
        item_encoded_feats = pd.get_dummies(item_features_df[item_feature_cols], dtype=np.float32)
        item_features_sparse = sp.hstack([sp.eye(n_items, dtype=dtype).tocsr(), sp.csr_matrix(item_encoded_feats.values)])
    else:
        logger.warning("No item features provided. Defaulting to Identity Matrix.")
        item_features_sparse = sp.eye(n_items, dtype=dtype).tocsr()

    # Ekstraksi array mapping final
    user_ids = user_map["user_id"].values
    item_ids = item_map["item_id"].values

    # Cleanup Memory
    logger.debug("Garbage collecting dynamic structures inside pipeline...")
    del encoded_df, row_arr, col_arr, rating_arr, weight_arr
    gc.collect()

    logger.info("Pipeline features completed successfully.")
    return interactions, user_features_sparse, item_features_sparse, sample_weight, user_ids, item_ids


if __name__ == "__main__":
    # 1. Load Data Contoh Anda (Simulasi data input)
    raw_data = pd.DataFrame({
        'SalesID': [843861, 3633024, 458339, 5845137],
        'CustomerID': [66045, 1740, 44708, 81783],
        'CategoryID': [4, 2, 11, 1],
        'Quantity': [17, 1, 12, 21],
        'TotalPrice': [95.096, 9.969, 902.471, 1093.53],
        'CityName': ['Cleveland', 'Aurora', 'Minneapolis', 'San Jose'], # Kandidat User Feature
        'Class': ['High', 'High', 'Medium', 'Medium']                  # Kandidat Item Feature
    })
    
    # STEP 1: Buat Pseudo Rating
    pseudo_rated_df = GenQuasi_Grade(
        data=raw_data,
        user_col="CustomerID",
        item_col="CategoryID",
        quantity_col="Quantity",
        total_col="TotalPrice",
        output_rating_col="pseudo_rating"
    )
    
    # STEP 2: Ekstrak ke tipe data Scipy Sparse untuk model Rekomendasi
    # Kita masukkan 'CityName' sebagai fitur user, dan 'Class' sebagai fitur item
    interactions, user_features, item_features, sample_weight, users, items = build_collaborative_features_v2(
        data=pseudo_rated_df.merge(raw_data[['CustomerID', 'CategoryID', 'CityName', 'Class']], on=['CustomerID', 'CategoryID']),
        user_col="CustomerID",
        item_col="CategoryID",
        rating_col="pseudo_rating",
        user_feature_cols=["CityName"],
        item_feature_cols=["Class"],
        weight_col="total_quantity"
    )
    
    print("\n--- HASIL AKHIR MATRIKS ---")
    print(f"Interactions Shape : {interactions.shape} | Tipe: {type(interactions)}")
    print(f"User Features Shape: {user_features.shape} | Tipe: {type(user_features)}")
    print(f"Item Features Shape: {item_features.shape} | Tipe: {type(item_features)}")
    print(f"Sample Weight Shape: {sample_weight.shape} | Tipe: {type(sample_weight)}")