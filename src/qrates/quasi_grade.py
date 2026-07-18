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
from typing  import Optional, Tuple, List

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg
from db      import duckdb_connection
from prepare import DetectReco_Identifier


def GenQuasi_Lazy(data : pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data, pd.DataFrame), 'This is not a dataframe.'
    assert not data.empty, 'The data is empty'
    assert data.shape[0] >= 20, 'The data lenght is too small.'
    ColmCollect   = DetectReco_Identifier(data)
    NoneKeys      = [key for key, value in ColmCollect.items() if value is None]
    logger.info(f'There are {len(NoneKeys)} keys that already Null.')
    Quasi         = GenQuasi_Grade(
                    data              = data,
                    user_col          = ColmCollect['user_col'],
                    item_col          = ColmCollect['item_col'],
                    quantity_col      = ColmCollect['quantity_col'],
                    total_col         = ColmCollect['total_col'])
    return Quasi


def GenQuasi_Grade(data              : pd.DataFrame,
                   user_col          : str,
                   item_col          : str,
                   quantity_col      : str,
                   total_col         : str,
                   output_rating_col : str = None,
                  ) -> pd.DataFrame:
    """
    Generate pseudo-ratings from implicit transaction data using DuckDB.
    Formula: ln(1 + sum(Quantity)) weighted by interaction frequency.
    """
    logger.debug("Starting pseudo-rating generation. Input shape: %s", data.shape)
    if output_rating_col is None:
        output_rating_col = _cfg.get('RATING', 'ColumnName')
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
        logger.warning(f"Found negative or zero values in '{total_col}'.")
    logger.debug("Registering raw data into DuckDB for aggregation.")

    # Query Compression of Transaction per user-item
    with duckdb_connection() as con:
        con.register_dataframe(name = "RAW_DATA", df = data)
        logger.debug('Formula Pseudo Rating - Log-scaling!')
        query = f'''
        SELECT 
            {user_col},
            {item_col},
            SUM("{quantity_col}") AS total_quantity,
            COUNT(*) AS transaction_count,
            SUM({total_col}::DOUBLE) AS total_spend,
            LN(1 + SUM({quantity_col})) AS {output_rating_col}
        FROM
            RAW_DATA
        GROUP BY
            {user_col}, {item_col}
        ORDER BY
            {user_col} ASC, {item_col} DESC
        '''
        #logger.debug(f"Executing aggregation: {query}")
        Aggs = con.query(query)

    # Checking for Extreme Sparsity
    n_users  = Aggs[user_col].nunique()
    n_items  = Aggs[item_col].nunique()
    sparsity = 1.0 - (len(Aggs) / (n_users * n_items))
    if sparsity > 0.99:
        logger.warning("Extreme matrix sparsity detected"
        f": {sparsity * 100:.2f}%. Collaborative filtering might struggle.")
    logger.info(f"Pseudo-ratings Unique pairs: {Aggs.shape[0]}")
    logger.debug("Rating distribution ::"
        f"Min  : {Aggs[output_rating_col].min():.4f}, "
        f"Max  : {Aggs[output_rating_col].max():.4f}, "
        f"Mean : {Aggs[output_rating_col].mean():.4f}.")
    return Aggs


def Decomposition_Matrix_Dev(
        data              : pd.DataFrame,
        user_col          : str,
        item_col          : str,
        rating_col        : str                 = None,
        user_feature_cols : Optional[List[str]] = None,
        item_feature_cols : Optional[List[str]] = None,
        weight_col        : Optional[str]       = "total_quantity",
        dtype             : str                 = "float32"
    ) -> Tuple[sp.coo_matrix, sp.spmatrix, sp.spmatrix, 
               sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Builds interaction matrix, user features, item features, and sample weights 
    using DuckDB and Scipy Sparse. It will return:
    - interactions  : sp.coo_matrix (n_users x n_items)
    - user_features : sp.csr_matrix (n_users x n_user_features)
    - item_features : sp.csr_matrix (n_items x n_item_features)
    - sample_weight : sp.coo_matrix (n_users x n_items)
    - user_ids      : np.ndarray (mapping index -> original user ID)
    - item_ids      : np.ndarray (mapping index -> original item ID)
    """
    logger.info("Initializing parameter to build Interaction Matrix")
    user_feature_cols = user_feature_cols or list()
    item_feature_cols = item_feature_cols or list()
    if rating_col is None:
        rating_col    = _cfg.get('RATING', 'ColumnName')

    # 1. Enkodasi ID User & Item Menggunakan DuckDB
    logger.debug("Encoding user and item IDs with DENSE_RANK in DuckDB.")
    existing_columns = data.columns.tolist()
    if weight_col in existing_columns:
        weight_sql   = '"total_quantity"'
    else:
        weight_sql   = "1.0"
    with duckdb_connection() as con:
        con.register_dataframe("ENCODED_RAW", data)
        
        # Buat mapping index sequential dari 0
        DataMaps = con.query(f"""
            SELECT
                DENSE_RANK() OVER (ORDER BY "{user_col}") - 1 AS user_idx,
                DENSE_RANK() OVER (ORDER BY "{item_col}") - 1 AS item_idx,
                "{rating_col}",
                {weight_sql} AS weight_val,
                "{user_col}" AS user_id,
                "{item_col}" AS item_id
            FROM
                ENCODED_RAW;""")
        con.register_dataframe("DataMaps", DataMaps)
        user_map = con.query("SELECT DISTINCT user_idx, user_id FROM DataMaps ORDER BY user_idx")
        item_map = con.query("SELECT DISTINCT item_idx, item_id FROM DataMaps ORDER BY item_idx")
    n_users = len(user_map)
    n_items = len(item_map)
    logger.info(f"Dimensions resolved: Unique Users = {n_users}, Unique Items = {n_items}")

    # 2. Bangun Matriks Interaksi & Sample Weights
    logger.debug("Building interaction and sample_weight matrices.")
    row_arr       = DataMaps["user_idx"].values.astype(np.int32)
    col_arr       = DataMaps["item_idx"].values.astype(np.int32)
    rating_arr    = DataMaps[rating_col].values.astype(dtype)
    weight_arr    = DataMaps["weight_val"].values.astype(dtype)
    interactions  = sp.coo_matrix((rating_arr, (row_arr, col_arr)), 
                                   shape = (n_users, n_items), dtype = dtype)
    sample_weight = sp.coo_matrix((weight_arr, (row_arr, col_arr)), 
                                   shape = (n_users, n_items), dtype = dtype)

    # 3. Bangun User Features (CSR Matrix)
    if user_feature_cols:
        logger.debug("Processing User Features.")
        # Cari representasi unik fitur per user.
        # Jika user punya multi-value, ambil nilai modus/terakhir
        with duckdb_connection() as con:
            con.register_dataframe("RAW_CONN", data)
            con.register_dataframe("USER_MAP_CONN", user_map)
            select_feats = ", ".join([f'MAX("{col}") AS "{col}"' for col in user_feature_cols])
            UserFeats    = con.query(f"""
            SELECT
                m.user_idx, 
                {select_feats}
            FROM
                USER_MAP_CONN AS m
            JOIN
                RAW_CONN AS r
            ON
                m.user_id = r."{user_col}"
            GROUP BY
                m.user_idx
            ORDER BY
                m.user_idx;""")
        logger.debug("One-hot encoding user features.")
        user_encoded_feats   = pd.get_dummies(UserFeats[user_feature_cols], dtype = np.float32)
        # Sifat hybrid CF mewajibkan identity matrix ditambahkan ke dalam fitur asli
        user_features_sparse = sp.hstack([sp.eye(n_users, dtype = dtype
                               ).tocsr(), sp.csr_matrix(user_encoded_feats.values)])
    else:
        logger.warning("No user features provided. Defaulting to Identity Matrix.")
        user_features_sparse = sp.eye(n_users, dtype = dtype).tocsr()

    # 4. Bangun Item Features (CSR Matrix)
    if item_feature_cols:
        logger.info("Processing Item Features.")
        with duckdb_connection() as con:
            con.register_dataframe("RAW_CONN", data)
            con.register_dataframe("ITEM_MAP_CONN", item_map)
            select_feats = ", ".join([f'MAX("{col}") AS "{col}"' for col in item_feature_cols])
            ItemFeats    = con.query(f"""
            SELECT
                m.item_idx,
                {select_feats}
            FROM
                ITEM_MAP_CONN AS m
            JOIN
                RAW_CONN AS r
            ON
                m.item_id = r."{item_col}"
            GROUP BY
                m.item_idx
            ORDER BY
                m.item_idx;""")
        logger.debug("One-hot encoding item features.")
        item_encoded_feats   = pd.get_dummies(ItemFeats[item_feature_cols], dtype=np.float32)
        item_features_sparse = sp.hstack([sp.eye(n_items, dtype = dtype
                               ).tocsr(), sp.csr_matrix(item_encoded_feats.values)])
    else:
        logger.warning("No item features provided. Defaulting to Identity Matrix.")
        item_features_sparse = sp.eye(n_items, dtype = dtype).tocsr()

    user_ids  = user_map["user_id"].values
    item_ids  = item_map["item_id"].values
    logger.debug("Garbage collecting dynamic structures inside pipeline.")
    del DataMaps, row_arr, col_arr, rating_arr, weight_arr
    gc.collect()
    logger.info("Pipeline features completed successfully.")
    TheResult = (interactions,
                 user_features_sparse, 
                 item_features_sparse,
                 sample_weight,
                 user_ids,
                 item_ids)
    return TheResult


if __name__ == "__main__":
    pathdf     = LocDir.parent / 'data' / 'sampledata.parquet'
    print(pathdf)
    assert pathdf.exists(), 'data is not exist.'
    data          = pd.read_parquet(str(pathdf))
    DataGrade     = GenQuasi_Lazy(data)
    Collect       = DetectReco_Identifier(data.columns.to_numpy())
    MergeData     = DataGrade.merge(data,
                    on = [Collect['user_col'], Collect['item_col']])
    UserFeats     = ['EmployeeAge', 'EmployeeGender','Resistant', 'IsAllergic', 'VitalityDays']
    ItemFeats     = ['ProductPrice', 'Quantity', 'Discount', 'TotalPrice', 'Class']
    Results       = Decomposition_Matrix_Dev(
                    data              = MergeData,
                    user_col          = Collect['user_col'],
                    item_col          = Collect['item_col'],
                    user_feature_cols = UserFeats,
                    item_feature_cols = ItemFeats,)
    interactions  = Results[0]
    user_features = Results[1]
    item_features = Results[2]
    logger.debug(f"Interactions Shape : {interactions.shape}  | Tipe: {type(interactions)}")
    logger.debug(f"User Features Shape: {user_features.shape} | Tipe: {type(user_features)}")
    logger.debug(f"Item Features Shape: {item_features.shape} | Tipe: {type(item_features)}")
