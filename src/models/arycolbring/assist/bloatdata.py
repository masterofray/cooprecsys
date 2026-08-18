#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"

import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib   import Path
from tqdm.auto import tqdm
from typing    import Optional, Tuple

from ....db       import duckdb_connection
from ....configs  import _cfg, logger
from ....features import load_data

DType = _cfg.get("model", "dtype", fallback = "float32")


def norm_exchange(data       : pd.DataFrame,
                  user_col   : str = "user_id",
                  item_col   : str = "item_id",
                  rating_col : Optional[str] = None,
                  dtype      : str = DType,
                 ) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Convert a Pandas DataFrame of (user, item, and/without rating) rows into a
    sparse COO interaction matrix.  Uses DuckDB for the encoding query
    so it stays efficient even for tens of millions of rows. This function
    only bound by those 2 or 3 items.
    _______________________________________________________________
    Parameters
    data       : DataFrame with at least ``user_col`` and ``item_col`` columns.
    user_col   : Name of the user identifier column.
    item_col   : Name of the item identifier column.
    rating_col : Name of an optional rating column.
                 If None, all interactions are set to 1.0.
    dtype      : NumPy dtype string for the sparse matrix data array.
    _______________________________________________________________
    Returns
    interactions : scipy.sparse.coo_matrix  (n_users × n_items)
    user_ids     : np.ndarray  — mapping from integer index ->> original user id
    item_ids     : np.ndarray  — mapping from integer index ->> original item id
    """
    required = {user_col, item_col}
    logger.info("norm_exchange: shape = %s user_col = %s "\
                "item_col = %s rating_col = %s",
                data.shape, user_col, item_col, rating_col)
    if rating_col:
        required.add(rating_col)
    missing = required - set(data.columns)
    if missing:
        logger.error(f"DataFrame is missing columns: {missing}")
        raise ValueError()

    with tqdm(total       = 6, 
              desc        = "Building interaction matrix",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:

        # Step 1. register data in DuckDB and encode IDs
        with duckdb_connection() as con:
            pbar.set_postfix_str("registering dataframe")
            con.register_dataframe("RAW", data)
            pbar.update(1)
            
            rating_expr = (f'CAST("{rating_col}" AS DOUBLE)'
                           if rating_col else "1.0")
            pbar.set_postfix_str("creating isolated ID mapping tables")
            con.execute(f"""
            CREATE OR REPLACE TABLE ENCODED AS
            SELECT
                DENSE_RANK() OVER (ORDER BY "{user_col}") - 1  AS user_idx,
                DENSE_RANK() OVER (ORDER BY "{item_col}") - 1  AS item_idx,
                {rating_expr} AS rating,
                "{user_col}"  AS user_id,
                "{item_col}"  AS item_id
            FROM
                RAW;
            """)
            pbar.update(1)

            # Step 2. Extract unique ID mappings
            pbar.set_postfix_str("extracting ID maps")
            user_map = con.query('''
                       SELECT
                           DISTINCT user_idx,
                           user_id
                       FROM
                           ENCODED
                       ORDER BY user_idx''')
            item_map = con.query('''
                       SELECT
                           DISTINCT item_idx,
                           item_id
                       FROM
                           ENCODED
                       ORDER BY item_idx''')
            # Separated UserID and ItemID to separated Dataframe
            pbar.update(1)

            pbar.set_postfix_str("Send out the Encoded data.")
            encoded = con.query('SELECT * FROM ENCODED;')
            pbar.update(1)

        # Step 3. convert to numpy C arrays
        pbar.set_postfix_str("converting to C arrays")
        row_arr  = encoded["user_idx"].values.astype(np.int32)
        col_arr  = encoded["item_idx"].values.astype(np.int32)
        data_arr = encoded["rating"].values.astype(dtype)
        user_ids = user_map["user_id"].values
        item_ids = item_map["item_id"].values
        n_users  = len(user_ids)
        n_items  = len(item_ids)
        pbar.update(1)
        
        #logger.warning(f'number of n_users = {n_users} and number of n_items = {n_items}...')
        #from pdb import set_trace
        #set_trace()

        # Step 4. build sparse matrix
        pbar.set_postfix_str("building COO sparse matrix")
        interactions = sp.coo_matrix(
                       (data_arr, (row_arr, col_arr)),
                        shape = (n_users, n_items),
                        dtype = np.dtype(dtype))
        pbar.update(1)

    logger.info("norm_exchange: n_users = %d, n_items = %d, nnz = %d",
                 n_users, n_items, len(data_arr))
    del encoded, row_arr, col_arr, data_arr
    gc.collect()
    logger.debug("norm_exchange: done")
    return interactions, user_ids, item_ids


def fileload_interactions(path       : str,
                          user_col   : str = "user_id",
                          item_col   : str = "item_id",
                          rating_col : Optional[str] = None,
                          dtype      : str = DType,
                         ) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Load interactions directly from a flatfiles.
    _________________________________________________________
    Parameters
    path       : Absolute or relative path to the DuckDB/Parquet/CSV file.
    user_col   : Column name for user identifiers.
    item_col   : Column name for item identifiers.
    rating_col : Optional rating column name.
    dtype      : NumPy dtype string.
    """
    path = Path(path)
    logger.info("load_interactions_from_csv: path = %s", path)
    if not path.is_file():
        raise FileNotFoundError(f"Interaction data (CSV/parquet/db) not found: {path}")
    datague = load_data(data_path = path)
    if rating_col is None:
        rating_col = _cfg.get('RATING', 'ColumnName')
    result  = norm_exchange(datague,
                            user_col   = user_col,
                            item_col   = item_col,
                            rating_col = rating_col,
                            dtype      = dtype)
    del datague
    gc.collect()
    return result


if __name__ == '__main__':
    pass
