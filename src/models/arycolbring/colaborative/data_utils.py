# coding=utf-8
"""
arycolbring.data_utils
~~~~~~~~~~~~~~~~~~~~~~
DuckDB-backed data loading and interaction-matrix construction utilities.
All public functions log via the standard ``logging`` module at DEBUG level.
"""

from __future__ import annotations

import gc
import logging
import configparser
import os
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

# ── logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────────────
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

TQDM_COLOUR   = _cfg.get("tqdm",   "colour",    fallback="#05ad46")
TQDM_NCOLS    = _cfg.getint("tqdm", "ncols",     fallback=80)
DEFAULT_DTYPE = _cfg.get("model",  "dtype",      fallback="float32")
DUCKDB_THREADS = _cfg.getint("duckdb", "threads", fallback=4)


# ── DuckDB connection context ─────────────────────────────────────────────────

@contextmanager
def duck_connection(threads: int = DUCKDB_THREADS):
    """
    Context manager that yields an in-process DuckDB connection
    and guarantees it is closed on exit.

    Usage::

        with duck_connection() as con:
            df = con.execute("SELECT * FROM my_table").df()
    """
    logger.debug("Opening DuckDB connection with %d threads", threads)
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={threads}")
    try:
        yield con
    finally:
        logger.debug("Closing DuckDB connection")
        con.close()


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_interactions_from_df(
    df: pd.DataFrame,
    user_col:  str = "user_id",
    item_col:  str = "item_id",
    rating_col: Optional[str] = None,
    dtype: str = DEFAULT_DTYPE,
) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Convert a Pandas DataFrame of (user, item[, rating]) rows into a
    sparse COO interaction matrix.  Uses DuckDB for the encoding query
    so it stays efficient even for tens of millions of rows.

    Parameters
    ----------
    df         : DataFrame with at least ``user_col`` and ``item_col`` columns.
    user_col   : Name of the user identifier column.
    item_col   : Name of the item identifier column.
    rating_col : Name of an optional rating column.
                 If None, all interactions are set to 1.0.
    dtype      : NumPy dtype string for the sparse matrix data array.

    Returns
    -------
    interactions : scipy.sparse.coo_matrix  (n_users × n_items)
    user_ids     : np.ndarray  — mapping from integer index → original user id
    item_ids     : np.ndarray  — mapping from integer index → original item id
    """
    logger.debug(
        "load_interactions_from_df: shape=%s user_col=%s item_col=%s rating_col=%s",
        df.shape, user_col, item_col, rating_col,
    )

    required = {user_col, item_col}
    if rating_col:
        required.add(rating_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    with tqdm(total=4, desc="Building interaction matrix",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:

        # Step 1 — register df in DuckDB and encode IDs
        with duck_connection() as con:
            pbar.set_postfix_str("registering dataframe")
            con.register("raw_df", df)

            rating_expr = (f'CAST("{rating_col}" AS DOUBLE)'
                           if rating_col else "1.0")

            encoded = con.execute(f"""
                SELECT
                    DENSE_RANK() OVER (ORDER BY "{user_col}") - 1 AS user_idx,
                    DENSE_RANK() OVER (ORDER BY "{item_col}") - 1 AS item_idx,
                    {rating_expr}                                  AS rating,
                    "{user_col}"                                   AS user_id,
                    "{item_col}"                                   AS item_id
                FROM raw_df
            """).df()
            pbar.update(1)

            # Step 2 — extract unique ID mappings
            pbar.set_postfix_str("extracting ID maps")
            user_map = con.execute(
                'SELECT DISTINCT user_idx, user_id FROM encoded ORDER BY user_idx'
            ).df()
            item_map = con.execute(
                'SELECT DISTINCT item_idx, item_id FROM encoded ORDER BY item_idx'
            ).df()
            pbar.update(1)

        # Step 3 — convert to numpy C arrays
        pbar.set_postfix_str("converting to C arrays")
        row_arr  = encoded["user_idx"].values.astype(np.int32)
        col_arr  = encoded["item_idx"].values.astype(np.int32)
        data_arr = encoded["rating"].values.astype(dtype)

        n_users = int(row_arr.max()) + 1
        n_items = int(col_arr.max()) + 1

        logger.debug(
            "load_interactions_from_df: n_users=%d n_items=%d nnz=%d",
            n_users, n_items, len(data_arr),
        )
        pbar.update(1)

        # Step 4 — build sparse matrix
        pbar.set_postfix_str("building COO sparse matrix")
        interactions = sp.coo_matrix(
            (data_arr, (row_arr, col_arr)),
            shape=(n_users, n_items),
            dtype=np.dtype(dtype),
        )
        user_ids = user_map["user_id"].values
        item_ids = item_map["item_id"].values
        pbar.update(1)

    del encoded, row_arr, col_arr, data_arr
    gc.collect()
    logger.debug("load_interactions_from_df: done")

    return interactions, user_ids, item_ids


def load_interactions_from_csv(
    path: str,
    user_col:   str = "user_id",
    item_col:   str = "item_id",
    rating_col: Optional[str] = None,
    dtype: str = DEFAULT_DTYPE,
) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Load interactions directly from a CSV file via DuckDB (zero-copy read).

    Parameters
    ----------
    path       : Absolute or relative path to the CSV file.
    user_col   : Column name for user identifiers.
    item_col   : Column name for item identifiers.
    rating_col : Optional rating column name.
    dtype      : NumPy dtype string.

    Returns
    -------
    Same tuple as ``load_interactions_from_df``.
    """
    logger.debug("load_interactions_from_csv: path=%s", path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Interaction CSV not found: {path}")

    with tqdm(total=2, desc="Reading CSV", colour=TQDM_COLOUR,
              ncols=TQDM_NCOLS) as pbar:
        pbar.set_postfix_str("reading via DuckDB")

        with duck_connection() as con:
            rating_expr = (f'CAST("{rating_col}" AS DOUBLE)'
                           if rating_col else "1.0")
            df = con.execute(f"""
                SELECT "{user_col}", "{item_col}", {rating_expr} AS rating
                FROM read_csv_auto('{path}')
            """).df()
        pbar.update(1)

        pbar.set_postfix_str("encoding interactions")
        result = load_interactions_from_df(
            df,
            user_col=user_col,
            item_col=item_col,
            rating_col="rating",
            dtype=dtype,
        )
        pbar.update(1)

    del df
    gc.collect()
    return result


def describe_interactions(interactions: sp.spmatrix) -> pd.DataFrame:
    """
    Return a summary DataFrame of a sparse interaction matrix via DuckDB.

    Columns: n_users, n_items, nnz, density, avg_interactions_per_user,
             min_interactions_per_user, max_interactions_per_user.
    """
    logger.debug("describe_interactions: shape=%s", interactions.shape)

    mat = interactions.tocsr()
    row_counts = np.diff(mat.indptr).astype(np.float64)

    with duck_connection() as con:
        con.register("row_counts_arr",
                     pd.DataFrame({"nnz_per_user": row_counts}))
        stats = con.execute("""
            SELECT
                COUNT(*)                     AS n_users,
                AVG(nnz_per_user)            AS avg_interactions_per_user,
                MIN(nnz_per_user)            AS min_interactions_per_user,
                MAX(nnz_per_user)            AS max_interactions_per_user
            FROM row_counts_arr
        """).df()

    n_users, n_items = interactions.shape
    nnz     = interactions.nnz
    density = nnz / (n_users * n_items) if n_users * n_items > 0 else 0.0

    summary = pd.DataFrame({
        "n_users":                   [n_users],
        "n_items":                   [n_items],
        "nnz":                       [nnz],
        "density":                   [density],
        "avg_interactions_per_user": [stats["avg_interactions_per_user"].iloc[0]],
        "min_interactions_per_user": [stats["min_interactions_per_user"].iloc[0]],
        "max_interactions_per_user": [stats["max_interactions_per_user"].iloc[0]],
    })

    logger.debug("describe_interactions: %s", summary.to_dict(orient="records")[0])
    return summary


def validate_sparse_matrix(mat: sp.spmatrix, name: str = "matrix") -> None:
    """
    Raise informative errors if *mat* is not a valid finite sparse matrix.

    Raises
    ------
    TypeError         – if mat is not a scipy sparse matrix
    ValueError        – if mat contains NaN / Inf values
    RuntimeError      – if mat has zero rows or columns
    ReferenceError    – if mat.data is None or empty unexpectedly
    """
    logger.debug("validate_sparse_matrix: name=%s type=%s", name, type(mat))

    if not sp.issparse(mat):
        raise TypeError(
            f"Expected a scipy sparse matrix for '{name}', "
            f"got {type(mat).__name__}."
        )
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise RuntimeError(
            f"Sparse matrix '{name}' has degenerate shape {mat.shape}."
        )
    if mat.nnz == 0:
        logger.warning("validate_sparse_matrix: '%s' has zero non-zero entries", name)
        return

    data = mat.data
    if data is None:
        raise ReferenceError(f"Sparse matrix '{name}' has None data array.")

    if not np.isfinite(data).all():
        raise ValueError(
            f"Sparse matrix '{name}' contains NaN or Inf values. "
            "Check your input data."
        )

    logger.debug("validate_sparse_matrix: '%s' OK — nnz=%d", name, mat.nnz)
