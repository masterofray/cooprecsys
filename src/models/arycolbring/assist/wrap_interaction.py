#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-30"


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


def describe_interactions(interactions: sp.spmatrix) -> pd.DataFrame:
    """
    Return a summary DataFrame of a sparse interaction matrix via DuckDB.

    Columns: n_users, n_items, nnz, density, avg_interactions_per_user,
             min_interactions_per_user, max_interactions_per_user.
    """
    logger.debug("describe_interactions: shape=%s", interactions.shape)

    mat = interactions.tocsr()
    row_counts = np.diff(mat.indptr).astype(np.float64)

    with duckdb_connection() as con:
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
