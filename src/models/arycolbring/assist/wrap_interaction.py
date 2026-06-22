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

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp

LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import _cfg, logger
from db      import duckdb_connection


def describe_interactions(interactions: sp.spmatrix) -> pd.DataFrame:
    """
    Return a summary DataFrame of a sparse interaction matrix via DuckDB.
    Columns: n_users, n_items, nnz, density, avg_interactions_per_user,
             min_interactions_per_user, max_interactions_per_user.
    """
    logger.debug("describe_interactions: shape = %s", interactions.shape)
    mat        = interactions.tocsr()
    row_counts = np.diff(mat.indptr).astype(np.float64)
    query      = """
                 SELECT
                    COUNT(*)          AS n_users,
                    AVG(nnz_per_user) AS avg_interactions_per_user,
                    MIN(nnz_per_user) AS min_interactions_per_user,
                    MAX(nnz_per_user) AS max_interactions_per_user
                 FROM
                    RowCounts"""
    with duckdb_connection() as con:
        con.register_dataframe("RowCounts",
        pd.DataFrame({"nnz_per_user": row_counts}))
        stats        = con.query(query)
    n_users, n_items = interactions.shape
    nnz     = interactions.nnz
    density = nnz / (n_users * n_items) if n_users * n_items > 0 else 0.0
    logger.debug(f'We got density interaction is {density}!')
    summary = pd.DataFrame({
              "n_users" : [n_users],
              "n_items" : [n_items],
              "nnz"     : [nnz],
              "density" : [density],
              "avg_interactions_per_user" : [stats["avg_interactions_per_user"].iloc[0]],
              "min_interactions_per_user" : [stats["min_interactions_per_user"].iloc[0]],
              "max_interactions_per_user" : [stats["max_interactions_per_user"].iloc[0]],
              })
    logger.debug("describe_interactions: %s", 
                  summary.to_dict(orient = "records")[0])
    return summary


def validate_sparse_matrix(mat  : sp.spmatrix, 
                           name : str = "matrix",
                          ) -> None:
    """
    Raise informative errors if *mat* is not a valid finite sparse matrix.
    TypeError      - if mat is not a scipy sparse matrix
    ValueError     - if mat contains NaN / Inf values
    RuntimeError   - if mat has zero rows or columns
    ReferenceError - if mat.data is None or empty unexpectedly
    """
    logger.debug("Validated: name = %s type = %s", name, type(mat))
    if not sp.issparse(mat):
        logger.error(f"Expected a scipy sparse matrix for '{name}', "
                     f"got {type(mat).__name__}.")
        raise TypeError()
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        logger.error(f"Sparse matrix '{name}' has degenerate shape {mat.shape}.")
        raise RuntimeError()
    if mat.nnz == 0:
        logger.warning("Validated: '%s' has zero non-zero entries", name)
        return None
    data = mat.data
    if data is None:
        logger.error(f"Sparse matrix '{name}' has None data array.")
        raise ReferenceError()
    if not np.isfinite(data).all():
        logger.error(f"Sparse matrix '{name}' contains NaN or Inf values. "
                      "Check your input data.")
        raise ValueError()
    logger.info("Validated: '%s' OK - nnz = %d", name, mat.nnz)


if __name__ == '__main__':
    pass