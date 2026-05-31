#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-31"


"""
data_utils.py
-------------
Data loading utilities for AryColBring collaborative filtering model.
Provides functions to load interactions from DataFrames and CSV files.
"""

import gc
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .assist.bloatdata import norm_exchange, fileload_interactions as _fileload
from .assist.wrap_interaction import describe_interactions, validate_sparse_matrix

logger = logging.getLogger(__name__)

__all__ = [
    "load_interactions_from_df",
    "load_interactions_from_csv",
    "describe_interactions",
    "validate_sparse_matrix",
]


def load_interactions_from_df(
    df:          pd.DataFrame,
    user_col:    str = "user_id",
    item_col:    str = "item_id",
    rating_col:  Optional[str] = None,
    dtype:       str = "float32",
) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Convert a Pandas DataFrame of (user, item[, rating]) rows into a
    sparse COO interaction matrix.

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
        "load_interactions_from_df: shape=%s user_col=%s item_col=%s",
        df.shape, user_col, item_col
    )
    result = norm_exchange(
        data=df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        dtype=dtype,
    )
    logger.info(
        "load_interactions_from_df: created matrix shape=%s nnz=%d",
        result[0].shape, result[0].nnz
    )
    return result


def load_interactions_from_csv(
    path:        str,
    user_col:    str = "user_id",
    item_col:    str = "item_id",
    rating_col:  Optional[str] = None,
    dtype:       str = "float32",
) -> Tuple[sp.coo_matrix, np.ndarray, np.ndarray]:
    """
    Load interactions directly from a CSV file.

    Parameters
    ----------
    path       : Absolute or relative path to the CSV file.
    user_col   : Column name for user identifiers.
    item_col   : Column name for item identifiers.
    rating_col : Optional rating column name.
    dtype      : NumPy dtype string.

    Returns
    -------
    interactions : scipy.sparse.coo_matrix  (n_users × n_items)
    user_ids     : np.ndarray  — mapping from integer index → original user id
    item_ids     : np.ndarray  — mapping from integer index → original item id
    """
    logger.debug("load_interactions_from_csv: path=%s", path)
    result = _fileload(
        path=path,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        dtype=dtype,
    )
    logger.info(
        "load_interactions_from_csv: loaded matrix shape=%s nnz=%d",
        result[0].shape, result[0].nnz
    )
    return result


if __name__ == '__main__':
    pass
