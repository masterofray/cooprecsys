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

'''
This script flex.py was for group split train test
that for COO or CSR matrix data. You could not use
train test split in sklearn usually, so this purpose
to be created.
'''

import gc
import sys
import numpy as np
import scipy.sparse as sp
from   pathlib import Path
from   tqdm.auto import tqdm
from   typing import Optional, Tuple

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg


def _shuffle(uids   : np.ndarray,
             iids   : np.ndarray,
             data   : np.ndarray,
             rstate : np.random.RandomState = 2,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-place shuffle of COO triplets via a permutation index array."""
    index = np.arange(len(uids), dtype=np.int32)
    rstate.shuffle(index)
    lreturn = [uids[index], iids[index], data[index]]
    logger.debug('Suffle data in success.')
    return lreturn


def coo_ttsplit(interactions: sp.spmatrix,
                tratio      : float = 0.2,
                rstate      : Optional[int | np.random.RandomState] = None,
               ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
    """
    Randomly split an interaction matrix into train and test sets.
    Non-zero entries are shuffled and the last ``tratio`` fraction
    is allocated to the test set.  The train set keeps the remaining
    interactions.  Both matrices preserve the original shape so that
    user/item index spaces are identical.
    interactions : scipy sparse matrix of any format (auto-converted to COO)
    tratio       : fraction of interactions to use as test  (default 0.20)
    rstate       : int seed or ``np.random.RandomState`` instance
    It will returns : (train, test) -> pair of scipy.sparse.coo_matrix
    """
    logger.debug("random train test split: shape = %s test_pct = %.1f",
        interactions.shape if hasattr(interactions, "shape") else "?",
        tratio)

    if not sp.issparse(interactions):
        raise TypeError("interactions must be a scipy sparse matrix, "
                        f"got {type(interactions).__name__}.")
    if not (0 < tratio < 1):
        logger.error(f"tratio must be in (0, 1), got {tratio}.")
        raise ValueError()
    if interactions.nnz == 0:
        logger.error("interactions has no non-zero entries"\
                     "- cannot split an empty matrix.")
        raise RuntimeError()
    if not isinstance(rstate, np.random.RandomState):
        rstate = np.random.RandomState(seed = rstate)

    with tqdm(total       = 4, 
              desc        = "Splitting interactions",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1)
              as pbar:
        pbar.set_postfix_str("converting to COO")
        interactions = interactions.tocoo()
        shape = interactions.shape
        uids  = np.array(interactions.row,  dtype=np.int32)
        iids  = np.array(interactions.col,  dtype=np.int32)
        data  = np.array(interactions.data, dtype=interactions.dtype)
        pbar.update(1)

        pbar.set_postfix_str("shuffling entries")
        uids, iids, data = _shuffle(uids, iids, data, rstate)
        pbar.update(1)

        pbar.set_postfix_str("computing cutoff")
        cutoff    = int((1.0 - tratio) * len(uids))
        train_idx = slice(None, cutoff)
        test_idx  = slice(cutoff, None)
        pbar.update(1)

        pbar.set_postfix_str("building sparse matrices")
        train = sp.coo_matrix(
                (data[train_idx], (uids[train_idx], iids[train_idx])),
                shape=shape,
                dtype=interactions.dtype)
        test  = sp.coo_matrix(
                (data[test_idx], (uids[test_idx], iids[test_idx])),
                shape=shape,
                dtype=interactions.dtype)
        pbar.update(1)

    logger.debug("coo_ttsplit: total = %d train = %d test = %d",
                 len(uids), cutoff, len(uids) - cutoff)
    del uids, iids, data
    gc.collect()
    logger.debug("coo_ttsplit: done  train.nnz = %d test.nnz = %d",
                 train.nnz, test.nnz)
    return train, test


def user_based_train_test_split(
        interactions : sp.spmatrix,
        tratio       : float = 0.2,
        rstate       : Optional[int | np.random.RandomState] = None,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
    """
    Split interactions such that each user's interactions are split
    independently.  Ensures every user appears in both train and test
    (as long as they have ≥ 2 interactions).
    Users with only one interaction are placed entirely in train.
    interactions : scipy sparse matrix
    tratio       : fraction of each user's interactions for test
    rstate       : int seed or ``np.random.RandomState``
    It will returns : (train, test) -> pair of scipy.sparse.coo_matrix
    """
    logger.debug("user_based_train_test_split: shape = %s test_pct= %.1f",
                  interactions.shape, tratio)

    if not sp.issparse(interactions):
        raise TypeError("interactions must be a scipy sparse matrix, "
                       f"got {type(interactions).__name__}.")
    if not (0 < tratio < 1):
        raise ValueError(f"test ratio must be in (0, 1), got {tratio}.")
    if not isinstance(rstate, np.random.RandomState):
        rstate = np.random.RandomState(seed = rstate)

    csr     = interactions.tocsr()
    n_users = csr.shape[0]
    train_rows, train_cols, train_data = list(), list(), list()
    test_rows,  test_cols,  test_data  = list(), list(), list()

    for user_id in tqdm(range(n_users)
                        desc        = "User-based split",
                        colour      = _cfg.get('tqdm', 'colour'),
                        ncols       = _cfg.getint('tqdm', 'ncols'),
                        bar_format  = _cfg.get('tqdm', 'BarFormats'),
                        unit        = 'User',
                        mininterval = 0.1):
        start  = csr.indptr[user_id]
        stop   = csr.indptr[user_id + 1]
        cols_u = csr.indices[start:stop]
        data_u = csr.data[start:stop]
        n_u    = len(cols_u)
        perm   = rstate.permutation(n_u)
        n_test = max(0, int(np.floor(n_u * tratio)))
        if n_u < 2:
            n_test = int()  # keep sole interaction in train
        test_idx  = perm[:n_test]
        train_idx = perm[n_test:]

        for idx in train_idx:
            train_rows.append(user_id)
            train_cols.append(cols_u[idx])
            train_data.append(data_u[idx])

        for idx in test_idx:
            test_rows.append(user_id)
            test_cols.append(cols_u[idx])
            test_data.append(data_u[idx])

    shape = interactions.shape
    dtype = interactions.dtype

    train = sp.coo_matrix(
            (np.array(train_data, dtype=dtype),
            (np.array(train_rows, dtype=np.int32),
             np.array(train_cols, dtype=np.int32))),
             shape = shape,
             dtype = dtype)
    test = sp.coo_matrix(
           (np.array(test_data, dtype=dtype),
           (np.array(test_rows, dtype=np.int32),
            np.array(test_cols, dtype=np.int32))),
            shape = shape, 
            dtype = dtype)
    logger.debug("user_based_train_test_split: train.nnz = %d test.nnz = %d",
                 train.nnz, test.nnz)
    gc.collect()
    return train, test


if __name__ == "__main__":
    pass