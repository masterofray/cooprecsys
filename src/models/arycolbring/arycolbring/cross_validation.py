# coding=utf-8
"""
arycolbring.cross_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset splitting utilities for collaborative-filtering evaluation.
All operations log at DEBUG level; tqdm tracks multi-step splits.
"""

from __future__ import annotations

import gc
import logging
import configparser
import os
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

logger = logging.getLogger(__name__)

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

TQDM_COLOUR = _cfg.get("tqdm", "colour", fallback="#05ad46")
TQDM_NCOLS  = _cfg.getint("tqdm", "ncols", fallback=80)


def _shuffle(
    uids: np.ndarray,
    iids: np.ndarray,
    data: np.ndarray,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-place shuffle of COO triplets via a permutation index array."""
    shuffle_indices = np.arange(len(uids), dtype=np.int32)
    random_state.shuffle(shuffle_indices)
    return uids[shuffle_indices], iids[shuffle_indices], data[shuffle_indices]


def random_train_test_split(
    interactions:    sp.spmatrix,
    test_percentage: float = 0.2,
    random_state:    Optional[int | np.random.RandomState] = None,
) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
    """
    Randomly split an interaction matrix into train and test sets.

    Non-zero entries are shuffled and the last ``test_percentage`` fraction
    is allocated to the test set.  The train set keeps the remaining
    interactions.  Both matrices preserve the original shape so that
    user/item index spaces are identical.

    Parameters
    ----------
    interactions    : scipy sparse matrix of any format (auto-converted to COO)
    test_percentage : fraction of interactions to use as test  (default 0.20)
    random_state    : int seed or ``np.random.RandomState`` instance

    Returns
    -------
    (train, test) : pair of scipy.sparse.coo_matrix

    Raises
    ------
    TypeError   – if interactions is not a scipy sparse matrix
    ValueError  – if test_percentage is not in (0, 1)
    RuntimeError – if the matrix has no non-zero entries
    """
    logger.debug(
        "random_train_test_split: shape=%s test_pct=%.2f",
        interactions.shape if hasattr(interactions, "shape") else "?",
        test_percentage,
    )

    if not sp.issparse(interactions):
        raise TypeError(
            "interactions must be a scipy sparse matrix, "
            f"got {type(interactions).__name__}."
        )
    if not (0 < test_percentage < 1):
        raise ValueError(
            f"test_percentage must be in (0, 1), got {test_percentage}."
        )
    if interactions.nnz == 0:
        raise RuntimeError(
            "interactions has no non-zero entries — cannot split an empty matrix."
        )

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)

    with tqdm(total=4, desc="Splitting interactions",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:

        pbar.set_postfix_str("converting to COO")
        interactions = interactions.tocoo()
        shape = interactions.shape
        uids  = np.array(interactions.row,  dtype=np.int32)
        iids  = np.array(interactions.col,  dtype=np.int32)
        data  = np.array(interactions.data, dtype=interactions.dtype)
        pbar.update(1)

        pbar.set_postfix_str("shuffling entries")
        uids, iids, data = _shuffle(uids, iids, data, random_state)
        pbar.update(1)

        pbar.set_postfix_str("computing cutoff")
        cutoff    = int((1.0 - test_percentage) * len(uids))
        train_idx = slice(None, cutoff)
        test_idx  = slice(cutoff, None)
        logger.debug(
            "random_train_test_split: total=%d train=%d test=%d",
            len(uids), cutoff, len(uids) - cutoff,
        )
        pbar.update(1)

        pbar.set_postfix_str("building sparse matrices")
        train = sp.coo_matrix(
            (data[train_idx], (uids[train_idx], iids[train_idx])),
            shape=shape,
            dtype=interactions.dtype,
        )
        test = sp.coo_matrix(
            (data[test_idx], (uids[test_idx], iids[test_idx])),
            shape=shape,
            dtype=interactions.dtype,
        )
        pbar.update(1)

    del uids, iids, data
    gc.collect()
    logger.debug("random_train_test_split: done  train.nnz=%d test.nnz=%d",
                 train.nnz, test.nnz)

    return train, test


def user_based_train_test_split(
    interactions:    sp.spmatrix,
    test_percentage: float = 0.2,
    random_state:    Optional[int | np.random.RandomState] = None,
) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
    """
    Split interactions such that each user's interactions are split
    independently.  Ensures every user appears in both train and test
    (as long as they have ≥ 2 interactions).

    Users with only one interaction are placed entirely in train.

    Parameters
    ----------
    interactions    : scipy sparse matrix
    test_percentage : fraction of each user's interactions for test
    random_state    : int seed or ``np.random.RandomState``

    Returns
    -------
    (train, test) : pair of scipy.sparse.coo_matrix
    """
    logger.debug(
        "user_based_train_test_split: shape=%s test_pct=%.2f",
        interactions.shape, test_percentage,
    )

    if not sp.issparse(interactions):
        raise TypeError(
            "interactions must be a scipy sparse matrix, "
            f"got {type(interactions).__name__}."
        )
    if not (0 < test_percentage < 1):
        raise ValueError(
            f"test_percentage must be in (0, 1), got {test_percentage}."
        )

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)

    csr = interactions.tocsr()
    n_users = csr.shape[0]

    train_rows, train_cols, train_data = [], [], []
    test_rows,  test_cols,  test_data  = [], [], []

    with tqdm(total=n_users, desc="User-based split",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:

        for user_id in range(n_users):
            start = csr.indptr[user_id]
            stop  = csr.indptr[user_id + 1]

            cols_u = csr.indices[start:stop]
            data_u = csr.data[start:stop]
            n_u    = len(cols_u)

            perm = random_state.permutation(n_u)
            n_test = max(0, int(np.floor(n_u * test_percentage)))

            if n_u < 2:
                n_test = 0  # keep sole interaction in train

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

            pbar.update(1)

    shape = interactions.shape
    dtype = interactions.dtype

    train = sp.coo_matrix(
        (np.array(train_data, dtype=dtype),
         (np.array(train_rows, dtype=np.int32),
          np.array(train_cols, dtype=np.int32))),
        shape=shape, dtype=dtype,
    )
    test = sp.coo_matrix(
        (np.array(test_data, dtype=dtype),
         (np.array(test_rows, dtype=np.int32),
          np.array(test_cols, dtype=np.int32))),
        shape=shape, dtype=dtype,
    )

    logger.debug("user_based_train_test_split: train.nnz=%d test.nnz=%d",
                 train.nnz, test.nnz)
    gc.collect()
    return train, test
