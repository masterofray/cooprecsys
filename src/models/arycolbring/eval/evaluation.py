#!/usr/bin/env python3

"""
arycolbring.evaluation
~~~~~~~~~~~~~~~~~~~~~~~
Ranking-quality metrics for evaluating a fitted AryColBring model.

All functions:
  - Accept the fitted model and sparse interaction matrices.
  - Call model.predict_rank() internally (which delegates to compiled Cython).
  - Log at DEBUG level.
  - Wrap multi-user loops in tqdm progress bars.

Metrics
-------
precision_at_k   : fraction of top-k recommendations that are true positives
recall_at_k      : fraction of true positives recovered in top-k
auc_score        : area under the ROC curve (per user, then averaged)
reciprocal_rank  : 1 / rank of the highest-ranked true positive per user
"""

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"

from __future__ import annotations

import gc
import logging
import configparser
import os
from typing import Optional

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from .cy import CSRMatrix, calculate_auc_from_rank

logger = logging.getLogger(__name__)

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

TQDM_COLOUR = _cfg.get("tqdm", "colour", fallback="#05ad46")
TQDM_NCOLS  = _cfg.getint("tqdm", "ncols", fallback=80)

__all__ = ["precision_at_k", "recall_at_k", "auc_score", "reciprocal_rank"]


def _get_ranks(
    model,
    test_interactions:   sp.spmatrix,
    train_interactions:  Optional[sp.spmatrix],
    user_features:       Optional[sp.spmatrix],
    item_features:       Optional[sp.spmatrix],
    num_threads:         int,
    check_intersections: bool,
    step_label:          str,
) -> sp.csr_matrix:
    """Shared rank-computation helper used by all metric functions."""
    logger.debug("_get_ranks: metric=%s num_threads=%d", step_label, num_threads)

    with tqdm(total=1, desc=f"Computing ranks ({step_label})",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:
        ranks = model.predict_rank(
            test_interactions,
            train_interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
            check_intersections=check_intersections,
        )
        pbar.update(1)

    return ranks


def precision_at_k(
    model,
    test_interactions:   sp.spmatrix,
    train_interactions:  Optional[sp.spmatrix] = None,
    k:                   int  = 10,
    user_features:       Optional[sp.spmatrix] = None,
    item_features:       Optional[sp.spmatrix] = None,
    preserve_rows:       bool = False,
    num_threads:         int  = 1,
    check_intersections: bool = True,
) -> np.ndarray:
    """
    Precision@k — fraction of the top-k recommendations that are true positives.

    Parameters
    ----------
    model               : fitted AryColBring instance
    test_interactions   : sparse [n_users × n_items]  — ground-truth positives
    train_interactions  : optional sparse — known positives to exclude from ranking
    k                   : cut-off rank
    user_features       : optional CSR [n_users × n_user_features]
    item_features       : optional CSR [n_items × n_item_features]
    preserve_rows       : if False, drop users with no test interactions
    num_threads         : OpenMP thread count
    check_intersections : raise if test/train overlap

    Returns
    -------
    np.ndarray float64, shape (n_active_users,) or (n_users,)
        Precision@k score for each user (0.0 – 1.0).
    """
    if num_threads < 1:
        raise ValueError("num_threads must be ≥ 1")
    if k < 1:
        raise ValueError("k must be ≥ 1")

    logger.debug("precision_at_k: k=%d num_threads=%d", k, num_threads)

    ranks = _get_ranks(model, test_interactions, train_interactions,
                       user_features, item_features,
                       num_threads, check_intersections, f"P@{k}")

    with tqdm(total=2, desc=f"Precision@{k}",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:
        pbar.set_postfix_str("masking top-k")
        ranks.data = np.less(ranks.data, k, ranks.data)
        precision  = np.squeeze(np.array(ranks.sum(axis=1))) / k
        pbar.update(1)

        pbar.set_postfix_str("filtering active users")
        if not preserve_rows:
            active = test_interactions.getnnz(axis=1) > 0
            precision = precision[active]
        pbar.update(1)

    gc.collect()
    logger.debug("precision_at_k: mean=%.4f", float(precision.mean()))
    return precision


def recall_at_k(
    model,
    test_interactions:   sp.spmatrix,
    train_interactions:  Optional[sp.spmatrix] = None,
    k:                   int  = 10,
    user_features:       Optional[sp.spmatrix] = None,
    item_features:       Optional[sp.spmatrix] = None,
    preserve_rows:       bool = False,
    num_threads:         int  = 1,
    check_intersections: bool = True,
) -> np.ndarray:
    """
    Recall@k — fraction of true positives recovered in the top-k results.

    A perfect score is 1.0.

    Parameters
    ----------
    Same as ``precision_at_k``.

    Returns
    -------
    np.ndarray float64, shape (n_active_users,) or (n_users,)
        Recall@k per user.
    """
    if num_threads < 1:
        raise ValueError("num_threads must be ≥ 1")
    if k < 1:
        raise ValueError("k must be ≥ 1")

    logger.debug("recall_at_k: k=%d num_threads=%d", k, num_threads)

    ranks = _get_ranks(model, test_interactions, train_interactions,
                       user_features, item_features,
                       num_threads, check_intersections, f"R@{k}")

    with tqdm(total=2, desc=f"Recall@{k}",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:
        pbar.set_postfix_str("masking top-k")
        ranks.data = np.less(ranks.data, k, ranks.data)
        retrieved  = np.squeeze(test_interactions.getnnz(axis=1)).astype(np.float64)
        hit        = np.squeeze(np.array(ranks.sum(axis=1), dtype=np.float64))
        pbar.update(1)

        pbar.set_postfix_str("filtering active users")
        if not preserve_rows:
            active    = test_interactions.getnnz(axis=1) > 0
            hit       = hit[active]
            retrieved = retrieved[active]
        pbar.update(1)

    # Guard against division by zero for users with 0 test positives
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(retrieved > 0, hit / retrieved, 0.0)

    gc.collect()
    logger.debug("recall_at_k: mean=%.4f", float(recall.mean()))
    return recall


def auc_score(
    model,
    test_interactions:   sp.spmatrix,
    train_interactions:  Optional[sp.spmatrix] = None,
    user_features:       Optional[sp.spmatrix] = None,
    item_features:       Optional[sp.spmatrix] = None,
    preserve_rows:       bool = False,
    num_threads:         int  = 1,
    check_intersections: bool = True,
) -> np.ndarray:
    """
    ROC AUC — probability that a random positive ranks above a random negative.

    A perfect model scores 1.0; a random model scores 0.5.
    Users with no test positives or where all items are negative return 0.5.

    Parameters
    ----------
    Same as ``precision_at_k`` without the ``k`` argument.

    Returns
    -------
    np.ndarray float32, shape (n_active_users,) or (n_users,)
    """
    if num_threads < 1:
        raise ValueError("num_threads must be ≥ 1")

    logger.debug("auc_score: num_threads=%d", num_threads)

    ranks = _get_ranks(model, test_interactions, train_interactions,
                       user_features, item_features,
                       num_threads, check_intersections, "AUC")

    if not np.all(ranks.data >= 0):
        raise RuntimeError(
            "Rank data contains negative values — this should not happen. "
            "Please report this as a bug."
        )

    with tqdm(total=2, desc="AUC",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:
        pbar.set_postfix_str("preparing train positive counts")
        auc = np.zeros(ranks.shape[0], dtype=np.float32)

        if train_interactions is not None:
            num_train_positives = np.squeeze(
                np.array(train_interactions.getnnz(axis=1)).astype(np.int32)
            )
        else:
            num_train_positives = np.zeros(
                test_interactions.shape[0], dtype=np.int32
            )
        pbar.update(1)

        pbar.set_postfix_str("computing AUC via Cython kernel")
        calculate_auc_from_rank(
            CSRMatrix(ranks),
            num_train_positives,
            ranks.data,
            auc,
            num_threads,
        )
        pbar.update(1)

    if not preserve_rows:
        auc = auc[test_interactions.getnnz(axis=1) > 0]

    gc.collect()
    logger.debug("auc_score: mean=%.4f", float(auc.mean()))
    return auc


def reciprocal_rank(
    model,
    test_interactions:   sp.spmatrix,
    train_interactions:  Optional[sp.spmatrix] = None,
    user_features:       Optional[sp.spmatrix] = None,
    item_features:       Optional[sp.spmatrix] = None,
    preserve_rows:       bool = False,
    num_threads:         int  = 1,
    check_intersections: bool = True,
) -> np.ndarray:
    """
    Mean Reciprocal Rank — 1 / rank of the highest-ranked true positive.

    A perfect model scores 1.0 (true positive is rank-1).
    Users with no positives score 0.0.

    Parameters
    ----------
    Same as ``auc_score``.

    Returns
    -------
    np.ndarray float64, shape (n_active_users,) or (n_users,)
    """
    if num_threads < 1:
        raise ValueError("num_threads must be ≥ 1")

    logger.debug("reciprocal_rank: num_threads=%d", num_threads)

    ranks = _get_ranks(model, test_interactions, train_interactions,
                       user_features, item_features,
                       num_threads, check_intersections, "MRR")

    with tqdm(total=2, desc="Reciprocal Rank",
              colour=TQDM_COLOUR, ncols=TQDM_NCOLS) as pbar:
        pbar.set_postfix_str("converting ranks to reciprocals")
        ranks.data = 1.0 / (ranks.data + 1.0)
        pbar.update(1)

        pbar.set_postfix_str("taking per-user max")
        rr = np.squeeze(np.array(ranks.max(axis=1).todense()))
        if not preserve_rows:
            rr = rr[test_interactions.getnnz(axis=1) > 0]
        pbar.update(1)

    gc.collect()
    logger.debug("reciprocal_rank: mean=%.4f", float(rr.mean()))
    return rr
