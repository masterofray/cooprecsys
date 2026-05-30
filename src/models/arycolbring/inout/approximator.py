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


"""
predictor.py
------------
Inference subclass of AryColBringBase.

Implements ``predict`` (pointwise scores) and ``predict_rank``
(per-user item rankings) against a fitted embedding matrix.

Typical production flow
-----------------------
1. Train with ``AryColBringTrainer``.
2. Export embeddings via ``get_item_representations`` /
   ``get_user_representations``, serialise them (e.g. with joblib/pickle).
3. At serving time, construct an ``AryColBringPredictor`` with the same
   hyper-parameters and load the saved embeddings directly onto the
   instance attributes before calling ``predict`` or ``predict_rank``.

Calling ``fit`` or ``fit_partial`` on this class raises
``NotImplementedError`` — use :class:`AryColBringTrainer` for training.
"""

import logging
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from ..CLproximity import (
    CSRMatrix,
    predict_arycolbring,
    predict_ranks,
)
from .Scaffold import (
    AryColBringBase,
    CYTHON_DTYPE,
)

logger = logging.getLogger(__name__)

__all__ = ["AryColBringPredictor"]


class AryColBringPredictor(AryColBringBase):
    """
    Inference interface for a fitted AryColBring collaborative filtering model.

    Exposes ``predict`` for pointwise scores and ``predict_rank`` for
    item-rank matrices.  The model must be fitted (embedding matrices must be
    populated) before any inference call.

    Raises ``NotImplementedError`` if ``fit`` or ``fit_partial`` are called —
    use :class:`AryColBringTrainer` for training.

    Parameters mirror ``AryColBringBase``; see that class for full documentation.
    """

    # ── public inference interface ────────────────────────────────────────────

    def predict(
        self,
        user_ids:      Union[int, list, np.ndarray],
        item_ids:      Union[list, np.ndarray],
        item_features: Optional[sp.spmatrix] = None,
        user_features: Optional[sp.spmatrix] = None,
        num_threads:   int = 1,
    ) -> np.ndarray:
        """
        Compute prediction scores for (user_id, item_id) pairs.

        Parameters
        ----------
        user_ids      : int, list, or int32 ndarray
                        Scalar broadcasts to all item_ids.
        item_ids      : list or int32 ndarray (same length as user_ids)
        item_features : optional CSR matrix [n_items × n_item_features]
        user_features : optional CSR matrix [n_users × n_user_features]
        num_threads   : OpenMP thread count (≥ 1)

        Returns
        -------
        np.ndarray, float32, shape (n_pairs,)
            Raw dot-product scores (higher = more relevant).
        """
        logger.debug(
            "AryColBringPredictor.predict: num_threads=%d", num_threads
        )
        self._check_initialized()

        # Scalar broadcast
        if isinstance(user_ids, int):
            user_ids = np.repeat(np.int32(user_ids), len(item_ids))

        # Coerce list / tuple → contiguous C arrays
        if isinstance(user_ids, (list, tuple)):
            user_ids = np.array(user_ids, dtype=np.int32)
        if isinstance(item_ids, (list, tuple)):
            item_ids = np.array(item_ids, dtype=np.int32)

        if user_ids.dtype != np.int32:
            user_ids = user_ids.astype(np.int32)
        if item_ids.dtype != np.int32:
            item_ids = item_ids.astype(np.int32)

        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"user_ids length ({len(user_ids)}) != "
                f"item_ids length ({len(item_ids)})"
            )
        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")
        if user_ids.min() < 0 or item_ids.min() < 0:
            raise ValueError(
                "Negative user_id or item_id found. "
                "Check for integer overflow or bad input."
            )

        n_users = int(user_ids.max()) + 1
        n_items = int(item_ids.max()) + 1

        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        predictions = np.empty(len(user_ids), dtype=np.float32)

        predict_arycolbring(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            user_ids,
            item_ids,
            predictions,
            self._get_model_data(),
            num_threads,
        )

        return predictions

    def predict_rank(
        self,
        test_interactions:   sp.spmatrix,
        train_interactions:  Optional[sp.spmatrix] = None,
        item_features:       Optional[sp.spmatrix] = None,
        user_features:       Optional[sp.spmatrix] = None,
        num_threads:         int  = 1,
        check_intersections: bool = True,
    ) -> sp.csr_matrix:
        """
        Compute item ranks for all test-positive interactions.

        Each entry in the returned matrix holds the 0-based rank of that
        test positive among all items (lower = better).  Training positives
        supplied via ``train_interactions`` are excluded from the ranking
        denominator.

        Parameters
        ----------
        test_interactions  : sparse matrix [n_users × n_items]
        train_interactions : optional sparse matrix (positives to exclude)
        item_features      : optional CSR feature matrix
        user_features      : optional CSR feature matrix
        num_threads        : OpenMP thread count (≥ 1)
        check_intersections: raise if test / train share any interaction

        Returns
        -------
        scipy.sparse.csr_matrix of the same sparsity pattern as
        test_interactions, where ``.data`` holds each item's 0-based rank.
        """
        logger.debug(
            "AryColBringPredictor.predict_rank: num_threads=%d", num_threads
        )
        self._check_initialized()

        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")

        if check_intersections:
            self._check_test_train_intersections(
                test_interactions, train_interactions
            )

        n_users, n_items = test_interactions.shape
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError(
                "item_features column count does not match embedding rows: "
                f"{item_features.shape[1]} vs {self.item_embeddings.shape[0]}"
            )
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError(
                "user_features column count does not match embedding rows: "
                f"{user_features.shape[1]} vs {self.user_embeddings.shape[0]}"
            )

        test_interactions = test_interactions.tocsr()
        test_interactions = self._to_cython_dtype(test_interactions)

        if train_interactions is None:
            train_interactions = sp.csr_matrix(
                (n_users, n_items), dtype=CYTHON_DTYPE
            )
        else:
            train_interactions = train_interactions.tocsr()
            train_interactions = self._to_cython_dtype(train_interactions)

        ranks = sp.csr_matrix(
            (
                np.zeros_like(test_interactions.data),
                test_interactions.indices,
                test_interactions.indptr,
            ),
            shape=test_interactions.shape,
        )

        predict_ranks(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            CSRMatrix(test_interactions),
            CSRMatrix(train_interactions),
            ranks.data,
            self._get_model_data(),
            num_threads,
        )

        return ranks

    # ── training methods — not implemented on the predictor ──────────────────

    def fit(self, *args, **kwargs) -> "AryColBringPredictor":
        """Not available on the predictor.  Use ``AryColBringTrainer``."""
        raise NotImplementedError(
            "AryColBringPredictor does not support training. "
            "Use AryColBringTrainer to fit a model, then load the "
            "resulting embeddings into this predictor for serving."
        )

    def fit_partial(self, *args, **kwargs) -> "AryColBringPredictor":
        """Not available on the predictor.  Use ``AryColBringTrainer``."""
        raise NotImplementedError(
            "AryColBringPredictor does not support training. "
            "Use AryColBringTrainer to fit a model, then load the "
            "resulting embeddings into this predictor for serving."
        )

if __name__ == '__main__':
    pass
