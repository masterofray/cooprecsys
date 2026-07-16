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
approximator.py
_______________________________________________________________
Inference subclass of AryColBringBase.
Implements ``predict`` (pointwise scores) and ``predict_rank``
(per-user item rankings) against a fitted embedding matrix.

Typical production flow
_______________________________________________________________
1. Train with ``AryColBringTrainer``.
2. Export embeddings via ``get_item_representations`` /
   ``get_user_representations``, serialise them (e.g. with joblib/pickle).
3. At serving time, construct an ``AryColBringPredictor`` with the same
   hyper-parameters and load the saved embeddings directly onto the
   instance attributes before calling ``predict`` or ``predict_rank``.
"""

import sys
import numpy as np
from pathlib import Path
import scipy.sparse as sp
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder
from .scaffold import AryColBringBase, cydtype

try:
    import psutil
except ImportError:
    psutil = None

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[1]))
from CLproximity import (CSRMatrix,
                         predict_ranks,
                         predict_arycolbring)

sys.path.append(str(LocDir.parents[3]))
from configs import _cfg, logger



class AryColBringPredictor(AryColBringBase):
    """ 
    Inference interface for a fitted AryColBring collaborative filtering model.
    Exposes ``predict`` for pointwise scores and ``predict_rank`` for
    item-rank matrices.  The model must be fitted (embedding matrices must be
    populated) before any inference call.
    Parameters mirror ``AryColBringBase``; see that class for full documentation.
    """

    @staticmethod
    def _is_string_type(arr) -> bool:
        # Menangani input list/tuple biasa
        if not hasattr(arr, "dtype"):
            return isinstance(arr, (list, tuple)) and len(arr) > 0 and isinstance(arr[0], str)
        
        # Menangani tipe data NumPy/Pandas/Arrow
        dt_str = str(arr.dtype).lower()
        return (
            "string" in dt_str or 
            "object" in dt_str or 
            "arrow" in dt_str or 
            arr.dtype.kind in ('U', 'S', 'O'))

    def _warn_if_cross_join_memory_heavy(self, n_u: int, n_i: int) -> None:
        """
        Soft memory-safety check for predict()'s auto cross-join expansion
        (see the n_u != n_i branch). Never blocks execution — only logs a
        warning — since a caller may deliberately want a large batch.

        Instead of a hardcoded pair-count ceiling, this estimates the byte
        footprint of the expanded (user_ids_cy, item_ids_cy, predictions)
        arrays and compares it against a *fraction of currently available
        system RAM*, so the check adapts to whatever machine the package
        happens to run on (laptop vs. CI runner vs. production server).

        The fraction is read from the shared package config
        (``_cfg['predict']['cross_join_max_memory_fraction']``) so it can be
        tuned per-deployment without touching this module. If that config key
        is absent, a documented fallback of 0.25 (25% of available RAM) is
        used. If ``psutil`` isn't installed, the RAM figure can't be read at
        all, so the check is skipped entirely (debug-logged, not a hard
        dependency of the package).
        """
        n_pairs   = n_u * n_i
        est_bytes = n_pairs * (2 * np.dtype(np.int32).itemsize
                                + np.dtype(np.float32).itemsize)

        try:
            max_fraction = _cfg.getfloat('predict', 'cross_join_max_memory_fraction')
        except Exception:
            max_fraction = 0.25  # fallback only when [predict] section/key is absent

        if psutil is None:
            logger.debug(
                "psutil not installed; skipping RAM-aware cross-join check for "
                "%d pairs (~%.1f MB estimated). Install psutil to enable this "
                "safety check.", n_pairs, est_bytes / (1024 ** 2))
            return

        available = psutil.virtual_memory().available
        budget    = available * max_fraction
        if est_bytes > budget:
            logger.warning(
                "predict() cross-join is producing %d pairs (%d users x %d items, "
                "~%.1f MB estimated). This exceeds %.0f%% of currently available "
                "RAM (%.1f MB available right now). This may be unintentional and "
                "could cause heavy memory pressure or an OOM kill.",
                n_pairs, n_u, n_i, est_bytes / (1024 ** 2),
                max_fraction * 100, available / (1024 ** 2))

    # ── public inference interface ────────────────────────
    def predict(self,
                user_ids      : Union[int, list, np.ndarray],
                item_ids      : Union[list, np.ndarray],
                item_features : Optional[sp.spmatrix] = None,
                user_features : Optional[sp.spmatrix] = None,
                num_threads   : int = 1,
               ) -> np.ndarray:
        """
        Compute prediction scores for (user_id, item_id) pairs.
        user_ids      : int, list, or int32 ndarray
                        Scalar broadcasts to all item_ids.
        item_ids      : list or int32 ndarray
        item_features : optional CSR matrix [n_items x n_item_features]
        user_features : optional CSR matrix [n_users x n_user_features]
        num_threads   : OpenMP thread count (>= 1)

        Length handling for user_ids / item_ids:
        - Equal length N        -> strict pairwise scoring: (user_ids[i], item_ids[i]) for i in range(N).
        - One side has length 1 -> that scalar-like side is broadcast against the other (unchanged
                                    legacy behaviour; also now symmetric for a length-1 item_ids).
        - Unequal length > 1 on both sides -> treated as a reporting/cross-join call: every user is
                                    scored against every item (full N x M grid), flattened in
                                    row-major order (user 0 vs all items, then user 1 vs all items, ...).
                                    Returned array length is N * M in that case.

        The returns is np.ndarray with shape as (n_pairs,)
        Raw dot-product scores (higher = more relevant).
        """
        logger.debug("About how to run: num_threads = %d", num_threads)
        self._check_initialized()
        
        try:
            # 1. Normalisasi scalar -> array 1 elemen, list/tuple -> ndarray
            if np.ndim(user_ids) == 0:
                user_ids = np.array([user_ids])
            elif isinstance(user_ids, (list, tuple)):
                user_ids = np.array(user_ids)

            if np.ndim(item_ids) == 0:
                item_ids = np.array([item_ids])
            elif isinstance(item_ids, (list, tuple)):
                item_ids = np.array(item_ids)

            n_u, n_i = len(user_ids), len(item_ids)
            if n_u == 0 or n_i == 0:
                return np.array([], dtype=np.float32)

            # 2. Label Encoder untuk string, biarkan numeric as-is.
            #    Encode dulu pada array asli (belum di-expand) supaya LabelEncoder
            #    tidak memproses duplikat hasil cross-join secara sia-sia.
            user_encoder = None
            if self._is_string_type(user_ids):
                user_encoder = LabelEncoder()
                user_ids_cy = user_encoder.fit_transform(user_ids).astype(np.int32)
            else:
                user_ids_cy = np.asarray(user_ids, dtype=np.int32)

            item_encoder = None
            if self._is_string_type(item_ids):
                item_encoder = LabelEncoder()
                item_ids_cy = item_encoder.fit_transform(item_ids).astype(np.int32)
            else:
                item_ids_cy = np.asarray(item_ids, dtype=np.int32)

            # 3. Samakan dimensi user_ids_cy / item_ids_cy:
            #    - sama panjang            -> pairwise, tidak diapa-apakan
            #    - salah satu panjang 1    -> broadcast (scalar-like) ke panjang yang lain
            #    - keduanya > 1 & berbeda  -> cross-join penuh (dipakai untuk reporting)
            if n_u != n_i:
                if n_u == 1:
                    user_ids_cy = np.repeat(user_ids_cy, n_i)
                elif n_i == 1:
                    item_ids_cy = np.repeat(item_ids_cy, n_u)
                else:
                    logger.debug(
                        "user_ids length (%d) != item_ids length (%d); expanding to full "
                        "cross-join grid of %d pairs", n_u, n_i, n_u * n_i)
                    self._warn_if_cross_join_memory_heavy(n_u, n_i)
                    user_ids_cy = np.repeat(user_ids_cy, n_i)
                    item_ids_cy = np.tile(item_ids_cy, n_u)

            # Guardrails pencegah segmentation fault OpenMP (sanity check, seharusnya selalu lolos
            # setelah penyesuaian dimensi di atas)
            if len(user_ids_cy) != len(item_ids_cy):
                raise ValueError(f"user_ids length ({len(user_ids_cy)}) != item_ids length ({len(item_ids_cy)})")
            if num_threads < 1:
                raise ValueError("num_threads must be >= 1")

            if user_ids_cy.min() < 0 or item_ids_cy.min() < 0:
                raise ValueError("Negative user_id or item_id found after encoding.")

            # 3. Prediksi via backend Cython (predict_arycolbring)
            n_users = int(user_ids_cy.max()) + 1
            n_items = int(item_ids_cy.max()) + 1
            user_features, item_features = self._construct_feature_matrices(
                n_users, n_items, user_features, item_features)
            predictions = np.empty(len(user_ids_cy), dtype = np.float32)
            predict_arycolbring(CSRMatrix(item_features),
                                CSRMatrix(user_features),
                                user_ids_cy,
                                item_ids_cy,
                                predictions,
                                self._get_model_data(),
                                num_threads)

            # 4. Decoder label pada column yang telah di-encode sebelumnya
            if user_encoder is not None:
                user_ids = user_encoder.inverse_transform(user_ids_cy)
            if item_encoder is not None:
                item_ids = item_encoder.inverse_transform(item_ids_cy)

            # 5. Validasi singkat hasil prediksi (jangan sampai NULL)
            if len(predictions) > 0 and np.isnan(predictions).all():
                raise ValueError("Prediction Output Error: All predicted scores returned as NULL/NaN.")
            if len(predictions) > 0 and np.all(predictions == 0):
                logger.warning("Prediction Warning: All predicted scores returned exactly 0.0")
            return predictions

        except Exception as fatal_err:
            logger.error(
                f"Execution failed inside AryColBringPredictor.predict: {str(fatal_err)}", 
                exc_info=True)
            raise ValueError(str(fatal_err)) from fatal_err


    def predict_rank(self,
                    test_interactions   : sp.spmatrix,
                    train_interactions  : Optional[sp.spmatrix] = None,
                    item_features       : Optional[sp.spmatrix] = None,
                    user_features       : Optional[sp.spmatrix] = None,
                    num_threads         : int  = 1,
                    check_intersections : bool = True,
                   ) -> sp.csr_matrix:
        """
        Compute item ranks for all test-positive interactions.
        Each entry in the returned matrix holds the 0-based rank of that
        test positive among all items (lower = better).  Training positives
        supplied via ``train_interactions`` are excluded from the ranking
        denominator.
        test_interactions  : sparse matrix [n_users x n_items]
        train_interactions : optional sparse matrix (positives to exclude)
        item_features      : optional CSR feature matrix
        user_features      : optional CSR feature matrix
        num_threads        : OpenMP thread count (>= 1)
        check_intersections: raise if test / train share any interaction

        The Returns is scipy.sparse.csr_matrix of the same sparsity pattern as
        test_interactions, where ``.data`` holds each item's 0-based rank.
        """
        logger.debug("num_threads = %d", num_threads)
        self._check_initialized()
        assert num_threads >= 1, "num_threads must be >= 1"
        if check_intersections:
            self._check_test_train_intersections(
            test_interactions, train_interactions)
        n_users, n_items = test_interactions.shape
        user_features, item_features = self._construct_feature_matrices(
                                       n_users, n_items, 
                                       user_features, item_features)
        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError(
            "item_features column count does not match embedding rows: "
            f"{item_features.shape[1]} vs {self.item_embeddings.shape[0]}")
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError(
            "user_features column count does not match embedding rows: "
            f"{user_features.shape[1]} vs {self.user_embeddings.shape[0]}")
        test_interactions = test_interactions.tocsr()
        test_interactions = self._to_cython_dtype(test_interactions)
        if train_interactions is None:
            train_interactions = sp.csr_matrix((n_users, n_items),
                                               dtype = cydtype)
        else:
            train_interactions = train_interactions.tocsr()
            train_interactions = self._to_cython_dtype(train_interactions)
        ranks = sp.csr_matrix((np.zeros_like(test_interactions.data),
                               test_interactions.indices,
                               test_interactions.indptr),
                               shape = test_interactions.shape)
        predict_ranks(CSRMatrix(item_features),
                      CSRMatrix(user_features),
                      CSRMatrix(test_interactions),
                      CSRMatrix(train_interactions),
                      ranks.data,
                      self._get_model_data(),
                      num_threads)
        return ranks


    # ── training methods — not implemented on the predictor ──────────────────
    def fit(self, *args, **kwargs) -> "AryColBringPredictor":
        """Not available on the predictor. Use ``AryColBringTrainer``."""
        raise NotImplementedError(
        """AryColBringPredictor does not support training.
           Use AryColBringTrainer to fit a model, then load the
           resulting embeddings into this predictor for serving.""")

    def fit_partial(self, *args, **kwargs) -> "AryColBringPredictor":
        """Not available on the predictor. Use ``AryColBringTrainer``."""
        raise NotImplementedError(
        """AryColBringPredictor does not support training.
           Use AryColBringTrainer to fit a model, then load the
           resulting embeddings into this predictor for serving.""")


if __name__ == '__main__':
    pass
