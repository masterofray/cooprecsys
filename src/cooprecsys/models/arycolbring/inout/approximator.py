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

import psutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from .scaffold import AryColBringBase, cydtype
from ..CLproximity import (CSRMatrix,
                         predict_ranks,
                         predict_arycolbring)
from ....configs import _cfg, logger, verbose


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


    def Warn_RAM_Heavy(self, 
                       n_u: int,
                       n_i: int,
                      ) -> None:
        """
        Soft memory-safety check for predict()'s cross-join mode (n_u != n_i,
        both > 1 — see the `cross_join` branch and the native Cython
        ``predict_arycolbring(..., cross_join=True)`` kernel). Never blocks
        execution — only logs a warning — since a caller may deliberately
        want a large report batch.

        Since the cross-join grid is computed natively inside the Cython
        kernel (positional indexing, no Python-side repeat/tile of the id
        arrays), the only large allocation on the Python side is the
        ``predictions`` output buffer itself (float32, length n_u * n_i).
        That's what this estimates — compared against a fraction of
        *currently available* system RAM, so the check adapts to whatever
        machine the package happens to run on (laptop vs. CI runner vs.
        production server) instead of a fixed pair-count ceiling.

        The fraction is read from the shared package config
        (``_cfg['predict']['cross_join_max_memory_fraction']``) so it can be
        tuned per-deployment without touching this module. If that config key
        is absent, a documented fallback of 0.25 (25% of available RAM) is
        used. If ``psutil`` isn't installed, the RAM figure can't be read at
        all, so the check is skipped entirely (debug-logged, not a hard
        dependency of the package).
        """
        n_pairs   = n_u * n_i
        est_bytes = n_pairs * np.dtype(np.float32).itemsize  # predictions buffer only

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
                "~%.1f MB estimated for the predictions buffer). This exceeds %.0f%% "
                "of currently available RAM (%.1f MB available right now). This may "
                "be unintentional and could cause heavy memory pressure or an OOM kill.",
                n_pairs, n_u, n_i, est_bytes / (1024 ** 2),
                max_fraction * 100, available / (1024 ** 2))


    @staticmethod
    def build_pairs(user_ids   : Union[int, list, np.ndarray],
                    item_ids   : Union[int, list, np.ndarray],
                    cross_join : bool,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arrange the (user_id, item_id) label pairs/vectors that correspond,
        in the same order, to the score array ``predict()`` produces for a
        given mode. Pure label-arranging utility — no embeddings/features
        involved — so it's cheap to call on its own to preview which pairs
        a given call would score, or to line raw ids up against a scores
        array returned separately.

        user_ids   : int, list, or ndarray of raw user id labels (any dtype:
                     numeric or string — this function never encodes them).
        item_ids   : int, list, or ndarray of raw item id labels.
        cross_join : False -> strict pairwise. Requires len(user_ids) ==
                              len(item_ids), OR one side of length 1 (which
                              is broadcast against the other, matching
                              predict()'s own scalar-broadcast behaviour).
                              Returns two arrays of that length.
                     True  -> all-to-all grid: every user paired with every
                              item. Returns two arrays of length
                              len(user_ids) * len(item_ids), row-major
                              (user 0 vs every item, then user 1 vs every
                              item, ...) — this exact ordering matches what
                              ``predict_arycolbring(..., cross_join=True)``
                              produces, so zipping this output 1:1 with
                              predict()'s score array is always correct.

        Returns (paired_user_ids, paired_item_ids) two aligned 1-D ndarrays.
        """
        user_ids = np.atleast_1d(np.asarray(user_ids))
        item_ids = np.atleast_1d(np.asarray(item_ids))
        n_u, n_i = len(user_ids), len(item_ids)
        if cross_join:
            return np.repeat(user_ids, n_i), np.tile(item_ids, n_u)

        if n_u == n_i:
            return user_ids, item_ids
        if n_u == 1:
            return np.repeat(user_ids, n_i), item_ids
        if n_i == 1:
            return user_ids, np.repeat(item_ids, n_u)

        raise ValueError(
            f"Cannot form strict pairwise pairs from user_ids length ({n_u}) "
            f"and item_ids length ({n_i}); lengths must match (or one side "
            f"must be a single scalar to broadcast). Pass cross_join=True "
            f"for an all-to-all grid instead.")


    # ── public inference interface ────────────────────────
    def predict(self,
                user_ids      : Union[int, list, np.ndarray],
                item_ids      : Union[list, np.ndarray],
                item_features : Optional[sp.spmatrix] = None,
                user_features : Optional[sp.spmatrix] = None,
                num_threads   : int = 1,
                cross_join    : Optional[bool] = None,
               ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Compute prediction scores for users x items.
        user_ids      : int, list, or ndarray of raw user id labels
        item_ids      : int, list, or ndarray of raw item id labels
        item_features : optional CSR matrix [n_items x n_item_features]
        user_features : optional CSR matrix [n_users x n_user_features]
        num_threads   : OpenMP thread count (>= 1)
        cross_join    : bool or None, default None (auto-detect):
            None  -> auto: strict pairwise if len(user_ids) == len(item_ids), or if
                     either side is a scalar / length-1 (broadcast against the other,
                     unchanged legacy behaviour). All-to-all cross-join if both sides
                     have length > 1 and differ (e.g. a reporting batch).
            False -> force strict pairwise. Raises ValueError if the lengths can't be
                     matched (equal, or one side length 1) — use this to make a caller's
                     intent explicit / fail loudly instead of silently cross-joining.
            True  -> force an all-to-all grid, even when len(user_ids) == len(item_ids).

        Indexing convention (unaffected by cross_join, always the same rule):
            Pairwise mode  -> user_ids[i] / item_ids[i] are literal row indices into
                              user_features / item_features (classic LightFM-style
                              convention; the feature matrix must be sized/aligned to
                              the raw id range, e.g. via sp.identity(max_id + 1)).
            Cross-join mode-> row indices are POSITIONAL (the i-th requested user -> row
                              i, the j-th requested item -> row j), NOT the raw id
                              values. This lets the caller pass a compact feature matrix
                              sized exactly to the N users / M items requested, even when
                              the raw ids themselves are sparse/non-contiguous (e.g.
                              `trainer.py`'s reporting batch, where `self._user_ids` /
                              `self._item_ids` from `fileload_interactions` are the
                              *original* raw ids, not local 0-based indices). The N x M
                              grid is computed NATIVELY inside the Cython kernel
                              (predict_arycolbring(..., cross_join=True)) — no giant N*M
                              id arrays are materialised on the Python side.

        Return type depends on the EFFECTIVE mode (explicit or auto-detected):
            pairwise   -> np.ndarray, shape (n_pairs,), raw dot-product scores
                          (higher = more relevant). Unchanged from before — existing
                          callers (evaluation metrics, etc.) are unaffected.
            cross-join -> pandas.DataFrame with columns ['user_id', 'item_id', 'score'],
                          one row per pair, in row-major order (user 0 vs every item,
                          then user 1 vs every item, ...). Ready for e.g.
                          `predict(...).to_dict(orient='records')`.
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
                if cross_join:
                    return pd.DataFrame({'user_id': [], 'item_id': [], 'score': []})
                return np.array([], dtype=np.float32)

            # 2. Label Encoder untuk string, biarkan numeric as-is.
            #    Encode pada array asli (n_u / n_i, belum di-expand) — baik untuk mode
            #    pairwise/broadcast maupun cross-join, ini tetap murah karena tidak pernah
            #    memproses ukuran N*M.
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

            # 3. Resolve mode efektif: hormati override eksplisit `cross_join` kalau
            #    diberikan; kalau None, auto-detect dari panjang seperti sebelumnya.
            if cross_join is None:
                effective_cross_join = (n_u != n_i) and n_u > 1 and n_i > 1
            else:
                effective_cross_join = bool(cross_join)

            if effective_cross_join:
                logger.debug(
                    "predict(): cross_join mode (n_u=%d, n_i=%d) -> %d-pair grid, "
                    "positional indexing", n_u, n_i, n_u * n_i)
                self.Warn_RAM_Heavy(n_u, n_i)
            elif n_u != n_i:
                # Strict pairwise diminta (eksplisit atau auto), tapi panjang beda.
                # Boleh kalau salah satunya scalar-like (broadcast); selain itu invalid.
                if n_u == 1:
                    user_ids_cy = np.repeat(user_ids_cy, n_i)
                elif n_i == 1:
                    item_ids_cy = np.repeat(item_ids_cy, n_u)
                else:
                    raise ValueError(
                        f"cross_join=False but user_ids length ({n_u}) != item_ids "
                        f"length ({n_i}) and neither side is a single scalar to "
                        f"broadcast. Pass cross_join=True (or leave cross_join=None "
                        f"to auto-detect) for an all-to-all grid.")

            # Guardrails pencegah segmentation fault OpenMP (sanity check internal,
            # seharusnya selalu lolos setelah resolusi mode di atas)
            if not effective_cross_join and len(user_ids_cy) != len(item_ids_cy):
                raise ValueError(f"user_ids length ({len(user_ids_cy)}) != item_ids length ({len(item_ids_cy)})")
            if num_threads < 1:
                raise ValueError("num_threads must be >= 1")

            if user_ids_cy.min() < 0 or item_ids_cy.min() < 0:
                raise ValueError("Negative user_id or item_id found after encoding.")

            # 4. Prediksi via backend Cython (predict_arycolbring).
            #    Pairwise/broadcast: sizing feature matrix mengikuti raw id tertinggi (konvensi lama).
            #    Cross-join: sizing feature matrix POSITIONAL, persis n_u x n_i — cocok dengan
            #    feature matrix custom (mis. dari trainer.py) yang sengaja dibuat kompak untuk
            #    batch ini saja, walau raw id-nya sparse/non-contiguous.
            if effective_cross_join:
                n_users = n_u
                n_items = n_i
            else:
                n_users = int(user_ids_cy.max()) + 1
                n_items = int(item_ids_cy.max()) + 1
            user_features, item_features = self._construct_feature_matrices(
                n_users, n_items, user_features, item_features)
            n_predictions = (n_u * n_i) if effective_cross_join else len(user_ids_cy)
            predictions = np.empty(n_predictions, dtype = np.float32)
            predict_arycolbring(CSRMatrix(item_features),
                                CSRMatrix(user_features),
                                user_ids_cy,
                                item_ids_cy,
                                predictions,
                                self._get_model_data(),
                                num_threads,
                                effective_cross_join,
                                verbose)

            # 5. Decoder label pada column yang telah di-encode sebelumnya
            if user_encoder is not None:
                user_ids = user_encoder.inverse_transform(user_ids_cy)
            if item_encoder is not None:
                item_ids = item_encoder.inverse_transform(item_ids_cy)

            # 6. Validasi singkat hasil prediksi
            if len(predictions) > 0 and np.isnan(predictions).all():
                raise ValueError("Prediction Output Error: All predicted scores returned as NULL/NaN.")
            if len(predictions) > 0 and np.all(predictions == 0):
                logger.warning("Prediction Warning: All predicted scores returned exactly 0.0")

            # 7. Return type mengikuti mode efektif
            if effective_cross_join:
                paired_user_ids, paired_item_ids = self.build_pairs(
                                                   user_ids, item_ids, 
                                                   cross_join = True)
                return pd.DataFrame({
                    'user_id': paired_user_ids,
                    'item_id': paired_item_ids,
                    'score'  : predictions})
            return predictions

        except Exception as fatal_err:
            logger.error(
                f"Execution failed inside AryColBringPredictor.predict: {str(fatal_err)}", 
                exc_info = True)
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
