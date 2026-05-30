#!/usr/bin/env python3

"""
arycolbring.model
~~~~~~~~~~~~~~~~~
Python API for the AryColBring collaborative-filtering model.

All heavy computation is delegated to the compiled Cython kernels in
``arycolbring.cy``.  This module is responsible for:
  - Parameter validation and type coercion
  - Embedding initialisation
  - Feature-matrix construction (sparse identity fall-back)
  - Epoch loops with tqdm progress bars
  - Logging at DEBUG level throughout
  - Sklearn-style ``fit`` / ``fit_partial`` / ``predict`` / ``predict_rank``
    interface
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


import gc
import logging
from configparser import ConfigParser
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm

from .data_utils import validate_sparse_matrix
from .
 import (
    CSRMatrix,
    FastAryColBring,
    fit_logistic,
    fit_warp,
    fit_bpr,
    fit_warp_kos,
    predict_arycolbring,
    predict_ranks)

# ── logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── config ───────────────────────────────────────────────────────────────────
_cfg = ConfigParser()
_cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

TQDM_COLOUR  = _cfg.get("tqdm",  "colour",  fallback="#05ad46")
TQDM_NCOLS   = _cfg.getint("tqdm", "ncols",  fallback=80)
CYTHON_DTYPE = np.float32


__all__ = ["AryColBring"]


class AryColBring:
    """
    Ultra-optimised user-to-item collaborative filtering model.

    Supports four training objectives:
      - ``"logistic"``  — point-wise logistic regression
      - ``"warp"``      — Weighted Approximate-Rank Pairwise loss
      - ``"bpr"``       — Bayesian Personalised Ranking
      - ``"warp-kos"``  — WARP k-th Order Statistic variant

    Two optimiser schedules:
      - ``"adagrad"``   — per-feature adaptive learning rate
      - ``"adadelta"``  — gradient-squared momentum adaptive LR

    Parameters
    ----------
    no_components   : int   — number of latent dimensions (default 10)
    k               : int   — k for warp-kos anchor selection (default 5)
    n               : int   — n samples for kos anchor pool (default 10)
    learning_schedule: str  — "adagrad" | "adadelta"
    loss            : str   — "logistic" | "warp" | "bpr" | "warp-kos"
    learning_rate   : float — base learning rate (default 0.05)
    rho             : float — Adadelta decay factor ∈ (0, 1) (default 0.95)
    epsilon         : float — numerical stability term (default 1e-6)
    item_alpha      : float — L2 regularisation weight for items (default 0.0)
    user_alpha      : float — L2 regularisation weight for users (default 0.0)
    max_sampled     : int   — max negative samples per positive (default 10)
    random_state    : int | np.random.RandomState | None
    """

    # ── constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        no_components:    int   = 10,
        k:                int   = 5,
        n:                int   = 10,
        learning_schedule: str  = "adagrad",
        loss:             str   = "logistic",
        learning_rate:    float = 0.05,
        rho:              float = 0.95,
        epsilon:          float = 1e-6,
        item_alpha:       float = 0.0,
        user_alpha:       float = 0.0,
        max_sampled:      int   = 10,
        random_state             = None,
    ):
        logger.debug(
            "AryColBring.__init__: loss=%s schedule=%s no_components=%d",
            loss, learning_schedule, no_components,
        )

        # ── parameter validation ─────────────────────────────────────────────
        if item_alpha < 0.0:
            raise ValueError("item_alpha must be ≥ 0.0")
        if user_alpha < 0.0:
            raise ValueError("user_alpha must be ≥ 0.0")
        if no_components <= 0:
            raise ValueError("no_components must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if n <= 0:
            raise ValueError("n must be > 0")
        if not (0 < rho < 1):
            raise ValueError("rho must be in (0, 1)")
        if epsilon < 0:
            raise ValueError("epsilon must be ≥ 0")
        if max_sampled < 1:
            raise ValueError("max_sampled must be a positive integer")
        if learning_schedule not in ("adagrad", "adadelta"):
            raise ValueError(
                f"learning_schedule must be 'adagrad' or 'adadelta', "
                f"got '{learning_schedule}'"
            )
        if loss not in ("logistic", "warp", "bpr", "warp-kos"):
            raise ValueError(
                f"loss must be one of 'logistic','warp','bpr','warp-kos', "
                f"got '{loss}'"
            )

        self._loss              = loss
        self._learning_schedule = learning_schedule
        self._no_components     = no_components
        self._learning_rate     = learning_rate
        self._k                 = int(k)
        self._n                 = int(n)
        self._rho               = rho
        self._epsilon           = epsilon
        self._max_sampled       = max_sampled
        self._item_alpha        = item_alpha
        self._user_alpha        = user_alpha

        if random_state is None:
            self._random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        elif isinstance(random_state, int):
            self._random_state = np.random.RandomState(seed=random_state)
        else:
            raise TypeError(
                "random_state must be None, an int, or np.random.RandomState"
            )

        self._reset_state()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def loss(self) -> str:
        return self._loss

    @loss.setter
    def loss(self, value: str) -> None:
        if value not in ("logistic", "warp", "bpr", "warp-kos"):
            raise ValueError(f"Invalid loss '{value}'")
        self._loss = value

    @property
    def no_components(self) -> int:
        return self._no_components

    @no_components.setter
    def no_components(self, value: int) -> None:
        if value <= 0:
            raise ValueError("no_components must be > 0")
        self._no_components = int(value)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        self._learning_rate = float(value)

    @property
    def item_alpha(self) -> float:
        return self._item_alpha

    @item_alpha.setter
    def item_alpha(self, value: float) -> None:
        if value < 0:
            raise ValueError("item_alpha must be ≥ 0")
        self._item_alpha = float(value)

    @property
    def user_alpha(self) -> float:
        return self._user_alpha

    @user_alpha.setter
    def user_alpha(self, value: float) -> None:
        if value < 0:
            raise ValueError("user_alpha must be ≥ 0")
        self._user_alpha = float(value)

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` or ``fit_partial`` has been called."""
        return self.item_embeddings is not None

    @property
    def learning_schedule(self) -> str:
        return self._learning_schedule

    # ── internal state helpers ────────────────────────────────────────────────

    def _reset_state(self) -> None:
        logger.debug("AryColBring._reset_state: clearing all embeddings")
        self.item_embeddings         = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum = None
        self.item_biases             = None
        self.item_bias_gradients     = None
        self.item_bias_momentum      = None

        self.user_embeddings         = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum = None
        self.user_biases             = None
        self.user_bias_gradients     = None
        self.user_bias_momentum      = None

    def _check_initialized(self) -> None:
        attrs = [
            "item_embeddings", "item_embedding_gradients",
            "item_embedding_momentum", "item_biases",
            "item_bias_gradients", "item_bias_momentum",
            "user_embeddings", "user_embedding_gradients",
            "user_embedding_momentum", "user_biases",
            "user_bias_gradients", "user_bias_momentum",
        ]
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(
                    "Model is not yet fitted.  Call fit() or fit_partial() first."
                )

    def _initialize(self,
                    no_components:     int,
                    no_item_features:  int,
                    no_user_features:  int) -> None:
        logger.debug(
            "_initialize: no_components=%d n_item_feat=%d n_user_feat=%d",
            no_components, no_item_features, no_user_features,
        )

        # Item embeddings ± 0.5 / no_components, uniform
        self.item_embeddings = (
            (self._random_state.rand(no_item_features, no_components) - 0.5)
            / no_components
        ).astype(CYTHON_DTYPE)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum  = np.zeros_like(self.item_embeddings)
        self.item_biases              = np.zeros(no_item_features, dtype=CYTHON_DTYPE)
        self.item_bias_gradients      = np.zeros_like(self.item_biases)
        self.item_bias_momentum       = np.zeros_like(self.item_biases)

        # User embeddings
        self.user_embeddings = (
            (self._random_state.rand(no_user_features, no_components) - 0.5)
            / no_components
        ).astype(CYTHON_DTYPE)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum  = np.zeros_like(self.user_embeddings)
        self.user_biases              = np.zeros(no_user_features, dtype=CYTHON_DTYPE)
        self.user_bias_gradients      = np.zeros_like(self.user_biases)
        self.user_bias_momentum       = np.zeros_like(self.user_biases)

        # Adagrad initialises accumulators to 1 to avoid divide-by-zero
        if self._learning_schedule == "adagrad":
            self.item_embedding_gradients += 1
            self.item_bias_gradients      += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients      += 1

    @staticmethod
    def _to_cython_dtype(mat: sp.spmatrix) -> sp.spmatrix:
        if mat.dtype != CYTHON_DTYPE:
            return mat.astype(CYTHON_DTYPE)
        return mat

    def _construct_feature_matrices(
        self,
        n_users: int,
        n_items: int,
        user_features: Optional[sp.spmatrix],
        item_features: Optional[sp.spmatrix],
    ):
        logger.debug(
            "_construct_feature_matrices: n_users=%d n_items=%d", n_users, n_items
        )

        if user_features is None:
            user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
        else:
            item_features = item_features.tocsr()

        if n_users > user_features.shape[0]:
            raise ValueError(
                f"n_users ({n_users}) exceeds user_features rows "
                f"({user_features.shape[0]})"
            )
        if n_items > item_features.shape[0]:
            raise ValueError(
                f"n_items ({n_items}) exceeds item_features rows "
                f"({item_features.shape[0]})"
            )

        if self.user_embeddings is not None:
            if self.user_embeddings.shape[0] < user_features.shape[1]:
                raise ValueError(
                    "user_features specifies more columns than embedding matrix rows: "
                    f"{user_features.shape[1]} vs {self.user_embeddings.shape[0]}"
                )
        if self.item_embeddings is not None:
            if self.item_embeddings.shape[0] < item_features.shape[1]:
                raise ValueError(
                    "item_features specifies more columns than embedding matrix rows: "
                    f"{item_features.shape[1]} vs {self.item_embeddings.shape[0]}"
                )

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)
        return user_features, item_features

    def _get_positives_lookup_matrix(self, interactions: sp.coo_matrix) -> sp.csr_matrix:
        mat = interactions.tocsr()
        if not mat.has_sorted_indices:
            mat.sort_indices()
        return mat

    def _process_sample_weight(
        self,
        interactions:  sp.coo_matrix,
        sample_weight: Optional[sp.coo_matrix],
    ) -> np.ndarray:
        if sample_weight is not None:
            if self._loss == "warp-kos":
                raise NotImplementedError(
                    "Sample weights are not supported with warp-kos loss."
                )
            if not isinstance(sample_weight, sp.coo_matrix):
                raise TypeError("sample_weight must be a scipy COO matrix.")
            if sample_weight.shape != interactions.shape:
                raise ValueError(
                    "sample_weight and interactions must have the same shape."
                )
            if not (np.array_equal(interactions.row, sample_weight.row)
                    and np.array_equal(interactions.col, sample_weight.col)):
                raise ValueError(
                    "sample_weight and interactions entries must be in the same order."
                )
            data = sample_weight.data
            if data.dtype != CYTHON_DTYPE:
                data = data.astype(CYTHON_DTYPE)
            return data
        else:
            if np.array_equiv(interactions.data, 1.0):
                return interactions.data
            return np.ones_like(interactions.data, dtype=CYTHON_DTYPE)

    def _get_model_data(self) -> FastAryColBring:
        return FastAryColBring(
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
            self._no_components,
            int(self._learning_schedule == "adadelta"),
            self._learning_rate,
            self._rho,
            self._epsilon,
            self._max_sampled,
        )

    def _check_finite(self) -> None:
        for name, arr in [
            ("item_embeddings", self.item_embeddings),
            ("item_biases",     self.item_biases),
            ("user_embeddings", self.user_embeddings),
            ("user_biases",     self.user_biases),
        ]:
            if not np.isfinite(np.sum(arr)):
                raise ValueError(
                    f"Non-finite values detected in '{name}' after update. "
                    "Try reducing learning_rate or normalising input features."
                )

    def _check_input_finite(self, data: np.ndarray, name: str = "input") -> None:
        if not np.isfinite(np.sum(data)):
            raise ValueError(
                f"Non-finite values detected in '{name}'. "
                "Check your input for NaN or Inf."
            )

    def _check_test_train_intersections(
        self,
        test_mat:  sp.spmatrix,
        train_mat: Optional[sp.spmatrix],
    ) -> None:
        if train_mat is not None:
            n = test_mat.multiply(train_mat).nnz
            if n:
                raise ValueError(
                    f"test and train matrices share {n} interactions. "
                    "This will produce optimistic evaluation results. "
                    "Fix your data split before evaluating."
                )

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _epoch_iterator(n_epochs: int, verbose: bool):
        if verbose:
            return tqdm(range(n_epochs), desc="Epoch",
                        colour=TQDM_COLOUR, ncols=TQDM_NCOLS)
        return range(n_epochs)

    # ── public training interface ─────────────────────────────────────────────

    def fit(
        self,
        interactions:   sp.spmatrix,
        user_features:  Optional[sp.spmatrix] = None,
        item_features:  Optional[sp.spmatrix] = None,
        sample_weight:  Optional[sp.coo_matrix] = None,
        epochs:         int  = 1,
        num_threads:    int  = 1,
        verbose:        bool = False,
    ) -> "AryColBring":
        """
        Fit the model from scratch (discards any previous state).

        Parameters
        ----------
        interactions  : sparse matrix [n_users × n_items]
                        Non-zero values are positive interactions.
        user_features : optional CSR matrix [n_users × n_user_features]
        item_features : optional CSR matrix [n_items × n_item_features]
        sample_weight : optional COO matrix matching interactions shape
        epochs        : number of training epochs
        num_threads   : OpenMP thread count (≥ 1)
        verbose       : show tqdm epoch bar

        Returns
        -------
        self
        """
        logger.debug("AryColBring.fit: epochs=%d num_threads=%d", epochs, num_threads)
        self._reset_state()
        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions:   sp.spmatrix,
        user_features:  Optional[sp.spmatrix] = None,
        item_features:  Optional[sp.spmatrix] = None,
        sample_weight:  Optional[sp.coo_matrix] = None,
        epochs:         int  = 1,
        num_threads:    int  = 1,
        verbose:        bool = False,
    ) -> "AryColBring":
        """
        Partially fit the model (preserves previous embedding state).
        Suitable for incremental / online learning.

        Parameters mirror ``fit``.
        """
        logger.debug(
            "AryColBring.fit_partial: loss=%s epochs=%d num_threads=%d",
            self._loss, epochs, num_threads,
        )

        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")
        if epochs < 1:
            raise ValueError("epochs must be ≥ 1")

        # Convert to COO
        interactions = interactions.tocoo()
        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        validate_sparse_matrix(interactions, "interactions")

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        # Validate all inputs for finiteness
        for arr, lbl in [
            (user_features.data,  "user_features"),
            (item_features.data,  "item_features"),
            (interactions.data,   "interactions"),
            (sample_weight_data,  "sample_weight"),
        ]:
            self._check_input_finite(arr, lbl)

        if self.item_embeddings is None:
            self._initialize(
                self._no_components,
                item_features.shape[1],
                user_features.shape[1],
            )

        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError(
                f"item_features has {item_features.shape[1]} columns but "
                f"embedding has {self.item_embeddings.shape[0]} rows."
            )
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError(
                f"user_features has {user_features.shape[1]} columns but "
                f"embedding has {self.user_embeddings.shape[0]} rows."
            )

        # Positives lookup matrix (sorted indices required for bsearch)
        if self._loss in ("warp", "bpr", "warp-kos"):
            positives_lookup = CSRMatrix(
                self._get_positives_lookup_matrix(interactions)
            )

        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)

        with tqdm(total=epochs, desc="Training",
                  colour=TQDM_COLOUR, ncols=TQDM_NCOLS,
                  disable=not verbose) as pbar:

            for epoch in self._epoch_iterator(epochs, verbose=False):
                self._random_state.shuffle(shuffle_indices)
                model_data = self._get_model_data()

                self._run_epoch(
                    item_features, user_features,
                    interactions, sample_weight_data,
                    shuffle_indices, num_threads,
                    self._loss,
                    positives_lookup if self._loss in ("warp", "bpr", "warp-kos")
                    else None,
                    model_data,
                )

                self._check_finite()
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch + 1, "loss": self._loss})
                logger.debug("fit_partial: epoch %d/%d done", epoch + 1, epochs)

        gc.collect()
        logger.debug("fit_partial: training complete")
        return self

    def _run_epoch(
        self,
        item_features:    sp.csr_matrix,
        user_features:    sp.csr_matrix,
        interactions:     sp.coo_matrix,
        sample_weight:    np.ndarray,
        shuffle_indices:  np.ndarray,
        num_threads:      int,
        loss:             str,
        positives_lookup,
        model_data:       FastAryColBring,
    ) -> None:
        """Dispatch to the correct Cython training kernel."""
        logger.debug("_run_epoch: loss=%s", loss)

        cy_item = CSRMatrix(item_features)
        cy_user = CSRMatrix(user_features)

        if loss == "warp":
            fit_warp(
                cy_item, cy_user, positives_lookup,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads, self._random_state,
            )
        elif loss == "bpr":
            fit_bpr(
                cy_item, cy_user, positives_lookup,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads, self._random_state,
            )
        elif loss == "warp-kos":
            fit_warp_kos(
                cy_item, cy_user, positives_lookup,
                interactions.row, shuffle_indices,
                model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                self._k, self._n,
                num_threads, self._random_state,
            )
        else:  # logistic
            fit_logistic(
                cy_item, cy_user,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads,
            )

    # ── public inference interface ────────────────────────────────────────────

    def predict(
        self,
        user_ids:       Union[int, list, np.ndarray],
        item_ids:       Union[list, np.ndarray],
        item_features:  Optional[sp.spmatrix] = None,
        user_features:  Optional[sp.spmatrix] = None,
        num_threads:    int = 1,
    ) -> np.ndarray:
        """
        Compute prediction scores for (user_id, item_id) pairs.

        Parameters
        ----------
        user_ids     : int, list, or int32 ndarray
        item_ids     : list or int32 ndarray (same length as user_ids)
        item_features: optional CSR matrix [n_items × n_item_features]
        user_features: optional CSR matrix [n_users × n_user_features]
        num_threads  : OpenMP thread count

        Returns
        -------
        np.ndarray float32, shape (n_pairs,)
        """
        logger.debug("AryColBring.predict: num_threads=%d", num_threads)
        self._check_initialized()

        if isinstance(user_ids, int):
            user_ids = np.repeat(np.int32(user_ids), len(item_ids))

        # Convert Python list/tuple → numpy C array
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
                f"user_ids length ({len(user_ids)}) != item_ids length ({len(item_ids)})"
            )
        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")
        if user_ids.min() < 0 or item_ids.min() < 0:
            raise ValueError(
                "Negative user_id or item_id found.  Check for overflow or bad input."
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

        Parameters
        ----------
        test_interactions  : sparse matrix [n_users × n_items]
        train_interactions : optional sparse matrix (positives to exclude)
        item_features      : optional CSR feature matrix
        user_features      : optional CSR feature matrix
        num_threads        : OpenMP thread count
        check_intersections: raise if test/train share interactions

        Returns
        -------
        scipy.sparse.csr_matrix  of same structure as test_interactions,
        where .data holds the rank of each test positive (0-based).
        """
        logger.debug("AryColBring.predict_rank: num_threads=%d", num_threads)
        self._check_initialized()

        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")

        if check_intersections:
            self._check_test_train_intersections(test_interactions, train_interactions)

        n_users, n_items = test_interactions.shape
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError("item_features column count mismatches embedding rows.")
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError("user_features column count mismatches embedding rows.")

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
            (np.zeros_like(test_interactions.data),
             test_interactions.indices,
             test_interactions.indptr),
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

    # ── representation getters ────────────────────────────────────────────────

    def get_item_representations(
        self,
        features: Optional[sp.spmatrix] = None,
    ):
        """
        Return (biases, embeddings) for items.

        If *features* is None, returns the raw embedding arrays directly.
        Otherwise projects through the feature matrix.
        """
        self._check_initialized()
        if features is None:
            return self.item_biases, self.item_embeddings
        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)
        return features * self.item_biases, features * self.item_embeddings

    def get_user_representations(
        self,
        features: Optional[sp.spmatrix] = None,
    ):
        """
        Return (biases, embeddings) for users.

        If *features* is None, returns the raw embedding arrays directly.
        Otherwise projects through the feature matrix.
        """
        self._check_initialized()
        if features is None:
            return self.user_biases, self.user_embeddings
        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)
        return features * self.user_biases, features * self.user_embeddings

    # ── sklearn-style get/set params ──────────────────────────────────────────

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return a dict of constructor parameters (sklearn API compatible)."""
        return {
            "loss":              self._loss,
            "learning_schedule": self._learning_schedule,
            "no_components":     self._no_components,
            "learning_rate":     self._learning_rate,
            "k":                 self._k,
            "n":                 self._n,
            "rho":               self._rho,
            "epsilon":           self._epsilon,
            "max_sampled":       self._max_sampled,
            "item_alpha":        self._item_alpha,
            "user_alpha":        self._user_alpha,
            "random_state":      self._random_state,
        }

    def set_params(self, **params) -> "AryColBring":
        """Set parameters (sklearn API compatible)."""
        valid = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}' for AryColBring.  "
                    f"Valid params: {sorted(valid)}"
                )
            setattr(self, f"_{key}", value)
        return self

    # ── string representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AryColBring("
            f"loss='{self._loss}', "
            f"no_components={self._no_components}, "
            f"learning_schedule='{self._learning_schedule}', "
            f"learning_rate={self._learning_rate}, "
            f"item_alpha={self._item_alpha}, "
            f"user_alpha={self._user_alpha}"
            f")"
        )
