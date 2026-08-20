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
scaffold.py
_______________________________________________________________
Abstract base class for the AryColBring collaborative filtering model.
Owns all shared hyper-parameters, embedding state, helper utilities,
and sklearn-compatible get_params / set_params.  Concrete subclasses
must implement the four public API methods declared here as
`@abstractmethod`.
"""

import numpy as np
import scipy.sparse as sp
from   tqdm.auto import tqdm
from   pathlib   import Path
from   abc       import ABC, abstractmethod
from   typing    import Any, Dict, Optional, Union

from ..assist      import validate_sparse_matrix
from cooprecsys.models.arycolbring.CLproximity import CSRMatrix, FastAryColBring
from ....configs import _cfg, logger
cydtype = np.float32


class AryColBringBase(ABC):
    """
    Abstract base for the AryColBring collaborative filtering family.
    Holds all hyper-parameters, manages embedding matrices, and provides
    shared internal helpers.  Subclasses must implement:
    - `fit` / `fit_partial` (training contract)
    - `predict` / `predict_rank` (inference contract)

    Parameters
    _____________________________________________________________
    no_components    : int   - latent dimension count (default 10)
    k                : int   - warp-kos anchor count (default 5)
    n                : int   - warp-kos anchor pool size (default 10)
    learning_schedule: str   - "adagrad" | "adadelta"
    loss             : str   - "logistic" | "warp" | "bpr" | "warp-kos"
    learning_rate    : float - base LR (default 0.05)
    rho              : float - Adadelta decay part of (0, 1) (default 0.95)
    epsilon          : float - numerical stability (default 1e-6)
    item_alpha       : float - L2 item regularisation (default 0.0)
    user_alpha       : float - L2 user regularisation (default 0.0)
    max_sampled      : int   - max negatives per positive (default 10)
    random_state     : int | np.random.RandomState | None
    """
    def __init__(
            self,
            no_components:     int   = 10,
            k:                 int   = 5,
            n:                 int   = 10,
            learning_schedule: str   = "adagrad",
            loss:              str   = "logistic",
            learning_rate:     float = 0.05,
            rho:               float = 0.95,
            epsilon:           float = 1e-6,
            item_alpha:        float = 0.0,
            user_alpha:        float = 0.0,
            max_sampled:       int   = 10,
            random_state             = 4,
        ) -> None:
        logger.debug("Initialize loss = %s schedule = %s no_components = %d",
                      loss, learning_schedule, no_components)
        if item_alpha < 0.0:
            raise ValueError("item_alpha must be >= 0.0")
        if user_alpha < 0.0:
            raise ValueError("user_alpha must be >= 0.0")
        if no_components <= 0:
            raise ValueError("no_components must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if n <= 0:
            raise ValueError("n must be > 0")
        if not (0 < rho < 1):
            raise ValueError("rho must be in (0, 1)")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if max_sampled < 1:
            raise ValueError("max_sampled must be a positive integer")
        if learning_schedule not in ("adagrad", "adadelta"):
            raise ValueError(f"learning_schedule must be 'adagrad' or 'adadelta', "
                             f"got '{learning_schedule}'")
        if loss not in ("logistic", "warp", "bpr", "warp-kos"):
            raise ValueError(f"loss must be one of 'logistic','warp','bpr','warp-kos', "
                             f"got '{loss}'")

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
            self._random_state = np.random.RandomState(seed = random_state)
        else:
            raise TypeError(
            "random_state must be None, an int, or np.random.RandomState")
        self._reset_state()


    # ── properties ───────────────────────────
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
        """True once `fit` or `fit_partial` has been called successfully."""
        return self.item_embeddings is not None

    @property
    def learning_schedule(self) -> str:
        return self._learning_schedule



    # ── internal state management ───────────────────────────
    def _reset_state(self) -> None:
        """Zero out all embedding matrices and gradient accumulators."""
        logger.debug("clearing all embeddings")
        self.item_embeddings          = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum  = None
        self.item_biases              = None
        self.item_bias_gradients      = None
        self.item_bias_momentum       = None
        self.user_embeddings          = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum  = None
        self.user_biases              = None
        self.user_bias_gradients      = None
        self.user_bias_momentum       = None

    def _check_initialized(self) -> None:
        """Raise RuntimeError if embeddings have not been initialised yet."""
        attrs = ["item_embeddings", 
                 "item_embedding_gradients",
                 "item_embedding_momentum",
                 "item_biases",
                 "item_bias_gradients",
                 "item_bias_momentum",
                 "user_embeddings",
                 "user_embedding_gradients",
                 "user_embedding_momentum",
                 "user_biases",
                 "user_bias_gradients",
                 "user_bias_momentum"]
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(
                "Model is not yet fitted. Call fit() or fit_partial() first.")

    def _initialize(
            self,
            no_components:    int,
            no_item_features: int,
            no_user_features: int,
        ) -> None:
        """Randomly initialise embedding matrices and zero gradient buffers."""
        logger.debug("no_components = %d n_item_feat = %d n_user_feat = %d",
                      no_components, no_item_features, no_user_features)
        # Item embeddings
        self.item_embeddings = ((self._random_state.rand(
                               no_item_features, no_components) - 0.5)
                               / no_components).astype(cydtype)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum  = np.zeros_like(self.item_embeddings)
        self.item_biases              = np.zeros(no_item_features, dtype=cydtype)
        self.item_bias_gradients      = np.zeros_like(self.item_biases)
        self.item_bias_momentum       = np.zeros_like(self.item_biases)

        # User embeddings: same scheme
        self.user_embeddings          = ((self._random_state.rand(
                                        no_user_features, no_components) - 0.5)
                                        / no_components).astype(cydtype)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum  = np.zeros_like(self.user_embeddings)
        self.user_biases              = np.zeros(no_user_features, dtype=cydtype)
        self.user_bias_gradients      = np.zeros_like(self.user_biases)
        self.user_bias_momentum       = np.zeros_like(self.user_biases)

        # Adagrad: pre-fill gradient accumulators with 1 to avoid divide-by-zero
        if self._learning_schedule == "adagrad":
            self.item_embedding_gradients += 1
            self.item_bias_gradients      += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients      += 1


    # ── static helpers ───────────────────────────
    @staticmethod
    def _to_cython_dtype(mat: sp.spmatrix) -> sp.spmatrix:
        """Cast sparse matrix to cydtype (float32) if needed."""
        if mat.dtype != cydtype:
            return mat.astype(cydtype)
        return mat

    @staticmethod
    def _epoch_iterator(n_epochs: int, verbose: bool):
        """Return a plain range or a tqdm-wrapped range depending on verbosity."""
        if verbose:
            return tqdm(range(n_epochs),
                        desc        = f"Intra-List Diversity@{k}",
                        colour      = _cfg.get('tqdm', 'colour'),
                        ncols       = _cfg.getint('tqdm', 'ncols'),
                        bar_format  = _cfg.get('tqdm', 'BarFormats'),
                        unit        = 'batch',
                        mininterval = 0.1)
        return range(n_epochs)


    # ── feature matrix helpers ────────────────────────────────────────────────
    def _construct_feature_matrices(
            self,
            n_users:       int,
            n_items:       int,
            user_features: Optional[sp.spmatrix],
            item_features: Optional[sp.spmatrix],
        ):
        """Build or validate CSR user / item feature matrices.
           Falls back to identity matrices when features are `None`
           (standard matrix-factorisation mode)."""
        logger.debug("Try to construct Feature: n_users = %d n_items = %d",
                      n_users, n_items)
        if user_features is None:
            user_features = sp.identity(n_users, dtype=cydtype, format="csr")
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items, dtype=cydtype, format="csr")
        else:
            item_features = item_features.tocsr()

        if n_users > user_features.shape[0]:
            raise ValueError(
                f"n_users ({n_users}) exceeds user_features rows "
                f"({user_features.shape[0]})")
        if n_items > item_features.shape[0]:
            raise ValueError(
                f"n_items ({n_items}) exceeds item_features rows "
                f"({item_features.shape[0]})")

        if self.user_embeddings is not None:
            if self.user_embeddings.shape[0] < user_features.shape[1]:
                raise ValueError(
                    "user_features has more columns than embedding matrix rows: "
                    f"{user_features.shape[1]} vs {self.user_embeddings.shape[0]}")
        if self.item_embeddings is not None:
            if self.item_embeddings.shape[0] < item_features.shape[1]:
                raise ValueError(
                    "item_features has more columns than embedding matrix rows: "
                    f"{item_features.shape[1]} vs {self.item_embeddings.shape[0]}")
        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)
        return user_features, item_features


    def _get_positives_lookup_matrix(
            self, 
            interactions: sp.coo_matrix,
        ) -> sp.csr_matrix:
        """Return a sorted-index CSR view of positive interactions."""
        mat = interactions.tocsr()
        if not mat.has_sorted_indices:
            mat.sort_indices()
        return mat


    def _process_sample_weight(self,
                               interactions  : sp.coo_matrix,
                               sample_weight : Optional[sp.coo_matrix],
                              ) -> np.ndarray:
        """
        Validate and return the sample-weight array aligned with interactions.
        If `sample_weight` is `None`, returns a ones vector (uniform weighting).
        """
        if sample_weight is not None:
            if self._loss == "warp-kos":
                raise NotImplementedError(
                "Sample weights are not supported with warp-kos loss.")

            if not isinstance(sample_weight, sp.coo_matrix):
                raise TypeError("sample_weight must be a scipy COO matrix.")

            if sample_weight.shape != interactions.shape:
                raise ValueError(
                "sample_weight and interactions must have the same shape.")

            if not (np.array_equal(interactions.row, sample_weight.row)
            and np.array_equal(interactions.col, sample_weight.col)):
                raise ValueError(
                "sample_weight and interactions entries must be in the same order.")

            data = sample_weight.data
            if data.dtype != cydtype:
                data = data.astype(cydtype)
            return data

        else:
            if np.array_equiv(interactions.data, 1.0):
                return interactions.data
            return np.ones_like(interactions.data, dtype = cydtype)


    def _get_model_data(self) -> FastAryColBring:
        """Pack current embedding state into the Cython FastAryColBring struct."""
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
            self._max_sampled)


    # ── finite-value guards ──────────────────────────────────────
    def _check_finite(self) -> None:
        """Raise ValueError if any embedding or bias contains NaN / Inf."""
        for name, arr in [
            ("item_embeddings", self.item_embeddings),
            ("item_biases",     self.item_biases),
            ("user_embeddings", self.user_embeddings),
            ("user_biases",     self.user_biases)]:
            if not np.isfinite(np.sum(arr)):
                raise ValueError(
                f"Non-finite values detected in '{name}' after update. "
                 "Try reducing learning_rate or normalising input features.")

    def _check_input_finite(
            self,
            data : np.ndarray, 
            name : str = "input",
            ) -> None:
        """Raise ValueError if *data* contains NaN / Inf."""
        if not np.isfinite(np.sum(data)):
            raise ValueError(
            f"Non-finite values detected in '{name}'. "
             "Check your input for NaN or Inf.")

    def _check_test_train_intersections(
            self,
            test_mat:  sp.spmatrix,
            train_mat: Optional[sp.spmatrix],
        ) -> None:
        """Raise ValueError if test and train matrices share any interactions."""
        if train_mat is not None:
            n = test_mat.multiply(train_mat).nnz
            if n:
                raise ValueError(
                f"test and train matrices share {n} interactions. "
                "This produces optimistic evaluation results. "
                "Fix your data split before evaluating.")


    # ── representation getters ────────────────────────────────────────────────
    def get_item_representations(
            self,
            features: Optional[sp.spmatrix] = None,
        ):
        """Return `(biases, embeddings)` for items.
           If *features* is `None`, returns raw embedding arrays directly.
           Otherwise projects through the feature matrix."""
        self._check_initialized()
        if features is None:
            return self.item_biases, self.item_embeddings
        features = sp.csr_matrix(features, dtype=cydtype)
        return features * self.item_biases, features * self.item_embeddings

    def get_user_representations(
            self,
            features: Optional[sp.spmatrix] = None,
        ):
        """
        Return `(biases, embeddings)` for users.
        """
        self._check_initialized()
        if features is None:
            return self.user_biases, self.user_embeddings
        features = sp.csr_matrix(features, dtype = cydtype)
        return features * self.user_biases, features * self.user_embeddings


    # ── sklearn-style get/set params ──────────────────────────────────────────
    def get_params(
        self,
        deep: bool = True) -> Dict[str, Any]:
        """Return constructor parameters as a dict (sklearn API compatible)."""
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
            "random_state":      self._random_state}

    def set_params(self, **params) -> "AryColBringBase":
        """Set parameters in-place (sklearn API compatible)."""
        valid = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                f"Invalid parameter '{key}' for {self.__class__.__name__}. "
                f"Valid params: {sorted(valid)}")
            setattr(self, f"_{key}", value)
        return self


    # ── abstract public API ───────────────────────────────────────────────────
    @abstractmethod
    def fit(self,
            interactions  : sp.spmatrix,
            user_features : Optional[sp.spmatrix] = None,
            item_features : Optional[sp.spmatrix] = None,
            sample_weight : Optional[sp.coo_matrix] = None,
            epochs        : int  = 1,
            num_threads   : int  = 1,
            verbose       : bool = False,
           ) -> "AryColBringBase":
        """Train from scratch, discarding any previous state."""

    @abstractmethod
    def fit_partial(self,
                    interactions  : sp.spmatrix,
                    user_features : Optional[sp.spmatrix] = None,
                    item_features : Optional[sp.spmatrix] = None,
                    sample_weight : Optional[sp.coo_matrix] = None,
                    epochs        : int  = 1,
                    num_threads   : int  = 1,
                    verbose       : bool = False,
                   ) -> "AryColBringBase":
        """Incrementally fit, preserving previous embedding state."""

    @abstractmethod
    def predict(self,
                user_ids      : Union[int, list, np.ndarray],
                item_ids      : Union[list, np.ndarray],
                item_features : Optional[sp.spmatrix] = None,
                user_features : Optional[sp.spmatrix] = None,
                num_threads   : int = 1,
               ) -> np.ndarray:
        """Compute raw prediction scores for (user_id, item_id) pairs."""

    @abstractmethod
    def predict_rank(self,
                     test_interactions:   sp.spmatrix,
                     train_interactions:  Optional[sp.spmatrix] = None,
                     item_features:       Optional[sp.spmatrix] = None,
                     user_features:       Optional[sp.spmatrix] = None,
                     num_threads:         int  = 1,
                     check_intersections: bool = True,
                    ) -> sp.csr_matrix:
        """Return item ranks for all test-positive interactions."""


    # ── string representation ─────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loss              = '{self._loss}', "
            f"no_components     = {self._no_components}, "
            f"learning_schedule = '{self._learning_schedule}', "
            f"learning_rate     = {self._learning_rate}, "
            f"item_alpha        = {self._item_alpha}, "
            f"user_alpha        = {self._user_alpha}"
            f")")


if __name__ == '__main__':
    pass
