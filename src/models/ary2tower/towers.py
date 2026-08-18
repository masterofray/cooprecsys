#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-30"

"""
towers.py
_________________________________________
User/item tower state + forward pass, with two interchangeable
backends:

  1. Cython + OpenMP (CLtowers, built via cysetup.py/.sh/.ps1) -- used
     automatically when the compiled extension is importable.
  2. A pure-NumPy fallback -- algorithmically identical (same formulas
     as CLtowers/_cy_train.pyx's tower_forward_single /
     tower_backward_update), used automatically otherwise.

Both paths are exercised through the exact same TwoTowerWeights /
UserTower / ItemTower API, so callers (TwoTowerTrainer,
TwoTowerInference) never need to know which backend is active.

The NumPy path exists for two reasons: (a) graceful degradation in any
environment without a C/Cython toolchain, matching the resilience
pattern the arycolbring notebooks in this repo already use, and (b) it
is what makes this module's core math independently testable -- see
test/ary2tower_tests/, which numerically gradient-checks the backward
pass against this implementation.
"""

import logging
from typing import Optional, Tuple

import numpy as np

try:
    from ...configs import logger
except ImportError:  # pragma: no cover - fallback for standalone/test use
    logger = logging.getLogger(__name__)

try:
    from .CLtowers import TwoTowerModel as _CyTwoTowerModel
    from .CLtowers import tower_forward as _cy_tower_forward
    from .CLtowers import fit_two_tower as _cy_fit_two_tower
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False
    logger.warning(
        "ary2tower: compiled CLtowers extension not found (run "
        "`python cysetup.py build_ext --inplace` in src/models/ary2tower/ "
        "to build it). Falling back to a pure-NumPy implementation -- "
        "correct, but without the OpenMP-parallel speedup.")


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class TwoTowerWeights:
    """Holds every trainable array for both towers (embeddings, the two
    dense layers per tower, and their SGD-momentum accumulators).

    This is the NumPy-backed equivalent of the Cython TwoTowerModel
    cdef class (CLtowers/_cy_types.pyx) -- same field names/shapes, so
    a `TwoTowerWeights` instance's arrays can be handed directly to the
    compiled kernels when they're available.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int,
                 hidden_dim: int, output_dim: int,
                 random_state: Optional[int] = None):
        rng = np.random.default_rng(random_state)

        def init_weight(shape):
            # Small-scale init, consistent with arycolbring's own
            # embedding init scale in inout/scaffold.py.
            return rng.normal(scale=0.05, size=shape).astype(np.float32)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.user_embeddings = init_weight((n_users, embedding_dim))
        self.user_w1 = init_weight((embedding_dim, hidden_dim))
        self.user_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.user_w2 = init_weight((hidden_dim, output_dim))
        self.user_b2 = np.zeros(output_dim, dtype=np.float32)

        self.item_embeddings = init_weight((n_items, embedding_dim))
        self.item_w1 = init_weight((embedding_dim, hidden_dim))
        self.item_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.item_w2 = init_weight((hidden_dim, output_dim))
        self.item_b2 = np.zeros(output_dim, dtype=np.float32)

        for prefix, n in (("user", n_users), ("item", n_items)):
            setattr(self, f"{prefix}_embeddings_momentum",
                    np.zeros((n, embedding_dim), dtype=np.float32))
            setattr(self, f"{prefix}_w1_momentum",
                    np.zeros((embedding_dim, hidden_dim), dtype=np.float32))
            setattr(self, f"{prefix}_b1_momentum",
                    np.zeros(hidden_dim, dtype=np.float32))
            setattr(self, f"{prefix}_w2_momentum",
                    np.zeros((hidden_dim, output_dim), dtype=np.float32))
            setattr(self, f"{prefix}_b2_momentum",
                    np.zeros(output_dim, dtype=np.float32))

        logger.info("TwoTowerWeights initialized: n_users=%d n_items=%d "
                    "embedding_dim=%d hidden_dim=%d output_dim=%d "
                    "(backend=%s)", n_users, n_items, embedding_dim,
                    hidden_dim, output_dim, "cython" if _HAS_CYTHON else "numpy")

    def as_cython_model(self):
        """Build the compiled TwoTowerModel cdef class from these same
        arrays (zero-copy -- memoryviews over the numpy buffers)."""
        if not _HAS_CYTHON:
            raise ImportError("CLtowers is not built; call cysetup.py first.")
        return _CyTwoTowerModel(
            self.user_embeddings, self.user_w1, self.user_b1, self.user_w2, self.user_b2,
            self.item_embeddings, self.item_w1, self.item_b1, self.item_w2, self.item_b2,
            self.user_embeddings_momentum, self.user_w1_momentum, self.user_b1_momentum,
            self.user_w2_momentum, self.user_b2_momentum,
            self.item_embeddings_momentum, self.item_w1_momentum, self.item_b1_momentum,
            self.item_w2_momentum, self.item_b2_momentum)


class _Tower:
    """Shared forward-pass logic for one tower (user or item). Not used
    directly -- see UserTower / ItemTower below."""

    def __init__(self, weights: TwoTowerWeights, prefix: str):
        self._weights = weights
        self._prefix = prefix

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w = self._weights
        return (getattr(w, f"{self._prefix}_embeddings"),
                getattr(w, f"{self._prefix}_w1"), getattr(w, f"{self._prefix}_b1"),
                getattr(w, f"{self._prefix}_w2"), getattr(w, f"{self._prefix}_b2"))

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """Batch forward pass: ids -> tower output representation.
        Uses the compiled Cython kernel when available, else NumPy."""
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        embeddings, w1, b1, w2, b2 = self._arrays()

        if _HAS_CYTHON:
            n = ids.shape[0]
            hidden_out = np.zeros((n, w1.shape[1]), dtype=np.float32)
            tower_out = np.zeros((n, w2.shape[1]), dtype=np.float32)
            _cy_tower_forward(ids, embeddings, w1, b1, w2, b2,
                              hidden_out, tower_out,
                              num_threads=4, verbose=False)
            return tower_out

        # --- NumPy fallback (identical formula) ---
        emb = embeddings[ids]
        hidden = relu(emb @ w1 + b1)
        return hidden @ w2 + b2


class UserTower(_Tower):
    """User tower: user_id -> user representation vector."""

    def __init__(self, weights: TwoTowerWeights):
        super().__init__(weights, prefix="user")


class ItemTower(_Tower):
    """Item tower: item_id -> item representation vector."""

    def __init__(self, weights: TwoTowerWeights):
        super().__init__(weights, prefix="item")


def dot_product_similarity(user_out: np.ndarray, item_out: np.ndarray) -> np.ndarray:
    """Row-wise dot product between paired user/item tower outputs."""
    return np.sum(user_out * item_out, axis=-1)


def cosine_similarity(user_out: np.ndarray, item_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise cosine similarity between paired user/item tower outputs."""
    dot = np.sum(user_out * item_out, axis=-1)
    norm_u = np.linalg.norm(user_out, axis=-1)
    norm_i = np.linalg.norm(item_out, axis=-1)
    return dot / (norm_u * norm_i + eps)
