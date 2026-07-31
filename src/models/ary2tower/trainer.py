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
trainer.py
_________________________________________
TwoTowerTrainer: BPR-style pairwise training for the two-tower model.

For each observed (user, positive_item) interaction, one negative item
is sampled uniformly and one SGD+momentum step pulls the positive
pair's dot-product score above the negative pair's. See
CLtowers/_cy_train.pyx's module docstring for the full gradient
derivation and the (documented, Hogwild!-style) parallelism trade-off
this shares with arycolbring's own fit_bpr()/fit_warp() kernels.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

try:
    from ...configs import logger
except ImportError:  # pragma: no cover - fallback for standalone/test use
    logger = logging.getLogger(__name__)

from .config import TwoTowerConfig
from .towers import TwoTowerWeights, UserTower, ItemTower, sigmoid, _HAS_CYTHON

if _HAS_CYTHON:
    from .towers import _cy_fit_two_tower


class TwoTowerTrainer:
    """Trains a two-tower model from an implicit-feedback interactions
    matrix.

    Parameters
    ----------
    n_users, n_items : catalogue size.
    config           : TwoTowerConfig (validated at construction --
                       see config.py).
    """

    def __init__(self, n_users: int, n_items: int,
                 config: Optional[TwoTowerConfig] = None, **kwargs):
        self.config = config or TwoTowerConfig(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.weights = TwoTowerWeights(
            n_users, n_items,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            random_state=self.config.random_state)
        self.user_tower = UserTower(self.weights)
        self.item_tower = ItemTower(self.weights)
        self.is_fitted = False
        self.loss_history: List[float] = []
        logger.info("TwoTowerTrainer initialized: n_users=%d n_items=%d %s",
                    n_users, n_items, self.config.get_params())

    def fit(self, interactions: sp.spmatrix) -> "TwoTowerTrainer":
        """Train for `config.n_epochs` epochs over the nonzero entries
        of `interactions` (an (n_users x n_items) sparse matrix of
        implicit-feedback positives)."""
        coo = interactions.tocoo()
        user_ids = coo.row.astype(np.int32)
        positive_item_ids = coo.col.astype(np.int32)
        n_examples = len(user_ids)
        if n_examples == 0:
            raise ValueError("fit() called with zero interactions")

        rng = np.random.default_rng(self.config.random_state)
        logger.info("Starting training: %d examples, %d epochs",
                    n_examples, self.config.n_epochs)

        for epoch in range(self.config.n_epochs):
            shuffle_indices = rng.permutation(n_examples).astype(np.int32)

            if _HAS_CYTHON:
                model = self.weights.as_cython_model()
                _cy_fit_two_tower(user_ids, positive_item_ids, shuffle_indices,
                                  model, self.config.learning_rate,
                                  self.config.momentum, self.config.num_threads,
                                  rng, verbose=self.config.verbose)
                # Cython kernel mutates the weight arrays in-place
                # (memoryviews over the same buffers), so there is
                # nothing further to write back here.
                epoch_loss = float("nan")  # the Cython kernel doesn't return a loss value
            else:
                epoch_loss = self._fit_epoch_numpy(user_ids, positive_item_ids,
                                                   shuffle_indices, rng)

            self.loss_history.append(epoch_loss)
            if self.config.verbose:
                logger.info("Epoch %d/%d: loss=%s", epoch + 1,
                            self.config.n_epochs,
                            f"{epoch_loss:.4f}" if epoch_loss == epoch_loss else "n/a (cython backend)")

        self.is_fitted = True
        logger.info("Training complete.")
        return self

    def _fit_epoch_numpy(self, user_ids: np.ndarray, positive_item_ids: np.ndarray,
                          shuffle_indices: np.ndarray, rng: np.random.Generator) -> float:
        """One epoch of the NumPy fallback path -- same math as
        CLtowers/_cy_train.pyx's fit_two_tower, sample by sample (no
        OpenMP parallelism in this path; see towers.py for why)."""
        w = self.weights
        lr = self.config.learning_rate
        momentum = self.config.momentum
        total_loss = 0.0

        for idx in shuffle_indices:
            uid = int(user_ids[idx])
            pos_iid = int(positive_item_ids[idx])
            neg_iid = int(rng.integers(self.n_items))

            u_hidden, u_out = self._tower_forward_single(
                w.user_embeddings, w.user_w1, w.user_b1, w.user_w2, w.user_b2, uid)
            p_hidden, p_out = self._tower_forward_single(
                w.item_embeddings, w.item_w1, w.item_b1, w.item_w2, w.item_b2, pos_iid)
            n_hidden, n_out = self._tower_forward_single(
                w.item_embeddings, w.item_w1, w.item_b1, w.item_w2, w.item_b2, neg_iid)

            pos_score = float(u_out @ p_out)
            neg_score = float(u_out @ n_out)
            g = 1.0 - float(sigmoid(np.array(pos_score - neg_score)))

            d_user_out = -g * p_out + g * n_out
            d_pos_out = -g * u_out
            d_neg_out = g * u_out

            self._tower_backward_update(w.user_embeddings, w.user_w1, w.user_b1,
                                        w.user_w2, w.user_b2, w.user_embeddings_momentum,
                                        w.user_w1_momentum, w.user_b1_momentum,
                                        w.user_w2_momentum, w.user_b2_momentum,
                                        uid, u_hidden, d_user_out, lr, momentum)
            self._tower_backward_update(w.item_embeddings, w.item_w1, w.item_b1,
                                        w.item_w2, w.item_b2, w.item_embeddings_momentum,
                                        w.item_w1_momentum, w.item_b1_momentum,
                                        w.item_w2_momentum, w.item_b2_momentum,
                                        pos_iid, p_hidden, d_pos_out, lr, momentum)
            self._tower_backward_update(w.item_embeddings, w.item_w1, w.item_b1,
                                        w.item_w2, w.item_b2, w.item_embeddings_momentum,
                                        w.item_w1_momentum, w.item_b1_momentum,
                                        w.item_w2_momentum, w.item_b2_momentum,
                                        neg_iid, n_hidden, d_neg_out, lr, momentum)

            total_loss += -np.log(1e-9 + sigmoid(np.array(pos_score - neg_score)))

        return float(total_loss / len(shuffle_indices))

    @staticmethod
    def _tower_forward_single(embeddings, w1, b1, w2, b2, entity_id):
        emb = embeddings[entity_id]
        hidden = np.maximum(emb @ w1 + b1, 0.0)
        out = hidden @ w2 + b2
        return hidden, out

    @staticmethod
    def _tower_backward_update(embeddings, w1, b1, w2, b2,
                                emb_mom, w1_mom, b1_mom, w2_mom, b2_mom,
                                entity_id, hidden, d_out, lr, momentum):
        d_hidden = d_out @ w2.T

        w2_mom[:] = momentum * w2_mom - lr * np.outer(hidden, d_out)
        w2 += w2_mom
        b2_mom[:] = momentum * b2_mom - lr * d_out
        b2 += b2_mom

        relu_mask = (hidden > 0).astype(np.float32)
        d_hidden_pre = d_hidden * relu_mask

        w1_mom[:] = momentum * w1_mom - lr * np.outer(embeddings[entity_id], d_hidden_pre)
        w1 += w1_mom
        b1_mom[:] = momentum * b1_mom - lr * d_hidden_pre
        b1 += b1_mom

        d_embedding = w1 @ d_hidden_pre
        emb_mom[entity_id] = momentum * emb_mom[entity_id] - lr * d_embedding
        embeddings[entity_id] += emb_mom[entity_id]

    def save_model(self, path: Union[str, Path]) -> None:
        """Save all weight arrays to a single .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w = self.weights
        np.savez(path,
                 user_embeddings=w.user_embeddings, user_w1=w.user_w1, user_b1=w.user_b1,
                 user_w2=w.user_w2, user_b2=w.user_b2,
                 item_embeddings=w.item_embeddings, item_w1=w.item_w1, item_b1=w.item_b1,
                 item_w2=w.item_w2, item_b2=w.item_b2,
                 embedding_dim=self.config.embedding_dim, hidden_dim=self.config.hidden_dim,
                 output_dim=self.config.output_dim)
        logger.info("Model saved to %s", path)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "TwoTowerTrainer":
        """Load a model previously written by save_model()."""
        data = np.load(path)
        n_users = data["user_embeddings"].shape[0]
        n_items = data["item_embeddings"].shape[0]
        config = TwoTowerConfig(embedding_dim=int(data["embedding_dim"]),
                                hidden_dim=int(data["hidden_dim"]),
                                output_dim=int(data["output_dim"]))
        trainer = cls(n_users, n_items, config=config)
        w = trainer.weights
        w.user_embeddings[:] = data["user_embeddings"]
        w.user_w1[:] = data["user_w1"]; w.user_b1[:] = data["user_b1"]
        w.user_w2[:] = data["user_w2"]; w.user_b2[:] = data["user_b2"]
        w.item_embeddings[:] = data["item_embeddings"]
        w.item_w1[:] = data["item_w1"]; w.item_b1[:] = data["item_b1"]
        w.item_w2[:] = data["item_w2"]; w.item_b2[:] = data["item_b2"]
        trainer.is_fitted = True
        logger.info("Model loaded from %s", path)
        return trainer
