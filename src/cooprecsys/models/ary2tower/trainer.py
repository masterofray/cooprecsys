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
TwoTowerTrainer: the top-level, save/load-aware training entry point.

The actual BPR-style training loop now lives in
inout/model_architect.py's TwoTowerArchitect -- this class is a thin
orchestration wrapper around it (same relationship
arycolbring/trainer.py's AryColBringModelTrainer has to TheAdvisor in
arycolbring/inout/model_architect.py). Every attribute/method that
existed here before the inout/ reorganization (.config, .weights,
.user_tower, .item_tower, .is_fitted, .loss_history, .fit(),
.save_model(), .load_model()) is preserved with identical behavior --
this refactor is a pure reorganization, not a behavior change (see
test/ary2tower_tests/t01_towers.py, unmodified, still passing against
this file).
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import scipy.sparse as sp
from ...configs import logger
from .config import TwoTowerConfig
from .towers import UserTower, ItemTower
from .inout.model_architect import TwoTowerArchitect


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
        self._architect = TwoTowerArchitect(n_users, n_items, config, **kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.user_tower = UserTower(self._architect.weights)
        self.item_tower = ItemTower(self._architect.weights)
        logger.info("TwoTowerTrainer initialized: n_users=%d n_items=%d %s",
                    n_users, n_items, self.config.get_params())

    @property
    def config(self) -> TwoTowerConfig:
        return self._architect.config

    @property
    def weights(self):
        return self._architect.weights

    @property
    def is_fitted(self) -> bool:
        return self._architect.is_fitted

    @property
    def loss_history(self) -> List[float]:
        return self._architect.loss_history

    def get_params(self):
        return self._architect.get_params()

    def set_params(self, **kwargs) -> "TwoTowerTrainer":
        self._architect.set_params(**kwargs)
        return self

    def fit(self, interactions: sp.spmatrix) -> "TwoTowerTrainer":
        """Train for `config.n_epochs` epochs over the nonzero entries
        of `interactions` (an (n_users x n_items) sparse matrix of
        implicit-feedback positives). Delegates to
        TwoTowerArchitect.fit() -- see inout/model_architect.py for the
        actual training loop and CLtowers/_cy_train.pyx for the full
        gradient derivation."""
        self._architect.fit(interactions)
        return self

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
        trainer._architect._is_fitted = True
        logger.info("Model loaded from %s", path)
        return trainer
