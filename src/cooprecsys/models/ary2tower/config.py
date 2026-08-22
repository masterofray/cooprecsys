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
config.py
_________________________________________
Hyperparameter configuration for the two-tower model.

Validated eagerly in __post_init__ (construction time), matching the
convention this repo already uses in
cooprecsys.models.ary2tower/inout/scaffold.py's AryColBringBase.__init__ --
see t03_pytest.py's TestReasonerHyperparameterValidation, which is
where a matching bug (validation claimed to happen "at fit time" but
actually raised at construction) was caught and fixed for that module.
This module raises at construction on purpose, and is tested that way.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TwoTowerConfig:
    """Hyperparameters for TwoTowerTrainer / TwoTowerInference.

    Attributes
    ----------
    embedding_dim : size of the raw per-id embedding looked up for each
        user/item before either tower's dense layers.
    hidden_dim    : width of each tower's first (ReLU) dense layer.
    output_dim    : width of each tower's final output representation
        -- this is the vector space similarity is computed in.
    learning_rate : SGD step size.
    momentum      : SGD momentum coefficient (0 disables momentum).
    n_epochs      : number of full passes over the training pairs.
    num_threads   : OpenMP thread count for the Cython kernels.
    verbose       : if True, the Cython kernels print batch/epoch
        diagnostics to stderr (see the module docstring in
        CLtowers/_cy_train.pyx for why this is fprintf-based rather
        than routed through the project logger).
    random_state  : seed for embedding initialization and negative
        sampling.
    """
    embedding_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 16
    learning_rate: float = 0.01
    momentum: float = 0.9
    n_epochs: int = 10
    num_threads: int = 4
    verbose: bool = False
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0, got {self.n_epochs}")
        if self.num_threads <= 0:
            raise ValueError(f"num_threads must be > 0, got {self.num_threads}")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(f"random_state must be an int or None, got {type(self.random_state)}")

    def get_params(self) -> dict:
        """sklearn-style params dict, matching the get_params()
        convention used by AryColBringBase (TheReasoner/TheAdvisor)."""
        return {"embedding_dim": self.embedding_dim, "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim, "learning_rate": self.learning_rate,
                "momentum": self.momentum, "n_epochs": self.n_epochs,
                "num_threads": self.num_threads, "random_state": self.random_state}
