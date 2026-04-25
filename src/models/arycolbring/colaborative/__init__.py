#!/usr/bin/env python3

"""
arycolbring
===========
Hybrid FPGrowth Ultra-optimised user-to-item collaborative filtering.

Public API
----------
Model
~~~~~
.. code-block:: python
    from arycolbring import AryColBring

    model = AryColBring(no_components=32, loss="warp", num_threads=4)
    model.fit(train_interactions, epochs=20, num_threads=4, verbose=True)
    scores = model.predict(user_ids, item_ids)

Data utilities
~~~~~~~~~~~~~~
.. code-block:: python

    from arycolbring import load_interactions_from_df, random_train_test_split

Cross-validation
~~~~~~~~~~~~~~~~
.. code-block:: python

    from arycolbring import random_train_test_split, user_based_train_test_split

Evaluation
~~~~~~~~~~
.. code-block:: python

    from arycolbring import precision_at_k, recall_at_k, auc_score, reciprocal_rank
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


import logging
import configparser
import os

# ── read config ───────────────────────────────────────────────────────────────
_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"))

_log_level = _cfg.get("logging", "level", fallback="DEBUG")
logging.basicConfig(
    level=getattr(logging, _log_level, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── public symbols ────────────────────────────────────────────────────────────
from .model import AryColBring

from .cross_validation import (
    random_train_test_split,
    user_based_train_test_split,
)

from .evaluation import (
    precision_at_k,
    recall_at_k,
    auc_score,
    reciprocal_rank,
)

from .data_utils import (
    load_interactions_from_df,
    load_interactions_from_csv,
    describe_interactions,
    validate_sparse_matrix,
)

__all__ = [
    # Core model
    "AryColBring",
    # Cross-validation
    "random_train_test_split",
    "user_based_train_test_split",
    # Evaluation
    "precision_at_k",
    "recall_at_k",
    "auc_score",
    "reciprocal_rank",
    # Data utilities
    "load_interactions_from_df",
    "load_interactions_from_csv",
    "describe_interactions",
    "validate_sparse_matrix",
]
