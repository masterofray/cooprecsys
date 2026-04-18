# coding=utf-8
"""
arycolbring
===========
Ultra-optimised user-to-item collaborative filtering.

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
from .model import AryColBring                                      # noqa: E402,F401

from .cross_validation import (                                     # noqa: E402,F401
    random_train_test_split,
    user_based_train_test_split,
)

from .evaluation import (                                           # noqa: E402,F401
    precision_at_k,
    recall_at_k,
    auc_score,
    reciprocal_rank,
)

from .data_utils import (                                           # noqa: E402,F401
    load_interactions_from_df,
    load_interactions_from_csv,
    describe_interactions,
    validate_sparse_matrix,
)

__version__ = "0.1.0"
__author__  = "aryanto"

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
