#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
scaffold.py
_________________________________________
TwoTowerBase: shared base class for TwoTowerPredictor (approximator.py)
and TwoTowerArchitect (model_architect.py) -- mirrors the role
AryColBringBase plays for TheReasoner/TheAdvisor in
arycolbring/inout/scaffold.py: catalogue-size validation at
construction, and the sklearn-style get_params()/set_params()
contract, both delegated to the already-validated TwoTowerConfig
dataclass (see ../config.py) rather than re-implemented here.
"""

from typing import Any, Dict
from ....configs import logger
from ..config import TwoTowerConfig


class TwoTowerBase:
    """Shared state + validation for both towers' catalogue size and
    hyperparameters. Not used directly -- see TwoTowerPredictor
    (approximator.py) and TwoTowerArchitect (model_architect.py).
    """

    def __init__(self, n_users: int, n_items: int,
                 config: TwoTowerConfig = None, **kwargs):
        if n_users <= 0:
            raise ValueError(f"n_users must be > 0, got {n_users}")
        if n_items <= 0:
            raise ValueError(f"n_items must be > 0, got {n_items}")

        self.n_users = n_users
        self.n_items = n_items
        # TwoTowerConfig.__post_init__ already validates every
        # hyperparameter eagerly -- nothing to re-check here.
        self.config = config if config is not None else TwoTowerConfig(**kwargs)
        self._is_fitted = False

        logger.debug("%s base initialized: n_users=%d n_items=%d",
                     type(self).__name__, n_users, n_items)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> Dict[str, Any]:
        """sklearn-style params dict (delegates to TwoTowerConfig)."""
        return self.config.get_params()

    def set_params(self, **kwargs) -> "TwoTowerBase":
        """sklearn-style params setter. Rejects unknown parameter
        names (mirrors AryColBringBase.set_params's validation)."""
        valid = set(self.config.get_params().keys())
        for key in kwargs:
            if key not in valid:
                raise ValueError(f"Unknown parameter '{key}'. Valid: {sorted(valid)}")
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}(n_users={self.n_users}, n_items={self.n_items}, {params})"
