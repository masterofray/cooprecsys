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

from .scaffold          import TwoTowerBase
from .approximator      import TwoTowerPredictor as TheTwoTowerReasoner
from .model_architect   import TwoTowerArchitect as TheTwoTowerAdvisor
from .fallback_reasoner import TwoTowerFallBack

__all__ = ['TwoTowerBase',
           'TwoTowerPredictor', 'TheTwoTowerReasoner',
           'TwoTowerArchitect', 'TheTwoTowerAdvisor',
           'TwoTowerFallBack',
           ]

# Also expose the un-aliased names directly (both spellings work,
# matching arycolbring/inout/__init__.py's own __all__ list which
# exports both the raw class name and the short alias).
from .approximator import TwoTowerPredictor
from .model_architect import TwoTowerArchitect
