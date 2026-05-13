#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"

import sys
from pathlib import Path
from configparser import ConfigParser

LocDir = Path(__file__).resolve().parents[0]
sys.path.append(str(LocDir))

_cfg = ConfigParser()
_cfg.read(LocDir / "configuration.ini")

from .fallback_config import FallbackConfig
from .lgbm_config     import LTRConfig
from .logged          import setup_logging
logger = setup_logging()
