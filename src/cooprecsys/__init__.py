#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.2"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Release Candidate"
__created__    = "2026-07-18"

from . import assets
from . import configs
from . import db
from . import features
from . import models
from . import noisemaker
from . import prepare
from . import qrates

__all__ = ["assets",
           "configs",
           "db",
           "features",
           "models",
           "noisemaker",
           "prepare",
           "qrates",
           "__version__",]
