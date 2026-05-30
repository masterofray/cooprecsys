#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-30"

from .flex import _shuffle as datashuffle
from .flex import coo_ttsplit, user_based_train_test_split

__all__ = ['datashuffle', 'coo_ttsplit', 'user_based_train_test_split']