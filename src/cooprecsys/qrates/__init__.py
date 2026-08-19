#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-07"

from .multi_scann import QuasiRate_ScaNN
from .counterfeit import CFRatingEngine, CFRateLazy
from .quasi_grade import GenQuasi_Lazy, GenQuasi_Grade
from .quasi_grade import Decomposition_Matrix_Dev as DMD

__all__ = ['QuasiRate_ScaNN', 
           'CFRatingEngine',
           'CFRateLazy',
           'GenQuasi_Lazy',
           'GenQuasi_Grade',
           'DMD',
           ]
