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


"""
Models Module - Production API Gateway
Exposes core interfaces for AryColBring and Learning to Rank (LTR) LightGBM pipelines.
"""

from .            import arycolbring
from .            import ltr_lgbm
from .arycolbring import (AryColBringModelTrainer,
                          RunTrainer,
                          AryColBringInference,
                          InferenceService,
                          norm_exchange,
                          fileload_interactions,
                          describe_interactions,
                          validate_sparse_matrix, )

from .ltr_lgbm    import (BayesianTuner,
                          MLflowMonitor,
                          Visualizer,
                          LTRTrainer,
                          LTRInference,
                          render_report,
                          InferenceTest,
                          lgbm_fit_transform)

__all__ = ["arycolbring",
           "ltr_lgbm",

           # AryColBring API
           "AryColBringModelTrainer",
           "RunTrainer",
           "AryColBringInference",
           "InferenceService",
           'norm_exchange',
           'fileload_interactions',
           'describe_interactions',
           'validate_sparse_matrix',

           # LTR LGBM API
           "BayesianTuner",
           "MLflowMonitor",
           "Visualizer",
           "LTRTrainer",
           "LTRInference",
           "render_report",
           "InferenceTest",
           "lgbm_fit_transform"]