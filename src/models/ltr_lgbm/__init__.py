#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-30"


"""
ltr_framework
=============
Production-grade Learning-to-Rank pipeline built on LightGBM.

Public surface
--------------
    LTRConfig           — master dataclass config (compose from sub-configs)
    DataProcessor       — DuckDB-backed, parallel-safe data preparation
    BayesianTuner       — Optuna Bayesian hyper-parameter optimiser
    LTRTrainer          — LightGBM lambdarank trainer
    LTRInference        — Top-K inference engine
    Visualizer          — Feature-importance / distribution plots + HTML report
    MLflowMonitor       — Experiment tracking & artifact logging
    run_pipeline        — One-call convenience entry-point
"""

from .config import (
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    TuningConfig,
    InferenceConfig,
    PathConfig,
    LTRConfig,)

from ftcore.enhanced_byoptimz import BayesianTuner
from inout.trainer import LTRTrainer
from inout.inference import LTRInference
from ftcore.vizseason import Visualizer
from ftcore.mlflow_proc import MLflowMonitor
from .ltr_call import run_pipeline

__all__ = [
    "BayesianTuner",
    "LTRTrainer",
    "LTRInference",
    "Visualizer",
    "MLflowMonitor",
    "run_pipeline",
]
