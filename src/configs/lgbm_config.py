#!/usr/bin/env python3
from __future__ import annotations

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-28"


"""
lgbm_config.py
________________________________________
Typed dataclass configuration hierarchy for the LTR framework.

All runtime parameters live here so every module has a single source
of truth.  Instances can be built programmatically or deserialised from
the project ``config.ini`` via :meth:`LTRConfig.from_ini`.

Classes
________________________________________
FeatureConfig   — column-name metadata (features, label, query_id)
ModelConfig     — file paths & MLflow experiment identity
TrainingConfig  — LightGBM training knobs & default hyper-parameters
TuningConfig    — Optuna Bayesian optimisation settings
InferenceConfig — inference / ranking settings
PathConfig      — filesystem & MLflow URI settings
LTRConfig       — master composite config
"""

import os
import configparser
from pathlib import Path
from datetime import datetime
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .logged import setup_logging

CwdDir = Path(__file__).resolve().parents[0]
PosDir = CwdDir.parents[1] / 'artifacts'
logger = setup_logging()
dates = f'{datetime.now():%Y%m%d}'

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Column-level metadata shared across all pipeline stages.
    features:
        Ordered list of feature column names used for model training.
    label:
        Binary / graded relevance label column name.
    query_id:
        Column that identifies a query group (e.g. ``user_id``).
    """
    features: List[str]
    label:    str = "reordered"
    query_id: str = "user_id"

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return len(self.features)

    def validate(self) -> None:
        """Raise ``ValueError`` when the config is internally inconsistent."""
        if not self.features:
            raise ValueError("FeatureConfig.features must not be empty.")
        if self.label in self.features:
            raise ValueError(
            f"Label column '{self.label}' must not appear in features.")
        if self.query_id in self.features:
            raise ValueError(
            f"query_id column '{self.query_id}' must not appear in features.")


@dataclass
class ModelConfig:
    """Identity and persistence configuration for the trained model.
    model_path:
        Filesystem path where the booster is saved / loaded (``*.txt``).
    experiment_name:
        MLflow experiment label.
    large_data_threshold:
        Row count above which DuckDB parallel processing is activated.
    seed:
        Global random seed for reproducibility.
    """
    model_path:            str = "outputs/lightgbm_ltr_model.txt"
    experiment_name:       str = "LightGBM_LTR"
    large_data_threshold:  int = 50_000
    seed:                  int = 42


@dataclass
class TrainingConfig:
    """LightGBM training loop configuration and default hyper-parameters.
    num_boost_round:
        Maximum number of boosting iterations.
    early_stopping_rounds:
        Stop training if no improvement after this many rounds.
    log_evaluation:
        Frequency (in rounds) of log-evaluation callbacks.
    params:
        Raw LightGBM parameter dict; populated by :class:`BayesianTuner` or
        set from ``config.ini`` defaults.
    """
    num_boost_round        : int = 1_000
    early_stopping_rounds  : int = 100
    log_evaluation         : int = 50
    params: Dict[str, Any] = field(
        default_factory = lambda: {
            "objective"       : "lambdarank",
            "metric"          : "ndcg",
            "ndcg_eval_at"    : [5, 10],
            "learning_rate"   : 0.05,
            "num_leaves"      : 63,
            "max_depth"       : 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq"    : 5,
            "lambda_l1"       : 0.1,
            "lambda_l2"       : 0.1,
            "n_jobs"          : -1,
            "verbosity"       : -1,})

    def update_params(self, overrides: Dict[str, Any]) -> None:
        self.params.update(overrides)
        logger.debug("TrainingConfig.params updated: %s", overrides)


@dataclass
class TuningConfig:
    """Optuna Bayesian optimisation settings.
    n_trials:
        Number of Optuna trials.
    timeout:
        Wall-clock budget in seconds (``None`` = unlimited).
    direction:
        ``"maximize"`` for NDCG, ``"minimize"`` for loss-based metrics.
    study_name:
        Optuna study identifier.
    sampler:
        ``"tpe"`` | ``"random"`` | ``"cmaes"``
    pruner:
        ``"median"`` | ``"hyperband"`` | ``"none"``
    """
    n_trials:   int           = 50
    timeout:    Optional[int] = 3_600
    direction:  str           = "maximize"
    study_name: str           = "lgbm_ltr_study"
    sampler:    str           = "tpe"
    pruner:     str           = "median"


@dataclass
class InferenceConfig:
    """Inference and top-K ranking settings.
    top_k:
        Number of top-ranked items to return per query.
    score_col:
        Name of the score column added to inference output.
    """
    top_k:     int = 20
    score_col: str = "relevance_score"


@dataclass
class PathConfig:
    """Filesystem and MLflow URI settings.
    output_dir:
        Root directory for all artefacts (created if absent).
    mlflow_tracking_uri:
        MLflow tracking server URI or local path.
    html_report_path:
        Output path for the HTML monitoring report.
    """
    output_dir          : str = str(PosDir / "models" / dates)
    mlflow_tracking_uri : str = str(PosDir / "mlruns" / dates)
    html_report_path    : str = str(PosDir / "reports" / f'{dates}_LGBM_report.html')

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug("Output directory ensured: %s", self.output_dir)


# ---------------------------------------------------------------------------
# Master composite config
# ---------------------------------------------------------------------------
@dataclass
class LTRConfig:
    """Master composite configuration for the entire LTR pipeline.
    """
    feature   : FeatureConfig
    model     : ModelConfig     = field(default_factory=ModelConfig)
    training  : TrainingConfig  = field(default_factory=TrainingConfig)
    tuning    : TuningConfig    = field(default_factory=TuningConfig)
    inference : InferenceConfig = field(default_factory=InferenceConfig)
    path      : PathConfig      = field(default_factory=PathConfig)

    @classmethod
    def from_ini(cls, 
                 features: List[str], 
                 ini_path: str = None,
                ) -> "LTRConfig":
        """Deserialise an :class:`LTRConfig` from a ``configuration.ini`` file.
        ini_path: Path to the INI configuration file.
        features: Explicit list of feature column names (cannot be stored in
                  INI because they are dataset-dependent).
        Returns is LTRConfig class itself
        """
        cfg = configparser.ConfigParser()
        if not ini_path:
            ini_path = str(CwdDir / 'configuration.ini')
        cfg.read(ini_path)
        logger.debug("Loading config from: %s", ini_path)

        def _get(section: str, key: str, fallback: Any = None) -> Any:
            return cfg.get(section, key, fallback=str(fallback) if fallback is not None else None)

        # FeatureConfig
        feature_cfg = FeatureConfig(
            features = features,
            label    = _get("FEATURES", "label",    "reordered"),
            query_id = _get("FEATURES", "query_id", "user_id"),
        )

        # ModelConfig
        model_cfg = ModelConfig(
            model_path            = _get("MODEL_LGBM", "model_path", str(PosDir / 'models' / dates / 'lightgbm_ltr_model.txt')),
            experiment_name       = _get("MODEL_LGBM", "experiment_name", "LightGBM_LTR"),
            large_data_threshold  = int(_get("MODEL_LGBM", "large_data_threshold", 20_000)),
            seed                  = int(_get("MODEL_LGBM", "seed", 4)),)

        # TrainingConfig
        ndcg_at_raw = _get("TRAINING", "ndcg_eval_at", "5,10")
        ndcg_at     = [int(x.strip()) for x in ndcg_at_raw.split(",")]
        params = {
            "objective"        : _get("TRAINING", "objective", "lambdarank"),
            "metric"           : _get("TRAINING", "metric", "ndcg"),
            "ndcg_eval_at"     :  ndcg_at,
            "learning_rate"    : float(_get("TRAINING", "learning_rate",   0.05)),
            "max_depth"        : int(_get("TRAINING", "max_depth",            6)),
            "num_leaves"       : int(_get("TRAINING", "num_leaves",          63)),
            "feature_fraction" : float(_get("TRAINING", "feature_fraction", 0.8)),
            "bagging_fraction" : float(_get("TRAINING", "bagging_fraction", 0.8)),
            "bagging_freq"     : int(_get("TRAINING", "bagging_freq",         5)),
            "lambda_l1"        : float(_get("TRAINING", "lambda_l1",        0.1)),
            "lambda_l2"        : float(_get("TRAINING", "lambda_l2",        0.1)),
            "n_jobs"           : int(_get("TRAINING", "n_jobs",              -1)),
            "verbosity"        : int(_get("TRAINING", "verbosity",           -1)),
            "seed"             : int(_get("MODEL", "seed",                   42)),}
        training_cfg = TrainingConfig(
            num_boost_round       = int(_get("TRAINING", "num_boost_round", 1_000)),
            early_stopping_rounds = int(_get("TRAINING", "early_stopping_rounds", 100)),
            log_evaluation        = int(_get("TRAINING", "log_evaluation", 50)),
            params                = params,)

        # TuningConfig
        tuning_cfg = TuningConfig(
            n_trials   = int(_get("TUNING", "n_trials", 50)),
            timeout    = int(_get("TUNING", "timeout", 3_600)),
            direction  = _get("TUNING", "direction", "maximize"),
            study_name = _get("TUNING", "study_name", "lgbm_ltr_study"),
            sampler    = _get("TUNING", "sampler", "tpe"),
            pruner     = _get("TUNING", "pruner", "median"),)

        # InferenceConfig
        inference_cfg = InferenceConfig(
            top_k     = int(_get("INFERENCE", "top_k", 20)),
            score_col = _get("INFERENCE", "score_col", "relevance_score"),)

        # PathConfig
        path_cfg = PathConfig(
            output_dir          = _get("PATHS", "output_dir", str(PosDir / "models" / dates)),
            mlflow_tracking_uri = _get("PATHS", "mlflow_tracking_uri", str(PosDir / "mlruns" / dates)),
            html_report_path    = _get("PATHS", "html_report_path", str(PosDir / "reports" / f'{dates}_LGBM_report.html')),)

        logger.debug("LTRConfig successfully loaded.")
        return cls(
            feature   = feature_cfg,
            model     = model_cfg,
            training  = training_cfg,
            tuning    = tuning_cfg,
            inference = inference_cfg,
            path      = path_cfg,
        )

    def validate(self) -> None:
        self.feature.validate()
        if "seed" not in self.training.params:
            self.training.params["seed"] = self.model.seed
        logger.debug("LTRConfig validation passed.")

if __name__ == '__main__':
    print(f'Pass on {dates}.')