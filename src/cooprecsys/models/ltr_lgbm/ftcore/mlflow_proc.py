#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-01"


"""
mlflow_proc.py
______________
MLflow experiment tracking, parameter logging, metric logging, and
artifact registration for the LTR pipeline. All run-level state is 
stored on ``self`` so the caller can inspect ``self.run_id`` after 
the context is closed.
"""

import os
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from pathlib import Path
from typing import Any, Dict, Optional
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from ....configs import LTRConfig, logger, _cfg

if _cfg.getboolean('DEFAULT', 'is_cicd'):
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

class MLflowMonitor:
    """Log parameters, metrics, and artefacts to an MLflow experiment.
    The monitor is used as a context manager:

    .. code-block:: python
        with monitor:
            monitor.log_params(params)
            monitor.log_metrics(metrics)
            monitor.log_artifacts([...])
            monitor.log_model(booster)

    ____________________________
    Parameters
    config   : type :class:`~configs.LTRConfig` master config.
    run_name : Optional descriptive run name.  Defaults to ``"ltr_run"``.
    """

    def __init__(self,
            config:   LTRConfig,
            run_name: str = "ltr_run",
        ) -> None:
        self._config   = config
        self._run_name = run_name
        self.run_id:  Optional[str] = None
        self._run:    Optional[mlflow.ActiveRun] = None
        logger.debug("MLflowMonitor initialised — experiment: %s", config.model.experiment_name)


    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "MLflowMonitor":
        tracking_uri = self._config.path.mlflow_tracking_uri
        aksi_github  = _cfg.getboolean('DEFAULT', 'is_cicd')
        if not aksi_github:
            os.makedirs(tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self._config.model.experiment_name)
        self._run    = mlflow.start_run(run_name = self._run_name)
        self.run_id  = self._run.info.run_id
        logger.info(
            "MLflow run started — experiment='%s' | run_id=%s",
            self._config.model.experiment_name,
            self.run_id)
        return self

    def __exit__(self, 
            exc_type: Any, 
            exc_val: Any, 
            exc_tb: Any,
        ) -> None:
        if exc_type is not None:
            logger.error("MLflow run ending with exception: %s — %s",
                         exc_type.__name__, exc_val)
        mlflow.end_run()
        logger.debug("MLflow run ended - run_id=%s", self.run_id)


    # _____________________________________________________
    # Build Logging MLFlow helpers function
    # _____________________________________________________
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a flat parameter dict to the active run.
        Lists (e.g. ``ndcg_eval_at``) are serialised as comma-separated
        strings because MLflow param values must be strings.
        params: Key-value parameter dict.
        """
        serialised = {k: ",".join(str(x) for x in v) if isinstance(v, list) else str(v)
                      for k, v in params.items()}
        mlflow.log_params(serialised)
        logger.debug("MLflow params logged: %d entries", len(serialised))

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log a flat metrics dict.
        metrics: Key-value metrics dict (values must be numeric).
        """
        mlflow.log_metrics(metrics)
        logger.debug("MLflow metrics logged: %s", list(metrics.keys()))

    def log_runtime(self, runtime_minutes: float) -> None:
        """Log wall-clock training time as a metric.
        runtime_minutes: Training duration in minutes.
        """
        mlflow.log_metric("runtime_minutes", runtime_minutes)
        logger.debug("MLflow runtime logged: %.2f min", runtime_minutes)

    def log_artifacts(self, paths: list[str]) -> None:
        """Log a list of local file paths as run artefacts.
        Non-existent files are skipped with a warning rather than raising.
        paths: List of absolute or relative filesystem paths.
        """
        for path in paths:
            if not os.path.exists(path):
                logger.warning("Artefact not found, skipping: %s", path)
                continue
            mlflow.log_artifact(path)
            logger.debug("MLflow artefact logged: %s", path)

    def log_model(self, booster: lgb.Booster) -> None:
        """Log the LightGBM booster as an MLflow model artefact.
        booster: Trained :class:`lgb.Booster` instance.
        """
        feature_names = booster.feature_name()
        input_schema  = Schema([ColSpec("double", name) for name in feature_names])
        sign          = ModelSignature(inputs = input_schema)
        mlflow.lightgbm.log_model(lgb_model = booster, 
                                  name      = "lightgbm_model", 
                                  signature = sign)
        logger.info("LightGBM booster logged to MLflow.")

    def log_tuner_summary(self, summary: Dict[str, Any]) -> None:
        """Log Bayesian tuner results to the active run.
        Logs scalar values as metrics and the best params as run params.
        summary: Dict returned by :meth:`BayesianTuner.summary`.
        """
        if not summary:
            logger.debug("Tuner summary is empty — skipping MLflow log.")
            return
        mlflow.log_metric("tuning_best_value",  float(summary.get("best_value", 0.0)))
        mlflow.log_metric("tuning_n_trials",    int(summary.get("n_trials",    0)))
        mlflow.log_metric("tuning_n_pruned",    int(summary.get("n_pruned",    0)))
        best_params = summary.get("best_params", {})
        if best_params:
            prefixed = {f"tuned_{k}": v for k, v in best_params.items()}
            self.log_params(prefixed)
        logger.info("Tuner summary logged to MLflow.")

    def log_feature_config(self) -> None:
        """Log feature and query config metadata as tags."""
        mlflow.set_tags({
            "feature_count": str(self._config.feature.n_features),
            "label_col":     self._config.feature.label,
            "query_id_col":  self._config.feature.query_id})
        logger.debug("Feature config tags set in MLflow.")


    # _____________________________________________________
    # Running the process by call the MLflowMonitor()
    # _____________________________________________________

    def __call__(self,
            booster:         lgb.Booster,
            params:          Dict[str, Any],
            metrics:         Dict[str, float],
            runtime_minutes: float,
            artifact_paths:  list[str],
            tuner_summary:   Optional[Dict[str, Any]] = None,
        ) -> None:
        """Convenience method — log everything in one call.
        booster        : Trained LightGBM booster.
        params         : Training hyper-parameter dict.
        metrics        : Evaluation metrics dict.
        runtime_minutes: Wall-clock training duration.
        artifact_paths : File paths to log as artefacts (PNGs, HTML report, etc.).
        tuner_summary  : Optional Bayesian tuner summary dict.
        """
        logger.debug("MLflowMonitor() — logging full run.")
        self.log_feature_config()
        self.log_params(params)
        self.log_metrics(metrics)
        self.log_runtime(runtime_minutes)
        self.log_artifacts(artifact_paths)
        self.log_model(booster)
        if tuner_summary:
            self.log_tuner_summary(tuner_summary)
        logger.info("Done!\nMLflowMonitor() -- complete.")

if __name__ == '__main__':
    pass