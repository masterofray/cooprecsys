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
__created__    = "2026-05-01"


"""
ltr_call.py
_____________________________
Top-level pipeline orchestrator for the LTR framework.
This module wires all components together into a coherent execution
sequence with full logging, MLflow tracking, and configurable Bayesian
tuning.  It exposes both a programmatic API (``run_pipeline``) and a
``__main__`` entry-point for CLI use.
_____________________________

Pipeline stages
1. Config loading & validation
2. Output directory setup
3. Data preparation   (DataProcessor)
4. Bayesian tuning    (BayesianTuner — optional)
5. Model training     (LTRTrainer)
6. Visualisation      (Visualizer)
7. MLflow logging     (MLflowMonitor)
8. Inference demo     (LTRInference — on test set)
"""


import os
import time
import argparse
import pandas as pd
import cloudpickle as cp
import matplotlib.pyplot as plt
from pathlib     import Path
from datetime    import datetime
from typing      import Any, Dict, List, Optional

from .ftcore      import (BayesianTuner, MLflowMonitor, 
                         Visualizer, MLPstyle)
from .inout       import LTRTrainer, LTRInference
from .dataprepared import data_aftermath
from ...prepare     import Dict2Json
from ...configs     import LTRConfig, logger
from ...features    import DataProcessor, TrueString

plt.rcParams.update(MLPstyle)


# ---------------------------------------------------------------------------
# Pipeline stages (private)
# ---------------------------------------------------------------------------
def _stage_banner(stage: str) -> None:
    """Emit a clearly visible stage separator to the log."""
    border = "─" * 60
    logger.info(border)
    logger.info("  STAGE : %s", stage)
    logger.info(border)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def run_pipeline(
        config:      LTRConfig,
        train:       pd.DataFrame,
        test:        pd.DataFrame,
        run_tuning:  bool = True,
        run_name:    str  = "ltr_run",
    ) -> LTRTrainer:
    """Execute the full LTR pipeline end-to-end.
    config     : Fully-initialised `LTRConfig`.
    train   : Raw training DataFrame.
    test    : Raw validation / test DataFrame.
    run_tuning : Whether to run Bayesian hyper-parameter optimisation before
                 training.  Set ``False`` to use the config defaults.
    run_name   : Human-readable MLflow run name.
    """
    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("  LightGBM LTR Framework — Pipeline Start")
    logger.info("  Experiment : %s", config.model.experiment_name)
    logger.info("  run_tuning : %s", run_tuning)
    logger.info("=" * 60)

    # ── 1. Validate config ────────────────────────────────────────────
    _stage_banner("1 / 7  Config Validation")
    config.validate()
    config.path.ensure_output_dir()
    logger.debug("Config validated. Output dir: %s", config.path.output_dir)

    # ── 2. Data preparation ───────────────────────────────────────────
    _stage_banner("2 / 7  Data Preparation")
    strcols = TrueString(data  = train, )
              #exclude_patterns = [r'product', r'category', r'type'])
    train_ready, test_ready, encoman = data_aftermath(
        train_df       = train, 
        test_df        = test, 
        string_columns = strcols)
    encoman.save()
    processor = DataProcessor(config)
    processor.prepare(train_ready, test_ready)

    # ── 3. Bayesian tuning (optional) ─────────────────────────────────
    tuner_summary: Optional[Dict[str, Any]] = None
    if run_tuning:
        _stage_banner("3 / 7  Bayesian Hyper-parameter Tuning")
        tuner = BayesianTuner(
            config      = config,
            X_train     = processor.X_train,
            y_train     = processor.y_train,
            group_train = processor.group_train)
        tuner.tune()
        tuner_summary = tuner.summary()
        logger.debug("Tuning summary: %s", tuner_summary)
        tunepath = Path(config.path.html_report_path).resolve().parent / 'BayesTune.json'
        Dict2Json(tuner_summary, tunepath)
        logger.debug(f'Tuner summary saved on {str(tunepath)}.')
    else:
        _stage_banner("3 / 7  Bayesian Tuning — SKIPPED")
        logger.debug("Using default / config-supplied params.")

    # ── 4. Training ───────────────────────────────────────────────────
    _stage_banner("4 / 7  Model Training")
    trainer = LTRTrainer(config)
    trainer.train(
        X_train     = processor.X_train,
        y_train     = processor.y_train,
        group_train = processor.group_train,
        X_test      = processor.X_test,
        y_test      = processor.y_test,
        group_test  = processor.group_test)
    trainer.save_model()
    
    dates = f'{datetime.now():%Y%m%d}'
    configfile = Path(config.path.output_dir).resolve() / f'{dates}_config.cloudpickle'
    logger.info(
        "Training complete — best_iteration = %d | runtime = %.2f min",
        trainer.best_iteration, trainer.runtime_minutes)
    with configfile.open('wb') as file:
        cp.dump(config, file)
        logger.info(f'Done to save LTRConfig file: {str(configfile)}.')

    # ── 5. Visualisation ──────────────────────────────────────────────
    _stage_banner("5 / 7  Visualisation")
    viz = Visualizer(
        config       = config,
        model        = trainer.model,
        evals_result = trainer.evals_result,
        X_test       = processor.X_test,
        metrics      = trainer.metrics)
    viz()

    # ── 6. MLflow logging ─────────────────────────────────────────────
    _stage_banner("6 / 7  MLflow Logging")
    artifact_paths: List[str] = [
        os.path.join(config.path.output_dir, "feature_importance.png"),
        os.path.join(config.path.output_dir, "prediction_distribution.png"),
        os.path.join(config.path.output_dir, "learning_curves.png"),
        os.path.join(config.path.output_dir, "metrics_summary.png"),
        os.path.join(config.path.output_dir, "feature_correlation.png"),
        config.path.html_report_path]
    with MLflowMonitor(config, run_name=run_name) as monitor:
        monitor(booster         = trainer.model,
                params          = config.training.params,
                metrics         = trainer.metrics,
                runtime_minutes = trainer.runtime_minutes,
                artifact_paths  = artifact_paths,
                tuner_summary   = tuner_summary)

    # ── 7. Inference demo ─────────────────────────────────────────────
    _stage_banner("7 / 7  Inference Demo (Top-K on Test Set)")
    inference = LTRInference(config, model = trainer.model)
    inference.rank_top_k(test_ready)
    predictpath = inference.save_rankings()
    logger.info("Top-%d rankings saved.", config.inference.top_k)
    logger.info('Begin to write report with HTML format.')
    viz.genreport(tuner_summary = tuner_summary, 
                  preddata      = predictpath,
                  model_metric  = trainer.metrics)
    logger.info("Created HTML report is done!")

    # ── Summary ───────────────────────────────────────────────────────
    total_minutes = (time.perf_counter() - pipeline_start) / 60.0
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.2f minutes.", total_minutes)
    logger.info("  Model   : %s", config.model.model_path)
    logger.info("  Report  : %s", config.path.html_report_path)
    logger.info("  MLflow  : %s", config.path.mlflow_tracking_uri)
    logger.info("=" * 60)

    return trainer


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _build_cli_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog        = "python -m ltr_framework.main",
        description = "Run the LightGBM LTR pipeline from the command line.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    cli.add_argument(
        "--train",  required=True,
        help="Path to the training Parquet / CSV file.")
    cli.add_argument(
        "--test",   required=True,
        help="Path to the test / validation Parquet / CSV file.")
    cli.add_argument(
        "--config", default="config.ini",
        help="Path to the config.ini file.")
    cli.add_argument(
        "--features", nargs = "+", default = None,
        help=("Explicit list of feature column names.  "
              "If omitted, all columns except --label and --query-id are used."))

    cli.add_argument("--label",     default = "reordered",  help = "Label column name.")
    cli.add_argument("--query-id",  default = "user_id",    help = "Query / group ID column name.")
    cli.add_argument("--no-tuning", action  = "store_true", help = "Skip Bayesian hyper-parameter tuning.")
    cli.add_argument("--run-name",  default = "ltr_run",    help = "MLflow run name.")
    return cli


def _load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        logger.info("Loading CSV: %s", path)
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        logger.info("Loading Parquet: %s", path)
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .parquet.")


if __name__ == "__main__":
    args = _build_cli_parser().parse_args()
    train_df_ = _load_dataframe(args.train)
    test_df_  = _load_dataframe(args.test)
    excluded = {args.label, args.query_id}
    feature_cols: List[str] = (
        args.features
        if args.features
        else [c for c in train_df_.columns if c not in excluded])
    cfg = LTRConfig.from_ini(args.config, features = feature_cols)
    cfg.feature.label    = args.label
    cfg.feature.query_id = args.query_id.replace("-", "_")
    run_pipeline(config     = cfg,
                 train      = train_df_,
                 test       = test_df_,
                 run_tuning = not args.no_tuning,
                 run_name   = args.run_name)
