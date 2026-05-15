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
enhanced_byoptimz.py
============================
Optuna-powered Bayesian hyper-parameter optimisation tailored for the
LightGBM ``lambdarank`` objective.

Key features
____________
    * TPE, CMA-ES, or Random sampler support (configured via :class:`TuningConfig`).
    * Median or HyperBand pruner support.
    * Each trial is a full LightGBM ``lgb.cv`` fold — no data leakage.
    * Best params are written back to ``config.training.params`` so the
    downstream :class:`LTRTrainer` picks them up automatically.
    * All trial results are logged to MLflow as child runs.
    * ``tqdm``-wrapped trial loop for real-time progress feedback.
"""

import re
import mlflow
import optuna
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from typing          import Any, Dict, Callable
from optuna.pruners  import (BasePruner, 
                             HyperbandPruner, 
                             MedianPruner, 
                             NopPruner)
from optuna.samplers import (BaseSampler, 
                             CmaEsSampler, 
                             RandomSampler, 
                             TPESampler)

import sys
from pathlib import Path
LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))

from configs import LTRConfig, _cfg, logger
try:
    log_level = _cfg.get('logging', 'OptunaLevel', 
                fallback = 'WARNING').upper()
    level = getattr(optuna.logging, log_level)
    optuna.logging.set_verbosity(level)
except AttributeError:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

if _cfg.getboolean('DEFAULT', 'is_cicd'):
    mlflow.set_tracking_uri('file:./mlruns')
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------
def _build_sampler(name: str, seed: int) -> BaseSampler:
    """Instantiate an Optuna sampler by name."""
    name = name.lower().strip()
    if name == "tpe":
        return TPESampler(seed=seed)
    if name in ("cmaes", "cma-es", "cma_es"):
        return CmaEsSampler(seed=seed)
    if name == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler '{name}'. Choose from: tpe, cmaes, random.")

def _build_pruner(name: str) -> BasePruner:
    """Instantiate an Optuna pruner by name."""
    name = name.lower().strip()
    if name == "median":
        return MedianPruner(n_startup_trials = 5, n_warmup_steps = 30)
    if name in ("hyperband", "hb"):
        return HyperbandPruner()
    if name in ("none", "nop"):
        return NopPruner()
    raise ValueError(f"Unknown pruner '{name}'. Choose from: median, hyperband, none.")


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
class _LambdaRankObjective:
    """Callable objective passed to ``study.optimize``.
    Cross-validates a single set of LightGBM hyper-parameters and returns
    the mean NDCG@10 over folds.
    ___________
    Parameters:
    X_train, y_train, group_train:
        Pre-processed training arrays from :class:`DataProcessor`.
    base_params: Fixed LightGBM params that are *not* tuned (e.g. objective, metric).
    num_boost_round, early_stopping_rounds: Training loop settings.
    n_cv_folds: Number of cross-validation folds.
    """
    def __init__(
            self,
            X_train:     np.ndarray,
            y_train:     np.ndarray,
            group_train: np.ndarray,
            base_params: Dict[str, Any],
            num_boost_round:       int = 300,
            early_stopping_rounds: int = 50,
            n_cv_folds:            int = 3,
        ) -> None:
        self._X_train     = X_train
        self._y_train     = y_train
        self._group_train = group_train
        self._base_params = base_params
        self._num_boost   = num_boost_round
        self._early_stop  = early_stopping_rounds
        self._n_folds     = n_cv_folds
        self._RunMLflow   = _cfg.getboolean('Tuning', 'run_mlflow', fallback = True)
        self._train_lgb   = lgb.Dataset(X_train, 
                                        label = y_train, 
                                        group = group_train, 
                                        free_raw_data = False)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        trial_params: Dict[str, Any] = {
            **self._base_params,
            # ---- Tunable search space --------------------------------
            "learning_rate":    trial.suggest_float("learning_rate",    1e-3, 0.3,  log=True),
            "num_leaves":       trial.suggest_int("num_leaves",         16,   255),
            "max_depth":        trial.suggest_int("max_depth",          3,    10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5,  1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5,  1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq",       1,    10),
            "lambda_l1":        trial.suggest_float("lambda_l1",        1e-4, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2",        1e-4, 10.0, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples",  5,    100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_pre_filter": False,
            "verbose": -1}
        try:
            cv_result = lgb.cv(
                trial_params,
                self._train_lgb,
                num_boost_round   = self._num_boost,
                nfold             = self._n_folds,
                stratified        = False,
                callbacks         = [lgb.early_stopping(stopping_rounds = self._early_stop,
                                     verbose = False),
                                     # period=0 or None silences the output],
                                     lgb.log_evaluation(period = 0)],
                return_cvbooster  = False,
                eval_train_metric = False)
        except Exception as arc:
            logger.warning("Trial %d raised: %s", trial.number, arc)
            raise optuna.exceptions.TrialPruned()

        metric_key = [k for k in cv_result if k.endswith("-mean")]
        if not metric_key:
            raise optuna.exceptions.TrialPruned()
        preferred  = [k for k in metric_key if "10" in k]
        key        = preferred[0] if preferred else metric_key[0]
        score      = float(cv_result[key][-1])
        logger.debug("Trial %d | %s = %.6f | params: %s",
            trial.number, key, score, {k: v for k, v in trial.params.items()})

        if self._RunMLflow:
            safe_key = re.sub(r'[^a-zA-Z0-9._-]', '_', key.replace("-mean", ""))
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.log_params(trial.params)
                mlflow.log_metric(safe_key, score)
        return score


# ---------------------------------------------------------------------------
# Public tuner class
# ---------------------------------------------------------------------------
class BayesianTuner:
    """Orchestrate Optuna Bayesian optimisation for LambdaRank.
    __________________________________________________________
    Parameters
    config: type :class:`~ltr_framework.config.LTRConfig` master config.
    X_train, y_train, group_train: Training arrays from :class:`DataProcessor`.
    
    Attributes (populated after :meth:`tune`)
    _________________________________________
    best_params : Dict[str, Any] :: Full LightGBM parameter dict with the best trial values merged in.
    study : optuna.Study :: The completed Optuna study object.
    """
    def __init__(
            self,
            config:      LTRConfig,
            X_train:     np.ndarray,
            y_train:     np.ndarray,
            group_train: np.ndarray,
        ) -> None:
        self._config      = config
        self._X_train     = X_train
        self._y_train     = y_train
        self._group_train = group_train
        self.best_params: Dict[str, Any] = dict()
        self.study:       optuna.Study | None = None
        logger.debug("BayesianTuner initialised.")

    @property
    def config(self) -> LTRConfig:
        return self._config

    def _build_study(self) -> optuna.Study:
        tcfg = self._config.tuning
        sampler = _build_sampler(tcfg.sampler, self._config.model.seed)
        pruner  = _build_pruner(tcfg.pruner)
        study = optuna.create_study(
            study_name = tcfg.study_name,
            direction  = tcfg.direction,
            sampler    = sampler,
            pruner     = pruner)
        logger.debug(
            "Optuna study created: '%s' | direction=%s | sampler=%s | pruner=%s",
            tcfg.study_name, tcfg.direction, tcfg.sampler, tcfg.pruner)
        return study

    def _tqdm_callback(self, 
            pbar: tqdm,
        ) -> Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]:
        """Return an Optuna callback that advances *pbar* on each trial."""
        def _cb(study: optuna.study.Study,
                trial: optuna.trial.FrozenTrial,
               ) -> None:
            pbar.update(1)
            try:
                best_val = f"{study.best_value:.6f}"
            except ValueError:
                best_val = "—"
            finally:
                pbar.set_postfix(best = best_val, trial = trial.number)
        return _cb

    def tune(self) -> Callable:
        """Run Bayesian optimisation and merge best params into config.
        Updates ``self.best_params`` and writes the best hyper-parameters
        back to ``self._config.training.params`` in-place.
        """
        logger.info(
            "BayesianTuner.tune() - starting %d trials (timeout=%ss).",
            self._config.tuning.n_trials,
            self._config.tuning.timeout)
        tcfg  = self._config.tuning
        trcfg = self._config.training
        base_params: Dict[str, Any] = {
            "objective":    trcfg.params.get("objective",    "lambdarank"),
            "metric":       trcfg.params.get("metric",       "ndcg"),
            "ndcg_eval_at": trcfg.params.get("ndcg_eval_at", [5, 10]),
            "n_jobs":       trcfg.params.get("n_jobs",       -1),
            "verbosity":    -1,
            "seed":         self._config.model.seed}

        objective = _LambdaRankObjective(
            X_train              = self._X_train,
            y_train              = self._y_train,
            group_train          = self._group_train,
            base_params          = base_params,
            num_boost_round      = min(trcfg.num_boost_round, 300),
            early_stopping_rounds= min(trcfg.early_stopping_rounds, 50))
        self.study = self._build_study()

        with tqdm(total         = tcfg.n_trials,
                  desc          = "Bayesian tuning",
                  unit          = "trial",
                  colour        = _cfg.get('tqdm', 'colour'),
                  ncols         = _cfg.getint('tqdm', 'ncols'),
                  mininterval   = 0.1,
                  dynamic_ncols = True) as pbar:
            self.study.optimize(
                objective,
                n_trials  = tcfg.n_trials,
                timeout   = tcfg.timeout,
                callbacks = [self._tqdm_callback(pbar)],
                show_progress_bar = False)
        best = self.study.best_trial
        logger.info(
            "Tuning complete -- best trial #%d | value=%.6f",
            best.number, best.value)
        logger.info("Best hyper-params: %s", best.params)

        # Merge best trial params into base config
        self.best_params = {**base_params, **best.params}
        self._config.training.update_params(self.best_params)
        logger.info("Best params from Bayesian Optimization "
        "written back to config.training.params.")
        return self.best_params

    def summary(self) -> Dict[str, Any]:
        if self.study is None:
            return dict()
        n_pruned = sum(1 for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED)
        result = {"n_trials"    : len(self.study.trials),
                  "best_value"  : self.study.best_value,
                  "best_params" : self.study.best_params,
                  "n_pruned"    : n_pruned}
        return result

if __name__ == '__main__':
    pass
