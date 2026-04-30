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
------------
* TPE, CMA-ES, or Random sampler support (configured via :class:`TuningConfig`).
* Median or HyperBand pruner support.
* Each trial is a full LightGBM ``lgb.cv`` fold — no data leakage.
* Best params are written back to ``config.training.params`` so the
  downstream :class:`LTRTrainer` picks them up automatically.
* All trial results are logged to MLflow as child runs.
* ``tqdm``-wrapped trial loop for real-time progress feedback.

Classes
-------
BayesianTuner — wraps an Optuna ``Study`` and exposes :meth:`tune`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
from tqdm import tqdm

from ltr_framework.config import LTRConfig

logger = logging.getLogger(__name__)

# Silence Optuna's own verbose logging — use our logger instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    raise ValueError(
        f"Unknown sampler '{name}'. Choose from: tpe, cmaes, random."
    )


def _build_pruner(name: str) -> BasePruner:
    """Instantiate an Optuna pruner by name."""
    name = name.lower().strip()
    if name == "median":
        return MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    if name in ("hyperband", "hb"):
        return HyperbandPruner()
    if name in ("none", "nop"):
        return NopPruner()
    raise ValueError(
        f"Unknown pruner '{name}'. Choose from: median, hyperband, none."
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

class _LambdaRankObjective:
    """Callable objective passed to ``study.optimize``.

    Cross-validates a single set of LightGBM hyper-parameters and returns
    the mean NDCG@10 over folds.

    Parameters
    ----------
    X_train, y_train, group_train:
        Pre-processed training arrays from :class:`DataProcessor`.
    base_params:
        Fixed LightGBM params that are *not* tuned (e.g. objective, metric).
    num_boost_round, early_stopping_rounds:
        Training loop settings.
    n_cv_folds:
        Number of cross-validation folds.
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

        self._train_lgb = lgb.Dataset(
            X_train, label=y_train, group=group_train, free_raw_data=False
        )

    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Sample hyper-parameters, run CV, return mean NDCG@10."""

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
        }

        try:
            cv_result = lgb.cv(
                trial_params,
                self._train_lgb,
                num_boost_round      = self._num_boost,
                nfold                = self._n_folds,
                stratified           = False,
                callbacks            = [lgb.early_stopping(self._early_stop, verbose=False)],
                return_cvbooster     = False,
                eval_train_metric    = False,
                verbose_eval         = False,
            )
        except Exception as exc:          # pragma: no cover
            logger.warning("Trial %d raised: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned()

        # LightGBM CV returns e.g. {"valid ndcg@10-mean": [...]}
        metric_key = [k for k in cv_result if k.endswith("-mean")]
        if not metric_key:
            raise optuna.exceptions.TrialPruned()

        # Prefer NDCG@10 if available
        preferred = [k for k in metric_key if "10" in k]
        key = preferred[0] if preferred else metric_key[0]

        score: float = float(cv_result[key][-1])

        logger.debug(
            "Trial %d | %s = %.6f | params: %s",
            trial.number, key, score, {
                k: v for k, v in trial.params.items()
            },
        )

        # Log individual trial to MLflow as a nested run
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial.params)
            mlflow.log_metric(key.replace("-mean", "").replace(" ", "_"), score)

        return score


# ---------------------------------------------------------------------------
# Public tuner class
# ---------------------------------------------------------------------------

class BayesianTuner:
    """Orchestrate Optuna Bayesian optimisation for LambdaRank.

    Parameters
    ----------
    config:
        :class:`~ltr_framework.config.LTRConfig` master config.
    X_train, y_train, group_train:
        Training arrays from :class:`DataProcessor`.

    Attributes (populated after :meth:`tune`)
    -----------------------------------------
    best_params : Dict[str, Any]
        Full LightGBM parameter dict with the best trial values merged in.
    study : optuna.Study
        The completed Optuna study object.
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

        self.best_params: Dict[str, Any] = {}
        self.study:       optuna.Study | None = None

        logger.debug("BayesianTuner initialised.")

    # ------------------------------------------------------------------

    @property
    def config(self) -> LTRConfig:
        return self._config

    # ------------------------------------------------------------------

    def _build_study(self) -> optuna.Study:
        """Create the Optuna study with sampler/pruner from config."""
        tcfg = self._config.tuning
        sampler = _build_sampler(tcfg.sampler, self._config.model.seed)
        pruner  = _build_pruner(tcfg.pruner)

        study = optuna.create_study(
            study_name = tcfg.study_name,
            direction  = tcfg.direction,
            sampler    = sampler,
            pruner     = pruner,
        )
        logger.debug(
            "Optuna study created: '%s' | direction=%s | sampler=%s | pruner=%s",
            tcfg.study_name, tcfg.direction, tcfg.sampler, tcfg.pruner,
        )
        return study

    # ------------------------------------------------------------------

    def _tqdm_callback(self, pbar: tqdm) -> optuna.study.StudyCallback:  # type: ignore[type-arg]
        """Return an Optuna callback that advances *pbar* on each trial."""

        def _cb(
            study: optuna.Study,
            trial: optuna.trial.FrozenTrial,
        ) -> None:
            pbar.update(1)
            pbar.set_postfix(
                best = f"{study.best_value:.6f}" if study.best_value else "—",
                trial = trial.number,
            )

        return _cb

    # ------------------------------------------------------------------

    def tune(self) -> None:
        """Run Bayesian optimisation and merge best params into config.

        Updates ``self.best_params`` and writes the best hyper-parameters
        back to ``self._config.training.params`` in-place.
        """
        logger.info(
            "BayesianTuner.tune() — starting %d trials (timeout=%ss).",
            self._config.tuning.n_trials,
            self._config.tuning.timeout,
        )

        tcfg = self._config.tuning
        trcfg = self._config.training

        # Fixed params not subject to search
        base_params: Dict[str, Any] = {
            "objective":    trcfg.params.get("objective",    "lambdarank"),
            "metric":       trcfg.params.get("metric",       "ndcg"),
            "ndcg_eval_at": trcfg.params.get("ndcg_eval_at", [5, 10]),
            "n_jobs":       trcfg.params.get("n_jobs",       -1),
            "verbosity":    -1,
            "seed":         self._config.model.seed,
        }

        objective = _LambdaRankObjective(
            X_train              = self._X_train,
            y_train              = self._y_train,
            group_train          = self._group_train,
            base_params          = base_params,
            num_boost_round      = min(trcfg.num_boost_round, 300),
            early_stopping_rounds= min(trcfg.early_stopping_rounds, 50),
        )

        self.study = self._build_study()

        with tqdm(
            total = tcfg.n_trials,
            desc  = "Bayesian tuning",
            unit  = "trial",
            dynamic_ncols = True,
        ) as pbar:
            self.study.optimize(
                objective,
                n_trials  = tcfg.n_trials,
                timeout   = tcfg.timeout,
                callbacks = [self._tqdm_callback(pbar)],
                show_progress_bar = False,
            )

        best = self.study.best_trial
        logger.info(
            "Tuning complete — best trial #%d | value=%.6f",
            best.number, best.value,
        )
        logger.info("Best hyper-params: %s", best.params)

        # Merge best trial params into base config
        self.best_params = {**base_params, **best.params}
        self._config.training.update_params(self.best_params)

        logger.info(
            "Best params written back to config.training.params."
        )

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the completed study.

        Returns
        -------
        dict with keys: ``n_trials``, ``best_value``, ``best_params``,
        ``n_pruned``.
        """
        if self.study is None:
            return {}

        n_pruned = sum(
            1 for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )

        return {
            "n_trials":   len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_pruned":   n_pruned,
        }
