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
trainer.py
____________________________________________
Stateful LightGBM ``lambdarank`` trainer.
Wraps ``lgb.train`` with proper callback configuration, runtime tracking,
evaluation metric extraction, and model persistence.  All output (booster,
evals_result, runtime) is stored on ``self`` for downstream consumers
(:class:`Visualizer`, :class:`MLflowMonitor`).
"""


import os
import sys
import time
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from pathlib import Path
from typing import Any, Dict

LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import LTRConfig, logger


class LTRTrainer:
    """Train a LightGBM LambdaRank booster and expose evaluation artefacts.
    config          : `LTRConfig` master config.
    model           : lgb.Booster, The trained booster.
    evals_result    : dict, Training and validation metric history by round.
    metrics         : Dict[str, float], Flat summary metrics (NDCG@5, NDCG@10, pred mean/std, runtime).
    runtime_minutes : float, Wall-clock training time in minutes.
    best_iteration  : int, Booster's best iteration (early stopping).
    """
    def __init__(self, config: LTRConfig) -> None:
        self._config = config
        self.model:           lgb.Booster | None = None
        self.evals_result:    Dict[str, Any]     = dict()
        self.metrics:         Dict[str, float]   = dict()
        self.runtime_minutes: float              = 0.0
        self.best_iteration:  int                = 0
        logger.debug("LTRTrainer initialised.")

    @property
    def config(self) -> LTRConfig:
        return self._config

    @property
    def _params(self) -> Dict[str, Any]:
        return self._config.training.params

    def _build_lgb_datasets(
            self,
            X_train    : np.ndarray,
            y_train    : np.ndarray,
            group_train: np.ndarray,
            X_test     : np.ndarray,
            y_test     : np.ndarray,
            group_test : np.ndarray,
    ) -> tuple[lgb.Dataset, lgb.Dataset]:
        """Wrap NumPy arrays in :class:`lgb.Dataset` objects."""
        train_lgb = lgb.Dataset(
            X_train,
            label       = y_train,
            group       = group_train,
            free_raw_data = False)
        test_lgb = lgb.Dataset(
            X_test,
            label         = y_test,
            group         = group_test,
            reference     = train_lgb,
            free_raw_data = False)
        logger.debug(
            "LightGBM datasets built — train: %d rows | test: %d rows",
            len(y_train), len(y_test))
        return train_lgb, test_lgb

    def _extract_metrics_from_evals(self) -> None:
        """Parse ``self.evals_result`` and populate ``self.metrics``."""
        for split_name, metric_dict in self.evals_result.items():
            for metric_name, values in metric_dict.items():
                key = f"{split_name}_{metric_name.replace('@', 'at').replace(' ', '_')}"
                self.metrics[key] = float(values[-1])
                logger.debug("Metric captured -- %s: %.6f", key, self.metrics[key])


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self,
              X_train     : np.ndarray,
              y_train     : np.ndarray,
              group_train : np.ndarray,
              X_test      : np.ndarray,
              y_test      : np.ndarray,
              group_test  : np.ndarray,
        ) -> None:
        """Train the LambdaRank booster.
        """
        tcfg = self._config.training
        logger.info("LTRTrainer.train() — starting.")
        logger.info(
            "Params: num_boost_round=%d | early_stopping=%d | lr=%.4f | "
            "num_leaves=%d | max_depth=%d",
            tcfg.num_boost_round,
            tcfg.early_stopping_rounds,
            self._params.get("learning_rate", "—"),
            self._params.get("num_leaves",    "—"),
            self._params.get("max_depth",     "—"))
        train_lgb, test_lgb = self._build_lgb_datasets(
            X_train, y_train, group_train,
            X_test,  y_test,  group_test)

        # Progress bar via tqdm callback
        pbar = tqdm(total       = tcfg.num_boost_round,
                    desc        = "Training rounds",
                    unit        = "round",
                    dynamic_ncols = True)
        _pbar_state: Dict[str, Any] = {"last_iter": 0}

        def _tqdm_callback(env: lgb.callback.CallbackEnv) -> None:
            delta = env.iteration - _pbar_state["last_iter"]
            pbar.update(delta)
            _pbar_state["last_iter"] = env.iteration

        callbacks = [lgb.early_stopping(
                stopping_rounds = tcfg.early_stopping_rounds,
                verbose         = False),
            lgb.log_evaluation(period=tcfg.log_evaluation),
            lgb.record_evaluation(self.evals_result),
            _tqdm_callback]
        start_time = time.perf_counter()
        self.model = lgb.train(self._params,
                               train_lgb,
                               valid_sets   = [train_lgb, test_lgb],
                               valid_names  = ["train", "test"],
                               num_boost_round = tcfg.num_boost_round,
                               callbacks    = callbacks)
        elapsed = time.perf_counter() - start_time
        self.runtime_minutes = round(elapsed / 60.0, 2)
        self.best_iteration  = self.model.best_iteration
        pbar.close()
        logger.info("Training complete — best_iteration=%d | runtime=%.2f min",
                    self.best_iteration, self.runtime_minutes)
        self._extract_metrics_from_evals()
        self._compute_prediction_stats(X_test)


    def _compute_prediction_stats(self, X_test: np.ndarray) -> None:
        """Add prediction distribution stats to ``self.metrics``."""
        if self.model is None:
            return
        preds = self.model.predict(
            X_test, num_iteration=self.best_iteration
        )
        self.metrics["pred_mean"] = float(np.mean(preds))
        self.metrics["pred_std"]  = float(np.std(preds))
        self.metrics["pred_min"]  = float(np.min(preds))
        self.metrics["pred_max"]  = float(np.max(preds))
        logger.debug(
            "Prediction stats — mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
            self.metrics["pred_mean"], self.metrics["pred_std"],
            self.metrics["pred_min"],  self.metrics["pred_max"])


    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        path = self._config.model.model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_model(path)
        logger.info("Model saved to: %s", path)


    def load_model(self) -> None:
        path = self._config.model.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = lgb.Booster(model_file=path)
        logger.info("Model loaded from: %s", path)


if __name__ == '__main__':
    pass