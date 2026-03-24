#!/usr/bin/env python3

'''
Module Name : XGBoost with Optuna optimizer
Description : Handles modelling Learning to Rank with XGBoost + Optuna hyperparametrics.
author      : Aryanto
compiler    : python 3.10
date        : 20260324
Contact     : aryanto.dandan@gmail.com
'''

import xgboost as xgb
import optuna
import numpy as np
import time
import warnings
import logging
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import ray
import pandas as pd
import re


@ray.remote
def train_fold_remote(params, X_tr, y_tr, qid_tr, X_va, y_va, qid_va):
    # Sort by query_id (IMPORTANT for group sizes)
    def prepare(X, y, qid):
        df = pd.DataFrame({"qid": qid, "idx": np.arange(len(qid))})
        df = df.sort_values("qid")
        group_sizes = df.groupby("qid").size().values
        X_sorted = X[df["idx"].values]
        y_sorted = y[df["idx"].values]
        return X_sorted, y_sorted, group_sizes

    X_tr, y_tr, g_tr = prepare(X_tr, y_tr, qid_tr)
    X_va, y_va, g_va = prepare(X_va, y_va, qid_va)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(g_tr)

    dvalid = xgb.DMatrix(X_va, label=y_va)
    dvalid.set_group(g_va)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,   # faster tuning
        evals=[(dvalid, "valid")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    eval_str = model.eval(dvalid)
    ndcg = float(re.search(r"ndcg@5:([0-9.]+)", eval_str).group(1))
    map_score = float(re.search(r"map@5:([0-9.]+)", eval_str).group(1))

    return ndcg, map_score


class AdvancedXGBRanker:

    def __init__(
        self,
        X_train, y_train, group_sizes_train, query_ids_train,
        X_test, y_test, group_sizes_test, query_ids_test,
        n_splits=3,
        n_trials=20,
        sample_ratio=0.2,
        random_state=42,
        metric_weights={"ndcg": 0.7, "map": 0.3},
        use_ray=True,
        log_level=logging.INFO
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.group_sizes_train = group_sizes_train
        self.query_ids_train = query_ids_train

        self.X_test = X_test
        self.y_test = y_test
        self.group_sizes_test = group_sizes_test
        self.query_ids_test = query_ids_test

        self.n_splits = n_splits
        self.n_trials = n_trials
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.metric_weights = metric_weights
        self.use_ray = use_ray

        self.best_params = None
        self.best_model = None
        self.study = None

        # Logging
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ray init
        if self.use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Full DMatrix for final training
        self.dtrain_full = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtrain_full.set_group(self.group_sizes_train)

        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dtest.set_group(self.group_sizes_test)

    # =========================
    # Sampling (QUERY LEVEL)
    # =========================
    def _sample_data(self):
        unique_q = np.unique(self.query_ids_train)
        n_sample = max(1, int(len(unique_q) * self.sample_ratio))

        sampled_q = np.random.choice(unique_q, n_sample, replace=False)
        mask = np.isin(self.query_ids_train, sampled_q)

        return (
            self.X_train[mask],
            self.y_train[mask],
            self.query_ids_train[mask]
        )

    # =========================
    # Objective
    # =========================
    def _objective(self, trial):
        try:
            X_s, y_s, q_s = self._sample_data()

            params = {
                "objective": "rank:ndcg",
                "eval_metric": ["ndcg@5", "map@5"],
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "tree_method": "hist",
                "verbosity": 0,
                "nthread": 1 if self.use_ray else -1,
                "seed": self.random_state,
            }

            gkf = GroupKFold(n_splits=self.n_splits)
            folds = list(gkf.split(X_s, y_s, groups=q_s))

            scores = list()

            if self.use_ray:
                futures = list()
                for tr_idx, va_idx in folds:
                    futures.append(
                        train_fold_remote.remote(
                            params,
                            X_s[tr_idx], y_s[tr_idx], q_s[tr_idx],
                            X_s[va_idx], y_s[va_idx], q_s[va_idx]
                        )
                    )

                results = ray.get(futures)

            else:
                results = [
                    train_fold_remote.func(
                        params,
                        X_s[tr], y_s[tr], q_s[tr],
                        X_s[va], y_s[va], q_s[va]
                    )
                    for tr, va in folds
                ]

            for i, (ndcg, map_score) in enumerate(results):
                score = (
                    self.metric_weights["ndcg"] * ndcg +
                    self.metric_weights["map"] * map_score
                )
                scores.append(score)

                trial.report(score, step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return np.mean(scores)

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.logger.exception("Objective failed")
            raise e

    # =========================
    # Optimize
    # =========================
    def optimize(self):
        self.logger.info("Starting optimization")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )

            pbar = tqdm(total=self.n_trials)

            def cb(study, trial):
                pbar.update(1)

            self.study.optimize(self._objective, n_trials=self.n_trials, callbacks=[cb])
            pbar.close()

        self.best_params = self.study.best_params
        self.logger.info(f"Best Score: {self.study.best_value}")

    # =========================
    # Final Training (FULL DATA)
    # =========================
    def train_final(self):
        self.logger.info("Training final model on full data")

        params = {
            **self.best_params,
            "objective": "rank:ndcg",
            "eval_metric": ["ndcg@5", "map@5"],
            "tree_method": "hist",
            "nthread": -1,
            "verbosity": 1,
            "seed": self.random_state,
        }

        self.best_model = xgb.train(
            params,
            self.dtrain_full,
            num_boost_round=400,
            evals=[(self.dtrain_full, "train"), (self.dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=50
        )

    # =========================
    def evaluate(self):
        result = self.best_model.eval(self.dtest)
        self.logger.info(f"Test result: {result}")
        return result

    def plot_reports(self):
        self.logger.info("Generating Optuna reports and visualizations")

        try:
            import optuna.visualization as vis

            trials = self.study.trials
            values = [t.value for t in trials if t.value is not None]

            # 1. Optimization History
            plt.figure()
            plt.plot(values)
            plt.title("Optimization History")

            # 2. Score Distribution
            plt.figure()
            plt.hist(values, bins=20)
            plt.title("Score Distribution")

            # 3. Parameter vs Score (scatter)
            for param in self.best_params.keys():
                param_vals = [t.params.get(param) for t in trials if param in t.params]
                aligned_vals = values[:len(param_vals)]

                plt.figure()
                plt.scatter(param_vals, aligned_vals)
                plt.title(f"{param} vs Score")

            # 4. Param Importance
            importances = optuna.importance.get_param_importances(self.study)
            plt.figure()
            plt.bar(importances.keys(), importances.values())
            plt.title("Parameter Importance")

            # --- Optuna Interactive Plots ---
            try:
                vis.plot_optimization_history(self.study).show()
                vis.plot_param_importances(self.study).show()
                vis.plot_parallel_coordinate(self.study).show()
                vis.plot_slice(self.study).show()
                vis.plot_contour(self.study).show()
                vis.plot_edf(self.study).show()
            except Exception:
                self.logger.warning("Interactive plots failed (likely notebook issue)")

            plt.show()

        except Exception as e:
            self.logger.exception("Plotting failed")
            raise e

    # =========================
    def __call__(self):
        start = time.time()

        self.optimize()
        self.plot_reports()
        self.train_final()
        self.evaluate()

        # =========================
        # CALL TRACKER API
        # =========================
        tracker = ExperimentTracker(
            study=self.study,
            best_model=self.best_model
        )

        report_info = tracker.run()

        runtime = (time.time() - start) / 60
        self.logger.info(f"Total runtime: {runtime:.2f} minutes")
        self.logger.info(f"Report saved: {report_info}")

        return self.best_model

if __name__ == '__main__':
    import numpy as np
    import logging

    # =========================
    # Logging Configuration
    # =========================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # =========================
    # Initialize Ranker
    # =========================

    ranker = AdvancedXGBRanker(
        X_train = X_train,
        y_train = y_train,
        group_sizes_train = group_train, # Passing array of group counts
        query_ids_train = q_train,       # Passing array of user_ids for GroupKFold
        X_test = X_test,
        y_test = y_test,
        group_sizes_test = group_test,   # Passing array of group counts
        query_ids_test = q_test,         # Passing array of user_ids for GroupKFold
        n_splits=5,
        n_trials=20,
    )

    # =========================
    # Run Full Pipeline
    # =========================
    model = ranker()

    # =========================
    # Save Model (Optional)
    # =========================
    model.save_model("/content/xgb_ranker_model_02.json")

    print("Model training completed and saved.")