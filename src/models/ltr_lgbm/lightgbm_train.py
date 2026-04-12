#!/usr/bin/env python3

'''
Module Name : Advanced LightGBM LTR
Description : Handles modelling Learning to Rank with LightGBM.
author      : Aryanto
compiler    : python 3.10
date        : 20260328
Contact     : aryanto.dandan@gmail.com
'''

import os
import time
import json
import duckdb
import logging
import warnings
import mlflow
import mlflow.lightgbm

import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# Advanced LightGBM Ranker
# =========================================================

class AdvancedLightGBMLTR:

    def __init__(
        self,
        features,
        label="reordered",
        query_id="user_id",
        model_path="lightgbm_ltr_model.txt",
        experiment_name="LightGBM_LTR",
        large_data_threshold=50_000
    ):

        self.features = features
        self.label = label
        self.query_id = query_id
        self.model_path = model_path
        self.large_data_threshold = large_data_threshold
        self.experiment_name = experiment_name

        self.model = None
        self.metrics = dict()

        mlflow.set_experiment(self.experiment_name)

    # =====================================================
    # Data Preparation
    # =====================================================

    def _prepare_data(self, df):

        df = df.sort_values(self.query_id).reset_index(drop=True)

        X = df[self.features].values
        y = df[self.label].values
        group = df.groupby(self.query_id).size().values

        return X, y, group


    # =====================================================
    # DuckDB
    # =====================================================

    def _duckdb_prepare(self, df):

        logger.info("Using DuckDB")

        con = duckdb.connect(":memory:")
        con.register("df", df)

        query = f"""
        SELECT *
        FROM df
        ORDER BY {self.query_id}
        """

        df_sorted = con.execute(query).df()

        return self._prepare_data(df_sorted)


    # =====================================================
    # Dataset Builder
    # =====================================================

    def prepare_dataset(self, train_df, test_df):

        logger.info("Preparing Dataset")

        if len(train_df) > self.large_data_threshold:

            self.X_train, self.y_train, self.group_train = \
                self._duckdb_prepare(train_df)

            self.X_test, self.y_test, self.group_test = \
                self._duckdb_prepare(test_df)

        else:

            self.X_train, self.y_train, self.group_train = \
                self._prepare_data(train_df)

            self.X_test, self.y_test, self.group_test = \
                self._prepare_data(test_df)


    # =====================================================
    # Train
    # =====================================================

    def train(self):

        logger.info("Training Model")

        start = time.time()

        train_lgb = lgb.Dataset(
            self.X_train,
            label=self.y_train,
            group=self.group_train
        )

        test_lgb = lgb.Dataset(
            self.X_test,
            label=self.y_test,
            group=self.group_test
        )

        self.params = {

            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [5,10],

            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,

            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,

            "lambda_l1": 0.1,
            "lambda_l2": 0.1,

            "n_jobs": -1,
            "verbosity": -1,
            "seed": 42

        }

        self.model = lgb.train(

            self.params,
            train_lgb,
            valid_sets=[train_lgb, test_lgb],
            valid_names=["train","test"],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(50)
            ]

        )

        runtime = (time.time() - start) / 60

        logger.info(f"Training finished in {runtime:.2f} minutes")

        self.runtime = runtime


    # =====================================================
    # Metrics
    # =====================================================

    def evaluate(self):

        logger.info("Evaluating Model")

        preds = self.model.predict(self.X_test)

        self.metrics = {

            "NDCG": np.mean(preds),
            "PredictionMean": np.mean(preds),
            "PredictionStd": np.std(preds)

        }

        logger.info(self.metrics)


    # =====================================================
    # Visualization
    # =====================================================

    def plot_feature_importance(self):

        imp = self.model.feature_importance()

        df = pd.DataFrame({

            "Feature": self.features,
            "Importance": imp

        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=df,
            x="Importance",
            y="Feature"
        )

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig("feature_importance.png")

        plt.close()


    def plot_prediction_distribution(self):

        preds = self.model.predict(self.X_test)

        plt.figure(figsize=(8,5))

        sns.histplot(preds, bins=50, kde=True)

        plt.title("Prediction Distribution")

        plt.tight_layout()

        plt.savefig("prediction_distribution.png")

        plt.close()


    # =====================================================
    # Save Model
    # =====================================================

    def save_model(self):

        logger.info("Saving Model")

        self.model.save_model(self.model_path)


    # =====================================================
    # MLflow Logging
    # =====================================================

    def log_mlflow(self):

        logger.info("Logging MLflow")

        with mlflow.start_run():

            mlflow.log_params(self.params)

            mlflow.log_metrics(self.metrics)

            mlflow.log_metric("runtime_minutes", self.runtime)

            mlflow.log_artifact("feature_importance.png")

            mlflow.log_artifact("prediction_distribution.png")

            mlflow.lightgbm.log_model(
                self.model,
                "lightgbm_model"
            )


    # =====================================================
    # Full Pipeline
    # =====================================================

    def __call__(self, train_df, test_df):

        self.prepare_dataset(train_df, test_df)

        self.train()

        self.evaluate()

        self.plot_feature_importance()

        self.plot_prediction_distribution()

        self.save_model()

        self.log_mlflow()

        logger.info("Pipeline Finished")



# =========================================================
# Running Script
# =========================================================

if __name__ == "__main__":

    logger.info("Running Script Mode")

    # Example dataset loading
    train_df = pd.read_parquet("train.parquet")
    test_df = pd.read_parquet("test.parquet")

    feature_cols = [

        col for col in train_df.columns
        if col not in ["user_id","reordered"]

    ]

    model = AdvancedLightGBMLTR(
        features=feature_cols
    )

    model(train_df, test_df)