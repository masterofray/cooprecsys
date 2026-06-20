#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-04"


"""
t01_advisor.py
___________________________________________________________________
Training test script for AryColBring collaborative filtering model.
This script provides a comprehensive training pipeline for the AryColBring model
with integrated logging, progress tracking, and model persistence.
"""

import gc
import sys
import numpy        as np
import pandas       as pd
import scipy.sparse as sp
from pathlib  import Path
from copy     import deepcopy
from datetime import datetime
from pdb      import set_trace
from typing   import Optional, Tuple, Union, List
from argparse import ArgumentParser

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from configs  import _cfg, logger
from db       import duckdb_connection
from features import load_data
from prepare  import DetectReco_Identifier
from qrates   import GenQuasi_Lazy, DMD
from models.arycolbring import AryColBringModelTrainer as ACBmodel
from models.arycolbring.assist import fileload_interactions, describe_interactions


class AryColBring_Train_Test:
    """
    High-level training pipeline wrapper for AryColBring model.
    Handles data loading, training, evaluation, and reporting.
    """
    def __init__(self,
                 UserFeats : List  = None,
                 ItemFeats : List  = None,
                 testratio : float = float(),
                 output_dir: Union[str, Path] = None,
                ):
        if output_dir is None:
            self.output_dir= LocDir.parent / _cfg.get('PATHS', 'output_dir')
        else:
            self.output_dir= Path(output_dir)
        self.config        = self._load_config()
        self._Data         = pd.DataFrame([])
        self.DataMerge     = pd.DataFrame([])
        self.data_rate     = pd.DataFrame([])
        self.UserFeats     = UserFeats
        self.ItemFeats     = ItemFeats
        if self.UserFeats is None:
            self.UserFeats = ['EmployeeAge', 'EmployeeGender', 
                              'Resistant', 'IsAllergic', 'VitalityDays']
        if self.ItemFeats is None:
            self.ItemFeats = ['ProductPrice', 'Quantity', 'Discount',
                              'TotalPrice', 'Class']
        self._testratio    = testratio
        self.Collect       = dict()
        self._TRAIN        = None
        self._TEST         = None
        self.interactions  = None
        self.user_features = None
        self.item_features = None
        self.weight        = None
        self.user_ids      = None
        self.item_ids      = None
        self._report_path  = str()
        self.output_dir.mkdir(parents = True, exist_ok = True)
        logger.info("Training pipeline initialized in %s", self.output_dir)

    @property
    def Data(self) -> Path | pd.DataFrame:
        return self._Data

    @Data.setter
    def Data(self, value: str | Path | pd.DataFrame) -> None:
        if not isinstance(value, (str, Path, pd.DataFrame)):
            msg = f"Invalid data type: {type(value).__name__}."\
                   "Expected str, Path, or pandas.DataFrame."
            logger.error(msg)
            raise TypeError()
        if isinstance(value, (str, Path)):
            path_obj = Path(value)
            if not path_obj.exists():
                msg = f"Validation failed: File does not exist at '{path_obj}'."
                logger.error(msg)
                raise FileNotFoundError()
            if not path_obj.is_file():
                msg = f"Validation failed: Target path is a "\
                      f"directory, not a file -> '{path_obj}'."
                logger.error(msg)
                raise ValueError()
            if path_obj.stat().st_size == 0:
                msg = f"Validation failed: The file '{path_obj.name}' is empty."
                logger.error(msg)
                raise ValueError()
            try:
                with path_obj.open("r", encoding = "utf-8", errors = "ignore") as f:
                    line_count = sum(1 for _ in f)
                if line_count < 20:
                    msg = f"Validation failed: File has only {line_count} "\
                           "lines. A minimum of 20 lines is required."
                    logger.error(msg)
                    raise ValueError()
            except Exception as arc:
                msg = f"An error occurred while reading: {arg}"
                logger.error(msg)
                raise ValueError()
            finally:
                self._Data = load_data(data_path    = path_obj, 
                                       memory_limit = "16GB")
                logger.debug(f"Successfully assigned Data: '{path_obj}'.")
        elif isinstance(value, pd.DataFrame):
            if value.empty:
                msg = "Validation failed: The provided DataFrame is empty."
                logger.error(msg)
                raise ValueError()
            row_count = len(value)
            if row_count < 20:
                msg = f"Validation failed: DataFrame has only {row_count} rows."
                logger.error(msg)
                raise ValueError()
            self._Data = value
            logger.debug("Successfully assigned Data to pandas DataFrame.")

    @property
    def testratio(self) -> float:
        return self._testratio

    @Data.setter
    def testratio(self, value: float) -> None:
        if isinstance(value, float) and (0.01 <= value <= 0.8):
            self._testratio = value
        else:
            self._testratio = _cfg.getfloat("TRAINING", "test_ratio")

    def _load_config(self) -> dict:
        verbose = True if (_cfg.get('logging', 'level') in ['DEBUG', 'INFO']) else False
        config  = {
        "no_components"    : _cfg.getint('model', "no_components", fallback = 10),
        "loss"             : _cfg.get('model', "loss", fallback = "warp"),
        "learning_rate"    : _cfg.getfloat('model', "learning_rate", fallback = 0.05),
        "epochs"           : _cfg.getint('model', "epochs", fallback = 10),
        "num_threads"      : _cfg.getint('model', "num_threads", fallback = 4),
        "dtype"            : _cfg.get('model', "dtype", fallback = "float32"),
        "learning_schedule": _cfg.get('model', "learning_schedule", fallback = "adagrad"),
        'verbosity'        : verbose,}
        logger.debug("Configuration loaded: %s", config)
        return config

    def _RateProgress(self):
        self.data_rate = GenQuasi_Lazy(self.Data)
        logger.info("Data loaded: shape = %s", self.data_rate.shape)
        logger.debug("Columns: %s", self.data_rate.columns.tolist())
        return self

    def _RatePosthoc(self):
        self.Collect    = DetectReco_Identifier(self.Data.columns.to_numpy())
        self.DataMerge  = self.data_rate.merge(self.Data,
                          on = [self.Collect['user_col'], self.Collect['item_col']])
        it01            = [self.Collect["quantity_col"],
                           self.Collect["total_col"], 
                           self.Collect["discount_col"]]
        self.ItemFeats.extend(it01)
        self.ItemFeats  = list(set([it02 for it02 in self.ItemFeats if it02 is not None]))
        return self

    def _ttprocess(self, data: pd.DataFrame):
        Results = DMD(data              = data,
                      user_col          = self.Collect['user_col'],
                      item_col          = self.Collect['item_col'],
                      user_feature_cols = self.UserFeats,
                      item_feature_cols = self.ItemFeats)
        return Results

    def _brokedata(self):
        np.random.seed(4)
        n_rows      = self.DataMerge.shape[0]
        mask        = (np.random.rand(n_rows)) > self._testratio
        trainData   = deepcopy(self.DataMerge[mask])
        testData    = deepcopy(self.DataMerge[~mask])
        self._TRAIN = self._ttprocess(trainData)
        self._TEST  = self._ttprocess(testData)
        del mask, trainData, testData
        gc.collect()
        return self

    def train(self,
              epochs : Optional[int] = None,
              exname : str = "ACB Training Run",
             ) -> Tuple[ACBmodel, Path]:
        Epochs = epochs or self.config["epochs"]
        logger.info("Starting training: epochs = %d | threads = %d", 
                     epochs, self.config["num_threads"])
        self.ACBmodel = ACBmodel(no_components      = self.config["no_components"],
                                 loss               = self.config["loss"],
                                 learning_rate      = self.config["learning_rate"],
                                 item_alpha         = 0.01,
                                 user_alpha         = 0.01,
                                 learning_schedule  = self.config["learning_schedule"],
                                 random_state       = 4)
        self.ACBmodel.fit(interactions    = self._TRAIN[0], # Matriks Interaksi (COO)
                          user_features   = self._TRAIN[1], # Fitur User (CSR)
                          item_features   = self._TRAIN[2], # Fitur Item (CSR)
                          sample_weight   = self._TRAIN[3], # Bobot Interaksi (COO)
                          epochs          = Epochs,
                          num_threads     = self.config["num_threads"],
                          verbose         = self.config['verbosity'],
                          validation_data = self._TEST[0],
                          evaluate_every  = 1,
                         )
        self._report_path = self.ACBmodel.generate_training_report(
                            output_dir      = str(self.output_dir),
                            experiment_name = exname)
        logger.debug("Training completed. Report: %s", self._report_path)
        return self

    def save(self) -> Path:
        dates      = f'{datetime.now():%Y%m%d}'
        model_path = self.output_dir / "ACBmodel" / f'{dates}_models.npz'
        model_path.parent.mkdir(parents = True, exist_ok = True)
        self.ACBmodel.save_model(str(model_path))
        logger.info("Model saved: %s", model_path)
        return model_path




    # def run_full_pipeline(self,
                         # data_file: Union[str, Path],
                         # test_split: float = 0.2,
                         # epochs: Optional[int] = None,
                         # save_model: bool = True,
                         # experiment_name: str = "Full Training Pipeline") -> dict:
        # """
        # Run complete training pipeline.

        # Args:
            # data_file: Path to training data
            # test_split: Test data fraction
            # epochs: Number of epochs
            # save_model: Whether to save the trained model
            # experiment_name: Experiment name for reports

        # Returns:
            # Dictionary with results and paths
        # """
        # logger.info("Starting full pipeline: %s", experiment_name)

        # # Load data
        # train_data, val_data = self.load_training_data(data_file, test_split)

        # # Train
        # model, report_path = self.train(
            # train_data=train_data,
            # validation_data=val_data,
            # epochs=epochs,
            # experiment_name=experiment_name
        # )

        # # Save
        # model_path = None
        # if save_model:
            # model_path = self.save_model(model)

        # results = {
            # "status": "success",
            # "model": model,
            # "model_path": model_path,
            # "report_path": report_path,
            # "training_time": model.training_history[-1]["training_time_sec"] if model.training_history else 0,
            # "metrics": model.metrics_history[-1] if model.metrics_history else {},
            # "data_stats": describe_interactions(train_data)
        # }

        # logger.info("Pipeline completed successfully")
        # return results


def main():
    parser = ArgumentParser(description="Train AryColBring model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test data fraction")
    parser.add_argument("--no-save", action="store_true", help="Skip saving model")
    parser.add_argument("--experiment-name", type=str, default="AryColBring Training", help="Experiment name")

    args = parser.parse_args()

    # Run pipeline
    pipeline = AryColBring_Train_Test(
        data_dir="data",
        output_dir=args.output_dir
    )

    results = pipeline.run_full_pipeline(
        data_file=args.data,
        test_split=args.test_split,
        epochs=args.epochs,
        save_model=not args.no_save,
        experiment_name=args.experiment_name
    )

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    if results['model_path']:
        print(f"Model saved to: {results['model_path']}")
    print(f"Report generated: {results['report_path']}")
    if results['metrics']:
        print("\nMetrics:")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    print(f"\nData Statistics:")
    print(f"  Users: {results['data_stats']['n_users']}")
    print(f"  Items: {results['data_stats']['n_items']}")
    print(f"  Interactions: {results['data_stats']['n_interactions']}")
    print(f"  Sparsity: {results['data_stats']['sparsity']:.4f}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    print("Running the AryColBring_Reasoner_Test")
    #try:
    #    sys.exit(main())
    #except Exception as e:
    #    logger.exception("Training failed with error: %s", str(e))
    #    sys.exit(1)

    fx = AryColBring_Train_Test()
    fx.Data      = LocDir.parent / 'data' / 'sampledata.parquet'
    fx.testratio = 0.25
    fx._RateProgress()
    fx._RatePosthoc()
    fx._brokedata()
    fx.train()
    