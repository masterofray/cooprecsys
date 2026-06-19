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
import pandas       as pd
import scipy.sparse as sp
from pathlib  import Path
from copy     import deepcopy
from typing   import Optional, Tuple, Union

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from configs  import _cfg, logger
from db       import duckdb_connection
from features import load_data
from qrates   import GenQuasi_Lazy
from prepare  import DetectReco_Identifier
from models.arycolbring import AryColBringModelTrainer, RunTrainer
from models.arycolbring.assist import fileload_interactions, describe_interactions



class AryColBring_Reasoner_Test:
    """
    High-level training pipeline wrapper for AryColBring model.
    Handles data loading, training, evaluation, and reporting.
    """
    def __init__(self,
                 output_dir: Union[str, Path] = "artifacts",
                ):
        self.output_dir = Path(output_dir)
        self.config     = self._load_config()
        self._TRAIN     = None
        self._TEST      = None
        self._Data      = pd.DataFrame([])
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

    def _load_config(self) -> dict:
        config = {
        "no_components"    : _cfg.getint('model', "no_components", fallback=10),
        "loss"             : _cfg.get(model, "loss", fallback="warp"),
        "learning_rate"    : _cfg.getfloat(model, "learning_rate", fallback=0.05),
        "epochs"           : _cfg.getint(model, "epochs", fallback=10),
        "num_threads"      : _cfg.getint(model, "num_threads", fallback=4),
        "dtype"            : _cfg.get(model, "dtype", fallback="float32"),
        "learning_schedule": _cfg.get(model, "learning_schedule", fallback="adagrad")}
        logger.debug("Configuration loaded: %s", config)
        return config

        # Convert to sparse matrix (user-item interactions)
        self._TRAIN, _, _ = fileload_interactions(train_df)
        self._TEST, _, _  = fileload_interactions(test_df)
        gc.collect()

    def _RateProgress(self):
        self.data_rate = GenQuasi_Lazy(self.Data)
        logger.info("Data loaded: shape = %s", self.data_rate.shape)
        logger.debug("Columns: %s", self.data_rate.columns.tolist())

    def _RatePosthoc(self,
                     UserFeats: List = None,
                     ItemFeats: List = None,
                    ):
        Collect       = DetectReco_Identifier(self.Data.columns.to_numpy())
        MergeData     = self.data_rate.merge(self.Data,
                        on = [Collect['user_col'], Collect['item_col']])
        if UserFeats is None:
            UserFeats = ['EmployeeAge', 'EmployeeGender','Resistant', 'IsAllergic', 'VitalityDays']
        if ItemFeats is None:
            it01      = [Collect["quantity_col"], Collect["total_col"], Collect["discount_col"]]
            ItemFeats = ['ProductPrice', 'Quantity', 'Discount', 'TotalPrice', 'Class']
            ItemFeats.extend(it01)
            ItemFeats = list(set([it02 for it02 in ItemFeats if it02 is not None]))
        Results       = Decomposition_Matrix_Dev(
                        data              = MergeData,
                        user_col          = Collect['user_col'],
                        item_col          = Collect['item_col'],
                        user_feature_cols = UserFeats,
                        item_feature_cols = ItemFeats,)
        self.interactions  = Results[0]
        self.user_features = Results[1]
        self.item_features = Results[2]

    def load_training_data(self,
                           test_split : float = 0.2,
                          ) -> Tuple[sp.spmatrix, Optional[sp.spmatrix]]:
        np.random.seed(4)
        if not (1e-3 <= float(test_split) <= 0.7):
            logger.warning('Your test_split value outside our range!')
            test_split = 0.3
        
        mask = np.random.random(len(datafr)) > test_split
        train_df = deepcopy(datafr[mask])
        test_df  = deepcopy(datafr[~mask])
        logger.info("Data split: train = %d, test = %d",
                     len(train_df), len(test_df))


    # def train(self,
              # train_data: sp.spmatrix,
              # validation_data: Optional[sp.spmatrix] = None,
              # epochs: Optional[int] = None,
              # experiment_name: str = "AryColBring Training Run") -> Tuple[AryColBringModelTrainer, Path]:
        # """
        # Train the AryColBring model.

        # Args:
            # train_data: Training interaction matrix (sparse)
            # validation_data: Optional validation interaction matrix
            # epochs: Number of epochs (uses config if not specified)
            # experiment_name: Name for the experiment/report

        # Returns:
            # Tuple of (trained_model, report_path)
        # """
        # epochs = epochs or self.config["epochs"]

        # logger.info("Starting training: epochs=%d, threads=%d", epochs, self.config["num_threads"])

        # # Create and train model
        # model = AryColBringModelTrainer(
            # no_components=self.config["no_components"],
            # loss=self.config["loss"],
            # learning_rate=self.config["learning_rate"],
            # learning_schedule=self.config["learning_schedule"],
            # random_state=42
        # )

        # model.fit(
            # interactions=train_data,
            # epochs=epochs,
            # num_threads=self.config["num_threads"],
            # verbose=True,
            # validation_data=validation_data,
            # evaluate_every=1
        # )

        # # Generate report
        # report_path = model.generate_training_report(
            # output_dir=str(self.output_dir),
            # experiment_name=experiment_name
        # )

        # logger.info("Training completed. Report: %s", report_path)
        # return model, report_path

    # def save_model(self, model: AryColBringModelTrainer, model_name: str = "arycolbring_model.npz") -> Path:
        # """
        # Save trained model.

        # Args:
            # model: Trained AryColBringModelTrainer instance
            # model_name: Output model filename

        # Returns:
            # Path to saved model
        # """
        # model_path = self.output_dir / "models" / model_name
        # model_path.parent.mkdir(parents=True, exist_ok=True)

        # model.save_model(str(model_path))
        # logger.info("Model saved: %s", model_path)
        # return model_path

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
    """Main execution function for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train AryColBring model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test data fraction")
    parser.add_argument("--no-save", action="store_true", help="Skip saving model")
    parser.add_argument("--experiment-name", type=str, default="AryColBring Training", help="Experiment name")

    args = parser.parse_args()

    # Run pipeline
    pipeline = AryColBringTrainingPipeline(
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
