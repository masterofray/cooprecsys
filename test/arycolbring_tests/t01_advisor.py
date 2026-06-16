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

import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from configs import _cfg, logger
from models.arycolbring import AryColBringModelTrainer, RunTrainer
from db import DuckDBManager, duckdb_connection


from db.callduckdb import 
from models.arycolbring.trainer import AryColBringModelTrainer, RunTrainer
from models.arycolbring.assist import fileload_interactions, describe_interactions


class AryColBring_Reasoner_Test:
    """
    High-level training pipeline wrapper for AryColBring model.
    Handles data loading, training, evaluation, and reporting.
    """

    def __init__(self,
                 data_dir: Union[str, Path],
                 output_dir: Union[str, Path] = "artifacts",
                 config_section: str = "model"):
        """
        Initialize training pipeline.

        Args:
            data_dir: Directory containing training data
            output_dir: Directory for output models and reports
            config_section: Configuration section to load from INI
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config_section = config_section
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()
        logger.info("Training pipeline initialized with output_dir=%s", self.output_dir)

    def _load_config(self) -> dict:
        """Load model configuration from INI file."""
        config = {
            "no_components": _cfg.getint(self.config_section, "no_components", fallback=10),
            "loss": _cfg.get(self.config_section, "loss", fallback="warp"),
            "learning_rate": _cfg.getfloat(self.config_section, "learning_rate", fallback=0.05),
            "epochs": _cfg.getint(self.config_section, "epochs", fallback=10),
            "num_threads": _cfg.getint(self.config_section, "num_threads", fallback=4),
            "dtype": _cfg.get(self.config_section, "dtype", fallback="float32"),
            "learning_schedule": _cfg.get(self.config_section, "learning_schedule", fallback="adagrad"),
        }
        logger.debug("Configuration loaded: %s", config)
        return config

    def load_training_data(self,
                           data_file: Union[str, Path],
                           test_split: float = 0.2,
                           random_state: int = 42) -> Tuple[sp.spmatrix, Optional[sp.spmatrix]]:
        """
        Load training data from file.

        Args:
            data_file: Path to data file (CSV or Parquet)
            test_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_interactions, validation_interactions)
        """
        data_file = Path(data_file)
        logger.info("Loading data from: %s", data_file)

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load data
        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        elif data_file.suffix == ".csv":
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        logger.info("Data loaded: shape=%s", df.shape)
        logger.debug("Columns: %s", df.columns.tolist())

        # Split into train/test if requested
        if test_split > 0:
            np.random.seed(random_state)
            mask = np.random.random(len(df)) > test_split
            train_df = df[mask].copy()
            test_df = df[~mask].copy()
            logger.info("Data split: train=%d, test=%d", len(train_df), len(test_df))
        else:
            train_df = df.copy()
            test_df = None

        # Convert to sparse matrix (user-item interactions)
        train_interactions, _, _ = fileload_interactions(train_df)
        test_interactions = None
        if test_df is not None:
            test_interactions, _, _ = fileload_interactions(test_df)

        return train_interactions, test_interactions

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
