#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-31"


"""
trainer.py
----------
Complete training pipeline for AryColBring collaborative filtering model.

Integrates:
- Model training using Cython-optimized kernels from CLproximity
- Evaluation metrics computation
- Training report dashboard generation
- Model serialization for production deployment
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from ..inout.model_architect import AryColBringTrainer
from ..inout.approximator import AryColBringPredictor
from ..data_utils import (
    load_interactions_from_df,
    load_interactions_from_csv,
    describe_interactions,
)
from ..eval.evaluation import (
    precision_at_k,
    recall_at_k,
    auc_score,
    MRR_rank,
)
from .render_training import generate_training_report

logger = logging.getLogger(__name__)

__all__ = ["AryColBringModelTrainer", "train_and_report"]


class AryColBringModelTrainer:
    """
    High-level training wrapper for AryColBring collaborative filtering model.
    
    Provides a unified interface for:
    - Training with multiple loss functions
    - Automatic evaluation during training
    - Dashboard report generation
    - Model export for production
    
    Parameters
    ----------
    no_components : int
        Number of latent factors (default: 32)
    loss : str
        Loss function: "logistic", "warp", "bpr", or "warp-kos" (default: "warp")
    learning_rate : float
        Base learning rate (default: 0.05)
    item_alpha : float
        L2 regularization for items (default: 0.0)
    user_alpha : float
        L2 regularization for users (default: 0.0)
    learning_schedule : str
        Learning rate schedule: "adagrad" or "adadelta" (default: "adagrad")
    random_state : int | None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        no_components: int = 32,
        loss: str = "warp",
        learning_rate: float = 0.05,
        item_alpha: float = 0.0,
        user_alpha: float = 0.0,
        learning_schedule: str = "adagrad",
        random_state: Optional[int] = None,
    ) -> None:
        logger.info(
            "Initializing AryColBringModelTrainer: loss=%s components=%d lr=%.4f",
            loss, no_components, learning_rate
        )
        
        self.trainer = AryColBringTrainer(
            no_components=no_components,
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            learning_schedule=learning_schedule,
            random_state=random_state,
        )
        
        self.training_history: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, float]] = []
        self.config = {
            "no_components": no_components,
            "loss": loss,
            "learning_rate": learning_rate,
            "item_alpha": item_alpha,
            "user_alpha": user_alpha,
            "learning_schedule": learning_schedule,
            "random_state": random_state,
        }
    
    def fit(
        self,
        interactions: Union[sp.spmatrix, str, 'pd.DataFrame'],
        epochs: int = 10,
        num_threads: int = 4,
        verbose: bool = True,
        validation_data: Optional[sp.spmatrix] = None,
        evaluate_every: int = 1,
    ) -> "AryColBringModelTrainer":
        """
        Train the collaborative filtering model.
        
        Parameters
        ----------
        interactions : sparse matrix | str | DataFrame
            Training interactions (COO/CSR matrix, CSV path, or DataFrame)
        epochs : int
            Number of training epochs (default: 10)
        num_threads : int
            Number of parallel threads (default: 4)
        verbose : bool
            Show progress bar (default: True)
        validation_data : sparse matrix, optional
            Validation interactions for evaluation
        evaluate_every : int
            Evaluate metrics every N epochs (default: 1)
        
        Returns
        -------
        self
        """
        logger.info("Starting training: epochs=%d threads=%d", epochs, num_threads)
        
        # Load interactions if path provided
        if isinstance(interactions, str):
            logger.info("Loading interactions from CSV: %s", interactions)
            interactions, _, _ = load_interactions_from_csv(interactions)
        
        # Convert to COO format
        if not sp.isspmatrix_coo(interactions):
            interactions = interactions.tocoo()
        
        # Get data statistics
        data_stats = describe_interactions(interactions)
        logger.info(
            "Training data: users=%d items=%d interactions=%d sparsity=%.4f",
            data_stats["n_users"], data_stats["n_items"],
            data_stats["n_interactions"], data_stats["sparsity"]
        )
        
        # Training loop with evaluation
        start_time = datetime.now()
        
        self.trainer.fit(
            interactions=interactions,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info("Training completed in %.2f seconds", training_time)
        
        # Store training info
        self.training_history.append({
            "epochs": epochs,
            "num_threads": num_threads,
            "training_time_sec": training_time,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
        })
        
        # Evaluate on validation set if provided
        if validation_data is not None and epochs % evaluate_every == 0:
            logger.info("Evaluating on validation data...")
            metrics = self.evaluate(validation_data, num_threads=num_threads)
            self.metrics_history.append(metrics)
            logger.info(
                "Validation metrics: AUC=%.4f Precision@10=%.4f Recall@10=%.4f",
                metrics.get("auc", 0),
                metrics.get("precision_at_10", 0),
                metrics.get("recall_at_10", 0)
            )
        
        return self
    
    def evaluate(
        self,
        test_interactions: sp.spmatrix,
        train_interactions: Optional[sp.spmatrix] = None,
        num_threads: int = 4,
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Parameters
        ----------
        test_interactions : sparse matrix
            Test interaction matrix
        train_interactions : sparse matrix, optional
            Training interactions (to exclude from ranking)
        num_threads : int
            Number of threads for evaluation
        k_values : list of int
            K values for precision/recall (default: [5, 10, 20])
        
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        logger.info("Evaluating model with k=%s", k_values)
        
        # Create predictor for evaluation
        predictor = AryColBringPredictor(**self.config)
        predictor.item_embeddings = self.trainer.item_embeddings
        predictor.user_embeddings = self.trainer.user_embeddings
        predictor.item_biases = self.trainer.item_biases
        predictor.user_biases = self.trainer.user_biases
        
        metrics = {}
        
        # Compute metrics
        try:
            metrics["auc"] = float(auc_score(
                predictor,
                test_interactions,
                train_interactions=train_interactions,
                num_threads=num_threads,
            ))
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)
            metrics["auc"] = 0.0
        
        try:
            metrics["mrr"] = float(MRR_rank(
                predictor,
                test_interactions,
                train_interactions=train_interactions,
                num_threads=num_threads,
            ))
        except Exception as e:
            logger.warning("MRR computation failed: %s", e)
            metrics["mrr"] = 0.0
        
        for k in k_values:
            try:
                metrics[f"precision_at_{k}"] = float(precision_at_k(
                    predictor,
                    test_interactions,
                    k=k,
                    train_interactions=train_interactions,
                    num_threads=num_threads,
                ))
            except Exception as e:
                logger.warning("Precision@%d computation failed: %s", k, e)
                metrics[f"precision_at_{k}"] = 0.0
            
            try:
                metrics[f"recall_at_{k}"] = float(recall_at_k(
                    predictor,
                    test_interactions,
                    k=k,
                    train_interactions=train_interactions,
                    num_threads=num_threads,
                ))
            except Exception as e:
                logger.warning("Recall@%d computation failed: %s", k, e)
                metrics[f"recall_at_{k}"] = 0.0
        
        return metrics
    
    def generate_training_report(
        self,
        output_dir: Optional[str] = None,
        experiment_name: str = "Default Experiment",
        charts: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """
        Generate a comprehensive training dashboard report.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory for the report
        experiment_name : str
            Name of the experiment
        charts : list of dict, optional
            Custom charts to include
        
        Returns
        -------
        Path
            Path to the generated HTML report
        """
        logger.info("Generating training report...")
        
        # Build context data
        context_data = {
            "metrics": self.metrics_history[-1] if self.metrics_history else {},
            "data_statistics": {
                "n_users": self.trainer.user_embeddings.shape[0] if self.trainer.user_embeddings is not None else 0,
                "n_items": self.trainer.item_embeddings.shape[0] if self.trainer.item_embeddings is not None else 0,
            },
            "experiment_name": experiment_name,
            "loss": self.config["loss"],
            "epochs": self.training_history[-1]["epochs"] if self.training_history else 0,
            "no_components": self.config["no_components"],
            "learning_rate": self.config["learning_rate"],
            "item_alpha": self.config["item_alpha"],
            "user_alpha": self.config["user_alpha"],
            "learning_schedule": self.config["learning_schedule"],
            "charts": charts or [],
        }
        
        # Add training time info
        if self.training_history:
            context_data["training_time_sec"] = self.training_history[-1].get("training_time_sec", 0)
        
        # Generate report
        report_path = generate_training_report(
            context_data=context_data,
            output_dir=output_dir,
        )
        
        logger.info("Training report generated: %s", report_path)
        return report_path
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model for production deployment.
        
        Parameters
        ----------
        path : str
            Path to save the model (will save as .npz file)
        """
        logger.info("Saving model to: %s", path)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.savez(
            path,
            item_embeddings=self.trainer.item_embeddings,
            user_embeddings=self.trainer.user_embeddings,
            item_biases=self.trainer.item_biases,
            user_biases=self.trainer.user_biases,
            config=json.dumps(self.config),
        )
        
        logger.info("Model saved successfully")
    
    def load_model(self, path: str) -> "AryColBringModelTrainer":
        """
        Load a pre-trained model.
        
        Parameters
        ----------
        path : str
            Path to the saved model (.npz file)
        
        Returns
        -------
        self
        """
        logger.info("Loading model from: %s", path)
        
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        self.trainer.item_embeddings = data["item_embeddings"]
        self.trainer.user_embeddings = data["user_embeddings"]
        self.trainer.item_biases = data["item_biases"]
        self.trainer.user_biases = data["user_biases"]
        self.config = json.loads(str(data["config"]))
        
        logger.info("Model loaded successfully")
        return self
    
    def get_predictor(self) -> AryColBringPredictor:
        """
        Get a predictor instance for production inference.
        
        Returns
        -------
        AryColBringPredictor
            Ready-to-use predictor with loaded embeddings
        """
        predictor = AryColBringPredictor(**self.config)
        predictor.item_embeddings = self.trainer.item_embeddings
        predictor.user_embeddings = self.trainer.user_embeddings
        predictor.item_biases = self.trainer.item_biases
        predictor.user_biases = self.trainer.user_biases
        return predictor


def train_and_report(
    train_data: Union[sp.spmatrix, str],
    epochs: int = 10,
    no_components: int = 32,
    loss: str = "warp",
    learning_rate: float = 0.05,
    num_threads: int = 4,
    output_dir: Optional[str] = None,
    experiment_name: str = "Training Run",
    validation_data: Optional[sp.spmatrix] = None,
    save_model_path: Optional[str] = None,
) -> Tuple[AryColBringModelTrainer, Path]:
    """
    Convenience function to train a model and generate a report.
    
    Parameters
    ----------
    train_data : sparse matrix | str
        Training data (matrix or CSV path)
    epochs : int
        Number of training epochs
    no_components : int
        Number of latent factors
    loss : str
        Loss function ("logistic", "warp", "bpr", "warp-kos")
    learning_rate : float
        Learning rate
    num_threads : int
        Number of parallel threads
    output_dir : str, optional
        Directory for reports
    experiment_name : str
        Experiment name for the report
    validation_data : sparse matrix, optional
        Validation data for evaluation
    save_model_path : str, optional
        Path to save the trained model
    
    Returns
    -------
    tuple
        (trained_model, report_path)
    """
    logger.info("Starting training pipeline: %s", experiment_name)
    
    # Initialize trainer
    model = AryColBringModelTrainer(
        no_components=no_components,
        loss=loss,
        learning_rate=learning_rate,
    )
    
    # Train
    model.fit(
        interactions=train_data,
        epochs=epochs,
        num_threads=num_threads,
        validation_data=validation_data,
    )
    
    # Generate report
    report_path = model.generate_training_report(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    
    # Save model if requested
    if save_model_path:
        model.save_model(save_model_path)
    
    logger.info("Training pipeline completed successfully")
    return model, report_path


if __name__ == "__main__":
    # Example usage
    print("AryColBring Model Trainer")
    print("=" * 50)
    print("Use AryColBringModelTrainer class or train_and_report function")
    print("to train collaborative filtering models with dashboard reports.")
