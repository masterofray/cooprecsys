"""
Production-grade Python wrapper for Cython Collaborative Filtering modules.
Includes logging, MLflow tracking, DuckDB integration, and Joblib parallelization.
"""

import os
import sys
import logging
import pickle
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import duckdb
from joblib import Parallel, delayed, dump, load
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, csc_matrix
import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cloudpickle

# Import compiled Cython modules (will be available after compilation)
try:
    from collaborative_filtering import (
        UserBasedCF,
        ItemBasedCF,
        SVDBasedCF,
        NMFBasedCF
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    logging.warning("Cython modules not available. Ensure compilation is complete.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseCollaborativeFilteringModel(ABC):
    """
    Abstract base class for all collaborative filtering models.
    Provides common interface and utility methods.
    """
    
    def __init__(
        self,
        model_name: str,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: int = 42
    ):
        """Initialize base model."""
        self.model_name = model_name
        self.n_jobs = n_jobs if n_jobs > 0 else -1
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.training_history: Dict[str, List] = {}
        
        if self.verbose:
            logger.info(f"[{self.model_name}] Initialized with random_state={random_state}")
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self.model is not None
    
    @property
    def model_metadata(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'model_name': self.model_name,
            'fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history
        }
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], *args, **kwargs) -> None:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'training_history': self.training_history,
            'metadata': self.model_metadata
        }
        
        with open(path, 'wb') as f:
            cloudpickle.dump(model_data, f)
        
        logger.info(f"[{self.model_name}] Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = cloudpickle.load(f)
        
        self.model = model_data['model']
        self.encoders = model_data['encoders']
        self.training_history = model_data['training_history']
        
        logger.info(f"[{self.model_name}] Model loaded from {path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {e}")


class UserBasedCollaborativeFiltering(BaseCollaborativeFilteringModel):
    """
    User-Based Collaborative Filtering Model.
    Uses Cython-optimized similarity computation.
    """
    
    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = 'cosine',
        min_common_items: int = 1,
        **kwargs
    ):
        """Initialize user-based CF."""
        super().__init__("UserBasedCF", **kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_indices = {}
        self.item_indices = {}
    
    @property
    def n_neighbors_effective(self) -> int:
        """Get effective number of neighbors considering matrix size."""
        if self.user_item_matrix is not None:
            return min(self.n_neighbors, self.user_item_matrix.shape[0] - 1)
        return self.n_neighbors
    
    def fit(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        progress_bar: bool = True
    ) -> None:
        """
        Fit user-based CF model.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            ratings: Shape (n_samples,) array of ratings
            n_users: Total number of users
            n_items: Total number of items
            progress_bar: Show progress bar
        """
        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython modules not available")
        
        logger.info(f"[{self.model_name}] Fitting with {len(ratings)} ratings")
        
        # Build user-item matrix
        self.user_item_matrix = self._build_matrix(
            user_item_pairs, ratings, n_users, n_items, progress_bar
        )
        
        # Initialize Cython model
        self.model = UserBasedCF(
            self.user_item_matrix.astype(np.float64),
            n_neighbors=self.n_neighbors_effective
        )
        
        # Compute similarities
        if progress_bar:
            with tqdm_joblib(desc="Computing user means", total=1, colour='green'):
                self.model.compute_user_means()
        else:
            self.model.compute_user_means()
        
        if progress_bar:
            with tqdm_joblib(desc="Computing similarities", total=1, colour='green'):
                self.model.compute_similarities()
        else:
            self.model.compute_similarities()
        
        self.training_history['fit_samples'] = len(ratings)
        self.log_params({
            'n_neighbors': self.n_neighbors_effective,
            'metric': self.metric,
            'min_common_items': self.min_common_items
        })
        
        logger.info(f"[{self.model_name}] Fitting completed")
    
    def predict(
        self,
        user_item_pairs: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            progress_bar: Show progress bar
        
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        logger.info(f"[{self.model_name}] Making predictions for {len(user_item_pairs)} pairs")
        
        predictions = self.model.predict_ratings(
            user_item_pairs.astype(np.int32)
        )
        
        return np.clip(predictions, 0, 5)  # Clip to valid rating range
    
    def _build_matrix(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        progress_bar: bool = True
    ) -> np.ndarray:
        """Build user-item matrix from interactions."""
        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        
        iterator = tqdm(range(len(ratings)), desc="Building matrix", colour='green') if progress_bar else range(len(ratings))
        
        for idx in iterator:
            u, i = user_item_pairs[idx]
            matrix[u, i] = ratings[idx]
        
        logger.info(f"[{self.model_name}] Matrix built: {matrix.shape}, sparsity: {1 - np.count_nonzero(matrix) / matrix.size:.4f}")
        
        return matrix


class ItemBasedCollaborativeFiltering(BaseCollaborativeFilteringModel):
    """
    Item-Based Collaborative Filtering Model.
    Uses Cython-optimized item similarity computation.
    """
    
    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = 'cosine',
        **kwargs
    ):
        """Initialize item-based CF."""
        super().__init__("ItemBasedCF", **kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.user_item_matrix = None
    
    @property
    def n_neighbors_effective(self) -> int:
        """Get effective number of neighbors considering matrix size."""
        if self.user_item_matrix is not None:
            return min(self.n_neighbors, self.user_item_matrix.shape[1] - 1)
        return self.n_neighbors
    
    def fit(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        progress_bar: bool = True
    ) -> None:
        """
        Fit item-based CF model.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            ratings: Shape (n_samples,) array of ratings
            n_users: Total number of users
            n_items: Total number of items
            progress_bar: Show progress bar
        """
        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython modules not available")
        
        logger.info(f"[{self.model_name}] Fitting with {len(ratings)} ratings")
        
        # Build user-item matrix
        self.user_item_matrix = self._build_matrix(
            user_item_pairs, ratings, n_users, n_items, progress_bar
        )
        
        # Initialize Cython model
        self.model = ItemBasedCF(
            self.user_item_matrix.astype(np.float64),
            n_neighbors=self.n_neighbors_effective
        )
        
        # Compute similarities
        if progress_bar:
            with tqdm_joblib(desc="Computing item means", total=1, colour='green'):
                self.model.compute_item_means()
        else:
            self.model.compute_item_means()
        
        if progress_bar:
            with tqdm_joblib(desc="Computing similarities", total=1, colour='green'):
                self.model.compute_similarities()
        else:
            self.model.compute_similarities()
        
        self.training_history['fit_samples'] = len(ratings)
        self.log_params({
            'n_neighbors': self.n_neighbors_effective,
            'metric': self.metric
        })
        
        logger.info(f"[{self.model_name}] Fitting completed")
    
    def predict(
        self,
        user_item_pairs: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            progress_bar: Show progress bar
        
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        logger.info(f"[{self.model_name}] Making predictions for {len(user_item_pairs)} pairs")
        
        predictions = self.model.predict_ratings(
            user_item_pairs.astype(np.int32)
        )
        
        return np.clip(predictions, 0, 5)
    
    def _build_matrix(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        progress_bar: bool = True
    ) -> np.ndarray:
        """Build user-item matrix from interactions."""
        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        
        iterator = tqdm(range(len(ratings)), desc="Building matrix", colour='green') if progress_bar else range(len(ratings))
        
        for idx in iterator:
            u, i = user_item_pairs[idx]
            matrix[u, i] = ratings[idx]
        
        logger.info(f"[{self.model_name}] Matrix built: {matrix.shape}, sparsity: {1 - np.count_nonzero(matrix) / matrix.size:.4f}")
        
        return matrix


class SVDCollaborativeFiltering(BaseCollaborativeFilteringModel):
    """
    SVD-based Matrix Factorization Collaborative Filtering.
    Uses Cython-optimized gradient descent.
    """
    
    def __init__(
        self,
        latent_dim: int = 50,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        **kwargs
    ):
        """Initialize SVD CF."""
        super().__init__("SVDBasedCF", **kwargs)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
    
    def fit(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        epochs: int = 10,
        progress_bar: bool = True
    ) -> None:
        """
        Fit SVD CF model.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            ratings: Shape (n_samples,) array of ratings
            n_users: Total number of users
            n_items: Total number of items
            epochs: Number of training epochs
            progress_bar: Show progress bar
        """
        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython modules not available")
        
        logger.info(f"[{self.model_name}] Fitting with {len(ratings)} ratings for {epochs} epochs")
        
        # Initialize Cython model
        self.model = SVDBasedCF(
            n_users,
            n_items,
            latent_dim=self.latent_dim,
            learning_rate=self.learning_rate,
            regularization=self.regularization
        )
        
        # Fit model
        self.model.fit(
            user_item_pairs.astype(np.int32),
            ratings.astype(np.float64),
            epochs=epochs
        )
        
        self.training_history['epochs'] = epochs
        self.training_history['fit_samples'] = len(ratings)
        self.log_params({
            'latent_dim': self.latent_dim,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'epochs': epochs
        })
        
        logger.info(f"[{self.model_name}] Fitting completed")
    
    def predict(
        self,
        user_item_pairs: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            progress_bar: Show progress bar
        
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        logger.info(f"[{self.model_name}] Making predictions for {len(user_item_pairs)} pairs")
        
        predictions = self.model.predict_ratings(
            user_item_pairs.astype(np.int32)
        )
        
        return np.clip(predictions, 0, 5)


class NMFCollaborativeFiltering(BaseCollaborativeFilteringModel):
    """
    NMF-based Matrix Factorization Collaborative Filtering.
    Uses non-negative factorization.
    """
    
    def __init__(
        self,
        latent_dim: int = 50,
        regularization: float = 0.01,
        **kwargs
    ):
        """Initialize NMF CF."""
        super().__init__("NMFBasedCF", **kwargs)
        self.latent_dim = latent_dim
        self.regularization = regularization
    
    def fit(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        epochs: int = 10,
        progress_bar: bool = True
    ) -> None:
        """
        Fit NMF CF model.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            ratings: Shape (n_samples,) array of ratings
            n_users: Total number of users
            n_items: Total number of items
            epochs: Number of training epochs
            progress_bar: Show progress bar
        """
        if not CYTHON_AVAILABLE:
            raise RuntimeError("Cython modules not available")
        
        logger.info(f"[{self.model_name}] Fitting with {len(ratings)} ratings for {epochs} epochs")
        
        # Initialize Cython model
        self.model = NMFBasedCF(
            n_users,
            n_items,
            latent_dim=self.latent_dim,
            regularization=self.regularization
        )
        
        # Fit model
        self.model.fit(
            user_item_pairs.astype(np.int32),
            ratings.astype(np.float64),
            epochs=epochs
        )
        
        self.training_history['epochs'] = epochs
        self.training_history['fit_samples'] = len(ratings)
        self.log_params({
            'latent_dim': self.latent_dim,
            'regularization': self.regularization,
            'epochs': epochs
        })
        
        logger.info(f"[{self.model_name}] Fitting completed")
    
    def predict(
        self,
        user_item_pairs: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            progress_bar: Show progress bar
        
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        logger.info(f"[{self.model_name}] Making predictions for {len(user_item_pairs)} pairs")
        
        predictions = self.model.predict_ratings(
            user_item_pairs.astype(np.int32)
        )
        
        return np.clip(predictions, 0, 5)


class CollaborativeFilteringEnsemble(BaseCollaborativeFilteringModel):
    """
    Ensemble of multiple collaborative filtering models.
    Combines predictions from different CF algorithms.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialize ensemble.
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        super().__init__("CFEnsemble", **kwargs)
        self.weights = weights or {
            'UserBasedCF': 0.25,
            'ItemBasedCF': 0.25,
            'SVDBasedCF': 0.25,
            'NMFBasedCF': 0.25
        }
        self.models: Dict[str, BaseCollaborativeFilteringModel] = {}
    
    @property
    def ensemble_weights(self) -> Dict[str, float]:
        """Get ensemble weights."""
        return self.weights
    
    def fit(
        self,
        user_item_pairs: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        **kwargs
    ) -> None:
        """Fit all models in ensemble."""
        logger.info("[CFEnsemble] Fitting all models")
        
        # User-Based CF
        ub_cf = UserBasedCollaborativeFiltering(n_neighbors=10, verbose=self.verbose)
        ub_cf.fit(user_item_pairs, ratings, n_users, n_items)
        self.models['UserBasedCF'] = ub_cf
        
        # Item-Based CF
        ib_cf = ItemBasedCollaborativeFiltering(n_neighbors=10, verbose=self.verbose)
        ib_cf.fit(user_item_pairs, ratings, n_users, n_items)
        self.models['ItemBasedCF'] = ib_cf
        
        # SVD CF
        svd_cf = SVDCollaborativeFiltering(latent_dim=50, verbose=self.verbose)
        svd_cf.fit(user_item_pairs, ratings, n_users, n_items, epochs=10)
        self.models['SVDBasedCF'] = svd_cf
        
        # NMF CF
        nmf_cf = NMFCollaborativeFiltering(latent_dim=50, verbose=self.verbose)
        nmf_cf.fit(user_item_pairs, ratings, n_users, n_items, epochs=10)
        self.models['NMFBasedCF'] = nmf_cf
        
        logger.info("[CFEnsemble] All models fitted")
    
    def predict(
        self,
        user_item_pairs: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Predict using ensemble.
        
        Args:
            user_item_pairs: Shape (n_samples, 2) array of (user_id, item_id)
            progress_bar: Show progress bar
        
        Returns:
            Ensemble predictions
        """
        logger.info("[CFEnsemble] Making ensemble predictions")
        
        predictions = np.zeros(len(user_item_pairs))
        total_weight = sum(self.weights.values())
        
        iterator = tqdm(self.models.items(), desc="Ensemble prediction", colour='green') if progress_bar else self.models.items()
        
        for model_name, model in iterator:
            weight = self.weights.get(model_name, 0.25)
            pred = model.predict(user_item_pairs, progress_bar=False)
            predictions += (weight / total_weight) * pred
        
        return predictions