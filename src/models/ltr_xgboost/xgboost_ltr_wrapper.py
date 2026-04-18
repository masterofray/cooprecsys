'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Python Wrapper API for XGBoost LTR Ranker
High-level interface to Cython modules
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
import xgboost as xgb
from pathlib import Path
import pickle
import json

# Import Cython modules
try:
    from .cython_modules.feature_engineering import FeatureEngineer
    from .cython_modules.model_trainer import XGBoostLTRTrainer, RankingMetricsCalculator
    from .cython_modules.predictor import BatchPredictor, RankingFormatter
except ImportError:
    logging.warning("Cython modules not compiled. Ensure setup.py has been run.")

logger = logging.getLogger(__name__)


class XGBoostLTRWrapper:
    """
    Production-ready wrapper for XGBoost Learning-to-Rank model
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize wrapper
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.trainer = XGBoostLTRTrainer()
        self.predictor = BatchPredictor(batch_size=self.config.get('batch_size', 1000))
        self.formatter = RankingFormatter(top_k=self.config.get('top_k', 10))
        self.metrics_calc = RankingMetricsCalculator(k=self.config.get('eval_k', 10))
        self.feature_names = []
        self.is_fitted = False
        
        logger.info("XGBoostLTRWrapper initialized")
    
    def fit_features(self, X: Union[np.ndarray, pd.DataFrame]) -> 'XGBoostLTRWrapper':
        """
        Fit feature engineering statistics
        
        Parameters
        ----------
        X : ndarray or DataFrame
            Feature matrix
            
        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        X = np.asarray(X, dtype=np.float64)
        self.feature_engineer.fit(X)
        
        logger.info(f"Features fitted: {X.shape[1]} features, {X.shape[0]} samples")
        return self
    
    def normalize_features(self, X: Union[np.ndarray, pd.DataFrame], 
                          method: str = 'minmax') -> np.ndarray:
        """
        Normalize features
        
        Parameters
        ----------
        X : ndarray or DataFrame
            Feature matrix
        method : str
            'minmax' or 'zscore'
            
        Returns
        -------
        ndarray
            Normalized features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X, dtype=np.float64)
        
        if method == 'minmax':
            X_normalized = self.feature_engineer.normalize_minmax(X)
        elif method == 'zscore':
            X_normalized = self.feature_engineer.normalize_zscore(X)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Features normalized using {method} method")
        return X_normalized
    
    def create_interactions(self, X: np.ndarray, feature_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create interaction features
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        feature_pairs : list of tuples
            Feature pairs for interactions
            
        Returns
        -------
        ndarray
            Interaction features
        """
        X = np.asarray(X, dtype=np.float64)
        X_interactions = self.feature_engineer.create_interaction_features(X, feature_pairs)
        
        logger.info(f"Created {len(feature_pairs)} interaction features")
        return X_interactions
    
    def train_model(self, X_train: Union[np.ndarray, pd.DataFrame], 
                   y_train: Union[np.ndarray, pd.Series],
                   X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                   y_val: Optional[Union[np.ndarray, pd.Series]] = None,
                   params: Optional[Dict] = None,
                   epochs: int = 100,
                   early_stopping_rounds: int = 10) -> 'XGBoostLTRWrapper':
        """
        Train XGBoost LTR model
        
        Parameters
        ----------
        X_train : ndarray or DataFrame
            Training features
        y_train : ndarray or Series
            Training labels
        X_val : ndarray or DataFrame, optional
            Validation features
        y_val : ndarray or Series, optional
            Validation labels
        params : dict, optional
            Model hyperparameters
        epochs : int
            Number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
            
        Returns
        -------
        self
        """
        # Convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
        
        # Set default parameters if not provided
        if params is None:
            params = {
                'objective': 'rank:ndcg',
                'metric': 'ndcg@10',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',
                'device': 'cpu'
            }
        
        self.trainer.set_params(params)
        self.trainer.train(X_train, y_train, X_val, y_val, epochs, early_stopping_rounds)
        self.predictor.set_model(self.trainer.model)
        self.is_fitted = True
        
        logger.info(f"Model trained successfully. Shape: {X_train.shape}")
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions
        
        Parameters
        ----------
        X : ndarray or DataFrame
            Feature matrix
            
        Returns
        -------
        ndarray
            Prediction scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        predictions = self.predictor.predict_batch(X)
        logger.info(f"Predictions made for {len(predictions)} samples")
        return predictions
    
    def predict_and_rank(self, X: np.ndarray, group_ids: np.ndarray, 
                        top_k: int = 10) -> Dict:
        """
        Predict and rank within groups
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        group_ids : ndarray
            Group identifiers
        top_k : int
            Top K results per group
            
        Returns
        -------
        dict
            Ranking results by group
        """
        X = np.asarray(X, dtype=np.float64)
        group_ids = np.asarray(group_ids, dtype=np.int32)
        
        rankings = self.predictor.predict_with_ranking(X, group_ids, top_k)
        logger.info(f"Rankings generated for {len(rankings)} groups")
        return rankings
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores
        
        Parameters
        ----------
        importance_type : str
            'gain', 'weight', 'cover', 'total_gain', 'total_cover'
            
        Returns
        -------
        dict
            Feature importance dictionary
        """
        importance = self.trainer.get_feature_importance(importance_type)
        
        # Map feature indices to names if available
        if self.feature_names:
            importance_named = {self.feature_names[int(k.split('_')[1])]: v 
                               for k, v in importance.items()}
            return importance_named
        
        return importance
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       k: int = 10) -> Dict[str, float]:
        """
        Compute ranking metrics
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metrics
            
        Returns
        -------
        dict
            Computed metrics
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        metrics = {
            'ndcg@k': self.metrics_calc.compute_ndcg(y_true, y_pred),
            'map@k': self.metrics_calc.compute_map(y_true, y_pred),
            'mrr@k': self.metrics_calc.compute_mrr(y_true, y_pred),
            'precision@k': self.metrics_calc.compute_precision(y_true, y_pred),
            'recall@k': self.metrics_calc.compute_recall(y_true, y_pred)
        }
        
        logger.info(f"Metrics computed: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        self.trainer.save_model(filepath)
        
        # Save feature names
        config_path = Path(filepath).parent / 'config.pkl'
        with open(config_path, 'wb') as f:
            pickle.dump({'feature_names': self.feature_names}, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load pre-trained model
        
        Parameters
        ----------
        filepath : str
            Path to model file
        """
        self.trainer.load_model(filepath)
        self.predictor.set_model(self.trainer.model)
        
        # Load feature names
        config_path = Path(filepath).parent / 'config.pkl'
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                self.feature_names = config.get('feature_names', [])
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_params(self) -> Dict:
        """
        Get current model parameters
        
        Returns
        -------
        dict
            Model parameters
        """
        return self.trainer.best_params
    
    def set_model_params(self, params: Dict):
        """
        Set model parameters
        
        Parameters
        ----------
        params : dict
            Model hyperparameters
        """
        self.trainer.set_params(params)
        logger.info("Model parameters updated")