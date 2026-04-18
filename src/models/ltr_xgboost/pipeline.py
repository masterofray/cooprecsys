'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Production Pipeline for XGBoost LTR Recommender
Orchestrates data loading, feature engineering, model training, and evaluation
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import duckdb
import mlflow
import mlflow.xgboost
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import configparser
from datetime import datetime

from xgboost_ltr_wrapper import XGBoostLTRWrapper
from visualization import RankingVisualizer
from metrics import MetricsDB
from explainability import ExplainabilityEngine


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class RecommenderPipeline:
    """
    Production-grade recommendation pipeline
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Parameters
        ----------
        config_path : str
            Path to configuration .ini file
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.db_path = self.config.get('database', 'db_path', fallback='data.db')
        self.output_dir = Path(self.config.get('output', 'output_dir', fallback='./output'))
        self.output_dir.mkdir(exist_ok=True)
        
        self.wrapper = XGBoostLTRWrapper(self._get_model_config())
        self.visualizer = RankingVisualizer()
        self.metrics_db = MetricsDB(self.output_dir / 'metrics.db')
        self.explainer = ExplainabilityEngine()
        
        # MLflow setup
        mlflow.set_experiment(self.config.get('mlflow', 'experiment_name', fallback='xgboost_ltr'))
        
        logger.info(f"Pipeline initialized with config: {config_path}")
    
    def _get_model_config(self) -> Dict:
        """Get model configuration from config file"""
        return {
            'batch_size': self.config.getint('model', 'batch_size', fallback=1000),
            'top_k': self.config.getint('model', 'top_k', fallback=10),
            'eval_k': self.config.getint('model', 'eval_k', fallback=10)
        }
    
    def load_data_from_duckdb(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from DuckDB
        
        Parameters
        ----------
        query : str
            SQL query to fetch data
            
        Returns
        -------
        tuple
            (features, labels)
        """
        logger.info(f"Loading data from DuckDB using query")
        
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            result = con.execute(query).fetch_arrow_table()
            df = result.to_pandas()
            con.close()
            
            logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from DuckDB: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    feature_cols: Optional[List[str]] = None,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
        """
        Prepare data for training with train/val/test split
        
        Parameters
        ----------
        df : DataFrame
            Input data
        target_col : str
            Target column name
        feature_cols : list, optional
            List of feature columns
        train_ratio : float
            Training set ratio
        val_ratio : float
            Validation set ratio
            
        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Preparing data for training")
        
        # Handle feature columns
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def engineer_features(self, X_train: np.ndarray, X_test: np.ndarray,
                         normalize: bool = True,
                         create_interactions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply feature engineering
        
        Parameters
        ----------
        X_train : ndarray
            Training features
        X_test : ndarray
            Test features
        normalize : bool
            Whether to normalize features
        create_interactions : bool
            Whether to create interaction features
            
        Returns
        -------
        tuple
            (X_train_engineered, X_test_engineered)
        """
        logger.info("Engineering features")
        
        self.wrapper.fit_features(X_train)
        
        if normalize:
            X_train = self.wrapper.normalize_features(X_train, method='minmax')
            X_test = self.wrapper.normalize_features(X_test, method='minmax')
            logger.info("Features normalized")
        
        if create_interactions:
            n_features = X_train.shape[1]
            feature_pairs = [(i, j) for i in range(min(5, n_features)) 
                           for j in range(i+1, min(5, n_features))]
            
            X_train_inter = self.wrapper.create_interactions(X_train, feature_pairs)
            X_test_inter = self.wrapper.create_interactions(X_test, feature_pairs)
            
            X_train = np.hstack([X_train, X_train_inter])
            X_test = np.hstack([X_test, X_test_inter])
            logger.info(f"Interaction features created: {len(feature_pairs)}")
        
        return X_train, X_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   params: Optional[Dict] = None) -> XGBoostLTRWrapper:
        """
        Train XGBoost LTR model with MLflow tracking
        
        Parameters
        ----------
        X_train, y_train : ndarray
            Training data
        X_val, y_val : ndarray
            Validation data
        params : dict, optional
            Model hyperparameters
            
        Returns
        -------
        XGBoostLTRWrapper
            Trained wrapper
        """
        logger.info("Starting model training")
        
        if params is None:
            params = {
                'objective': 'rank:ndcg',
                'metric': 'ndcg@10',
                'max_depth': self.config.getint('hyperparams', 'max_depth', fallback=6),
                'learning_rate': self.config.getfloat('hyperparams', 'learning_rate', fallback=0.1),
                'subsample': self.config.getfloat('hyperparams', 'subsample', fallback=0.8),
                'colsample_bytree': self.config.getfloat('hyperparams', 'colsample_bytree', fallback=0.8),
                'tree_method': 'hist',
                'device': 'cpu'
            }
        
        with mlflow.start_run():
            mlflow.log_params(params)
            
            self.wrapper.train_model(
                X_train, y_train, X_val, y_val,
                params=params,
                epochs=self.config.getint('hyperparams', 'epochs', fallback=100),
                early_stopping_rounds=self.config.getint('hyperparams', 'early_stopping', fallback=10)
            )
            
            mlflow.log_param('train_samples', len(X_train))
            mlflow.log_param('val_samples', len(X_val))
        
        logger.info("Model training completed")
        return self.wrapper
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Evaluate model on test and validation sets
        
        Parameters
        ----------
        X_test, y_test : ndarray
            Test data
        X_val, y_val : ndarray
            Validation data
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Predictions
        y_pred_test = self.wrapper.predict(X_test)
        y_pred_val = self.wrapper.predict(X_val)
        
        # Metrics
        test_metrics = self.wrapper.compute_metrics(y_test.astype(np.int32), y_pred_test)
        val_metrics = self.wrapper.compute_metrics(y_val.astype(np.int32), y_pred_val)
        
        # Log to MLflow
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f'test_{metric_name}', metric_value)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f'val_{metric_name}', metric_value)
        
        # Store in metrics database
        self.metrics_db.insert_metrics({
            'timestamp': datetime.now().isoformat(),
            'split': 'test',
            'metrics': test_metrics
        })
        
        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Val metrics: {val_metrics}")
        
        return {'test': test_metrics, 'val': val_metrics}
    
    def visualize_results(self, X_test: np.ndarray, y_test: np.ndarray,
                         y_pred: np.ndarray):
        """
        Generate visualizations
        
        Parameters
        ----------
        X_test, y_test : ndarray
            Test data
        y_pred : ndarray
            Model predictions
        """
        logger.info("Generating visualizations")
        
        output_dir = self.output_dir / 'visualizations'
        output_dir.mkdir(exist_ok=True)
        
        try:
            self.visualizer.plot_prediction_distribution(y_pred, save_path=output_dir / 'pred_dist.png')
            self.visualizer.plot_actual_vs_predicted(y_test, y_pred, save_path=output_dir / 'actual_vs_pred.png')
            self.visualizer.plot_feature_importance(self.wrapper.get_feature_importance(), 
                                                    save_path=output_dir / 'feature_importance.png')
            logger.info(f"Visualizations saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def explain_predictions(self, X_test: np.ndarray, y_pred: np.ndarray):
        """
        Generate model explanations
        
        Parameters
        ----------
        X_test : ndarray
            Test features
        y_pred : ndarray
            Model predictions
        """
        logger.info("Generating model explanations")
        
        try:
            shap_values = self.explainer.compute_shap(self.wrapper.trainer.model, X_test)
            explanation = self.explainer.interpret_predictions(shap_values, X_test)
            
            output_file = self.output_dir / 'explanations.json'
            with open(output_file, 'w') as f:
                json.dump(explanation, f, indent=2)
            
            logger.info(f"Explanations saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
    
    def run(self, data_query: str, target_col: str, 
           feature_cols: Optional[List[str]] = None):
        """
        Run complete pipeline
        
        Parameters
        ----------
        data_query : str
            DuckDB SQL query
        target_col : str
            Target column name
        feature_cols : list, optional
            Feature columns
        """
        logger.info("="*50)
        logger.info("Starting XGBoost LTR Recommender Pipeline")
        logger.info("="*50)
        
        try:
            # Load data
            df = self.load_data_from_duckdb(data_query)
            
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
                df, target_col, feature_cols
            )
            
            # Feature engineering
            X_train, X_test = self.engineer_features(X_train, X_test)
            _, X_val_eng = self.engineer_features(X_train, X_val)  # Use train stats
            
            # Train model
            self.train_model(X_train, y_train, X_val_eng, y_val)
            
            # Evaluate
            metrics = self.evaluate_model(X_test, y_test, X_val_eng, y_val)
            
            # Visualize
            y_pred_test = self.wrapper.predict(X_test)
            self.visualize_results(X_test, y_test, y_pred_test)
            
            # Explain
            self.explain_predictions(X_test, y_pred_test)
            
            # Save model
            model_path = self.output_dir / 'model.xgb'
            self.wrapper.save_model(str(model_path))
            
            logger.info("="*50)
            logger.info("Pipeline completed successfully")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <config_path>")
        sys.exit(1)
    
    pipeline = RecommenderPipeline(sys.argv[1])
    
    # Example query - modify based on your data
    query = "SELECT * FROM FullTrainData LIMIT 100000"
    pipeline.run(query, target_col='reordered', feature_cols=None)