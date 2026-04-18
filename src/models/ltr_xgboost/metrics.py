'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Ranking Metrics Module with DuckDB Storage
NDCG@K, MAP@K, MRR, Precision, Recall, and more
"""

import numpy as np
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class MetricsDB:
    """
    Metrics database for tracking model performance
    """
    
    def __init__(self, db_path: str):
        """
        Initialize metrics database
        
        Parameters
        ----------
        db_path : str
            Path to DuckDB database file
        """
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._create_tables()
        logger.info(f"Metrics database initialized at {db_path}")
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_metrics'),
                timestamp VARCHAR,
                model_version VARCHAR,
                split VARCHAR,
                ndcg_at_10 DOUBLE,
                ndcg_at_5 DOUBLE,
                map_at_10 DOUBLE,
                map_at_5 DOUBLE,
                mrr_at_10 DOUBLE,
                precision_at_10 DOUBLE,
                recall_at_10 DOUBLE,
                auc_score DOUBLE,
                n_samples INTEGER,
                metadata VARCHAR
            )
        """)
        
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_metrics START 1
        """)
        
        logger.info("Metrics tables created")
    
    def insert_metrics(self, metrics_dict: Dict):
        """
        Insert metrics record
        
        Parameters
        ----------
        metrics_dict : dict
            Metrics to insert
        """
        timestamp = metrics_dict.get('timestamp', datetime.now().isoformat())
        split = metrics_dict.get('split', 'test')
        metrics = metrics_dict.get('metrics', {})
        
        self.con.execute(
            """
            INSERT INTO metrics (timestamp, split, ndcg_at_10, map_at_10, mrr_at_10,
                               precision_at_10, recall_at_10)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                timestamp,
                split,
                metrics.get('ndcg@k', 0.0),
                metrics.get('map@k', 0.0),
                metrics.get('mrr@k', 0.0),
                metrics.get('precision@k', 0.0),
                metrics.get('recall@k', 0.0)
            ]
        )
        
        logger.info(f"Metrics inserted for split: {split}")
    
    def get_latest_metrics(self, split: str = 'test') -> Dict:
        """
        Get latest metrics for a split
        
        Parameters
        ----------
        split : str
            Data split type
            
        Returns
        -------
        dict
            Latest metrics
        """
        result = self.con.execute(
            f"SELECT * FROM metrics WHERE split = '{split}' ORDER BY timestamp DESC LIMIT 1"
        ).fetchdf()
        
        if len(result) == 0:
            logger.warning(f"No metrics found for split: {split}")
            return {}
        
        return result.iloc[0].to_dict()
    
    def get_metrics_history(self, split: str = 'test', limit: int = 50) -> List[Dict]:
        """
        Get historical metrics
        
        Parameters
        ----------
        split : str
            Data split type
        limit : int
            Number of records to fetch
            
        Returns
        -------
        list
            List of metrics dictionaries
        """
        result = self.con.execute(
            f"""
            SELECT * FROM metrics 
            WHERE split = '{split}' 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """
        ).fetchdf()
        
        return result.to_dict('records')
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """
        Export all metrics to file
        
        Parameters
        ----------
        output_path : str
            Path to output file
        format : str
            'json' or 'csv'
        """
        df = self.con.execute("SELECT * FROM metrics ORDER BY timestamp DESC").fetchdf()
        
        if format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        
        logger.info(f"Metrics exported to {output_path}")
    
    def close(self):
        """Close database connection"""
        self.con.close()
        logger.info("Metrics database closed")


class RankingMetrics:
    """
    Comprehensive ranking metrics computation
    """
    
    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Compute NDCG@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metric
            
        Returns
        -------
        float
            NDCG@K score
        """
        sorted_indices = np.argsort(-y_pred)[:k]
        dcg = np.sum((2 ** y_true[sorted_indices] - 1) / np.log2(np.arange(2, k + 2)))
        
        ideal_indices = np.argsort(-y_true)[:k]
        idcg = np.sum((2 ** y_true[ideal_indices] - 1) / np.log2(np.arange(2, k + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Compute MAP@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores (binary or graded)
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metric
            
        Returns
        -------
        float
            MAP@K score
        """
        sorted_indices = np.argsort(-y_pred)[:k]
        relevant = y_true[sorted_indices] > 0
        
        if np.sum(relevant) == 0:
            return 0.0
        
        precision_at_k = np.cumsum(relevant) / np.arange(1, k + 1)
        return np.sum(precision_at_k * relevant) / min(k, np.sum(y_true > 0))
    
    @staticmethod
    def mrr_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Compute MRR@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metric
            
        Returns
        -------
        float
            MRR@K score
        """
        sorted_indices = np.argsort(-y_pred)[:k]
        relevant = y_true[sorted_indices] > 0
        
        if np.any(relevant):
            return 1.0 / (np.where(relevant)[0][0] + 1)
        return 0.0
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Compute Precision@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metric
            
        Returns
        -------
        float
            Precision@K score
        """
        sorted_indices = np.argsort(-y_pred)[:k]
        relevant = y_true[sorted_indices] > 0
        return np.sum(relevant) / k
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        Compute Recall@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
        k : int
            K for ranking metric
            
        Returns
        -------
        float
            Recall@K score
        """
        sorted_indices = np.argsort(-y_pred)[:k]
        relevant = y_true[sorted_indices] > 0
        total_relevant = np.sum(y_true > 0)
        
        return np.sum(relevant) / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           k: int = 10) -> Dict[str, float]:
        """
        Compute all ranking metrics at once
        
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
            All computed metrics
        """
        return {
            f'ndcg@{k}': RankingMetrics.ndcg_at_k(y_true, y_pred, k),
            f'map@{k}': RankingMetrics.map_at_k(y_true, y_pred, k),
            f'mrr@{k}': RankingMetrics.mrr_at_k(y_true, y_pred, k),
            f'precision@{k}': RankingMetrics.precision_at_k(y_true, y_pred, k),
            f'recall@{k}': RankingMetrics.recall_at_k(y_true, y_pred, k)
        }