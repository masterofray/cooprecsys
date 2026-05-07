"""
Complete Collaborative Filtering Pipeline with visualization and MLflow tracking.
Includes 10+ visualizations, DuckDB integration, and comprehensive logging.
"""

import os
import sys
import logging
import warnings
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import duckdb
import mlflow
import shlex
from joblib import Parallel, delayed, dump, load
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
import cloudpickle

from collaborative_filtering_wrapper import (
    UserBasedCollaborativeFiltering,
    ItemBasedCollaborativeFiltering,
    SVDCollaborativeFiltering,
    NMFCollaborativeFiltering,
    CollaborativeFilteringEnsemble
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure visualization
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class CollaborativeFilteringPipeline:
    """
    Production-grade collaborative filtering pipeline with:
    - Multiple CF algorithm variants
    - Comprehensive evaluation metrics
    - 10+ visualization outputs
    - MLflow tracking
    - Model persistence
    - DuckDB integration
    """
    
    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        output_dir: str = "./cf_outputs",
        mlflow_experiment: str = "collaborative_filtering",
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize CF pipeline.
        
        Args:
            duckdb_path: Path to DuckDB database
            output_dir: Directory for outputs
            mlflow_experiment: MLflow experiment name
            n_jobs: Number of parallel jobs
            verbose: Verbosity flag
        """
        self.duckdb_path = duckdb_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment = mlflow_experiment
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.conn = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.visualizations: List[Path] = []
        
        # MLflow setup
        mlflow.set_experiment(mlflow_experiment)
        
        logger.info("[Pipeline] Initialized with output_dir={self.output_dir}")
    
    @property
    def output_path(self) -> Path:
        """Get output directory path."""
        return self.output_dir
    
    @property
    def models_trained(self) -> List[str]:
        """Get list of trained models."""
        return list(self.models.keys())
    
    def connect_duckdb(self) -> None:
        """Connect to DuckDB database."""
        try:
            if self.duckdb_path:
                self.conn = duckdb.connect(self.duckdb_path)
            else:
                self.conn = duckdb.connect()
            logger.info("[Pipeline] DuckDB connection established")
        except Exception as e:
            logger.error(f"[Pipeline] DuckDB connection failed: {e}")
            raise
    
    def load_data_from_duckdb(
        self,
        table_name: str,
        user_col: str = "user_id",
        item_col: str = "product_id",
        rating_col: str = "total_quantity",
        limit: Optional[int] = None
    ) -> None:
        """
        Load data from DuckDB.
        
        Args:
            table_name: Table name in DuckDB
            user_col: User column name
            item_col: Item column name
            rating_col: Rating column name
            limit: Optional limit on rows
        """
        if self.conn is None:
            self.connect_duckdb()
        
        query = f"""
        SELECT 
            {user_col} as user_id,
            {item_col} as item_id,
            {rating_col} as rating
        FROM {table_name}
        WHERE {rating_col} > 0
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"[Pipeline] Loading data from {table_name}")
        
        try:
            self.data = self.conn.execute(query).df()
            logger.info(f"[Pipeline] Loaded {len(self.data)} interactions")
            logger.info(f"[Pipeline] Users: {self.data['user_id'].nunique()}, Items: {self.data['item_id'].nunique()}")
        except Exception as e:
            logger.error(f"[Pipeline] Data loading failed: {e}")
            raise
    
    def load_data_from_csv(self, csv_path: str, **kwargs) -> None:
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(csv_path)
            logger.info(f"[Pipeline] Loaded {len(self.data)} interactions from {csv_path}")
        except Exception as e:
            logger.error(f"[Pipeline] CSV loading failed: {e}")
            raise
    
    def preprocess_data(
        self,
        min_user_interactions: int = 2,
        min_item_interactions: int = 2,
        normalize_ratings: bool = True
    ) -> None:
        """
        Preprocess data.
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            normalize_ratings: Whether to normalize ratings to 0-5
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data_* first.")
        
        logger.info("[Pipeline] Preprocessing data")
        
        # Filter by user interactions
        user_counts = self.data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        self.data = self.data[self.data['user_id'].isin(valid_users)]
        logger.info(f"[Pipeline] After user filter: {len(self.data)} interactions")
        
        # Filter by item interactions
        item_counts = self.data['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        self.data = self.data[self.data['item_id'].isin(valid_items)]
        logger.info(f"[Pipeline] After item filter: {len(self.data)} interactions")
        
        # Normalize ratings
        if normalize_ratings:
            rating_min = self.data['rating'].min()
            rating_max = self.data['rating'].max()
            self.data['rating'] = 5 * (self.data['rating'] - rating_min) / (rating_max - rating_min)
            logger.info("[Pipeline] Ratings normalized to [0, 5]")
        
        logger.info(f"[Pipeline] Final data shape: {self.data.shape}")
        logger.info(f"[Pipeline] Users: {self.data['user_id'].nunique()}, Items: {self.data['item_id'].nunique()}")
    
    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Test set fraction
            random_state: Random seed
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        logger.info(f"[Pipeline] Splitting data with test_size={test_size}")
        
        self.train_data, self.test_data = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"[Pipeline] Train: {len(self.train_data)}, Test: {len(self.test_data)}")
    
    def train_models(self) -> None:
        """Train all CF model variants."""
        if self.train_data is None:
            raise ValueError("Data not split. Call split_data first.")
        
        logger.info("[Pipeline] Starting model training")
        
        user_item_pairs = self.train_data[['user_id', 'item_id']].values
        ratings = self.train_data['rating'].values
        n_users = self.data['user_id'].max() + 1
        n_items = self.data['item_id'].max() + 1
        
        mlflow.start_run(run_name=f"cf_training_{datetime.now().isoformat()}")
        
        try:
            # User-Based CF
            logger.info("[Pipeline] Training User-Based CF")
            ub_cf = UserBasedCollaborativeFiltering(n_neighbors=10, verbose=self.verbose)
            with tqdm_joblib(desc="User-Based CF", total=1, colour='green'):
                ub_cf.fit(user_item_pairs, ratings, n_users, n_items)
            self.models['UserBasedCF'] = ub_cf
            
            # Item-Based CF
            logger.info("[Pipeline] Training Item-Based CF")
            ib_cf = ItemBasedCollaborativeFiltering(n_neighbors=10, verbose=self.verbose)
            with tqdm_joblib(desc="Item-Based CF", total=1, colour='green'):
                ib_cf.fit(user_item_pairs, ratings, n_users, n_items)
            self.models['ItemBasedCF'] = ib_cf
            
            # SVD CF
            logger.info("[Pipeline] Training SVD CF")
            svd_cf = SVDCollaborativeFiltering(
                latent_dim=50,
                learning_rate=0.01,
                regularization=0.01,
                verbose=self.verbose
            )
            with tqdm_joblib(desc="SVD CF", total=10, colour='green'):
                svd_cf.fit(user_item_pairs, ratings, n_users, n_items, epochs=10)
            self.models['SVDBasedCF'] = svd_cf
            
            # NMF CF
            logger.info("[Pipeline] Training NMF CF")
            nmf_cf = NMFCollaborativeFiltering(
                latent_dim=50,
                regularization=0.01,
                verbose=self.verbose
            )
            with tqdm_joblib(desc="NMF CF", total=10, colour='green'):
                nmf_cf.fit(user_item_pairs, ratings, n_users, n_items, epochs=10)
            self.models['NMFBasedCF'] = nmf_cf
            
            # Ensemble
            logger.info("[Pipeline] Creating Ensemble CF")
            ensemble = CollaborativeFilteringEnsemble(verbose=self.verbose)
            with tqdm_joblib(desc="Ensemble CF", total=1, colour='green'):
                ensemble.fit(user_item_pairs, ratings, n_users, n_items)
            self.models['EnsembleCF'] = ensemble
            
            logger.info("[Pipeline] All models trained successfully")
            mlflow.log_param("n_models", len(self.models))
            
        finally:
            mlflow.end_run()
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models."""
        if len(self.models) == 0:
            raise ValueError("No models trained. Call train_models first.")
        
        if self.test_data is None:
            raise ValueError("No test data. Call split_data first.")
        
        logger.info("[Pipeline] Evaluating models")
        
        user_item_pairs = self.test_data[['user_id', 'item_id']].values
        ground_truth = self.test_data['rating'].values
        
        mlflow.start_run(run_name=f"cf_evaluation_{datetime.now().isoformat()}")
        
        try:
            with tqdm_joblib(desc="Model evaluation", total=len(self.models), colour='green'):
                for model_name, model in tqdm(self.models.items(), desc="Evaluating", colour='green'):
                    logger.info(f"[Pipeline] Evaluating {model_name}")
                    
                    predictions = model.predict(user_item_pairs, progress_bar=False)
                    
                    # Compute metrics
                    mse = mean_squared_error(ground_truth, predictions)
                    mae = mean_absolute_error(ground_truth, predictions)
                    rmse = np.sqrt(mse)
                    
                    self.metrics[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse
                    }
                    
                    mlflow.log_metrics({
                        f"{model_name}_mse": mse,
                        f"{model_name}_mae": mae,
                        f"{model_name}_rmse": rmse
                    })
                    
                    logger.info(f"[Pipeline] {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
        finally:
            mlflow.end_run()
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        logger.info("[Pipeline] Creating visualizations")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Data Distribution
        self._viz_data_distribution(viz_dir)
        
        # 2. User-Item Matrix Sparsity
        self._viz_sparsity(viz_dir)
        
        # 3. Rating Distribution
        self._viz_rating_distribution(viz_dir)
        
        # 4. User Interactions Distribution
        self._viz_user_interactions(viz_dir)
        
        # 5. Item Interactions Distribution
        self._viz_item_interactions(viz_dir)
        
        # 6. Model Metrics Comparison
        self._viz_model_metrics(viz_dir)
        
        # 7. Prediction Error Distribution
        self._viz_prediction_errors(viz_dir)
        
        # 8. User Engagement
        self._viz_user_engagement(viz_dir)
        
        # 9. Item Popularity
        self._viz_item_popularity(viz_dir)
        
        # 10. Train-Test Split Distribution
        self._viz_train_test_split(viz_dir)
        
        # 11. Model Predictions Correlation
        self._viz_predictions_correlation(viz_dir)
        
        # 12. Latent Factor Heatmap
        self._viz_latent_factors(viz_dir)
        
        logger.info(f"[Pipeline] Created {len(list(viz_dir.glob('*.png')))} visualizations")
    
    def _viz_data_distribution(self, viz_dir: Path) -> None:
        """Visualization 1: Data distribution overview."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Distribution Overview', fontsize=16, fontweight='bold')
        
        # Users per item
        ax = axes[0, 0]
        users_per_item = self.data.groupby('item_id').size()
        ax.hist(users_per_item, bins=30, edgecolor='black', color='#1f77b4')
        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Frequency')
        ax.set_title('Users per Item Distribution')
        ax.grid(alpha=0.3)
        
        # Items per user
        ax = axes[0, 1]
        items_per_user = self.data.groupby('user_id').size()
        ax.hist(items_per_user, bins=30, edgecolor='black', color='#ff7f0e')
        ax.set_xlabel('Number of Items')
        ax.set_ylabel('Frequency')
        ax.set_title('Items per User Distribution')
        ax.grid(alpha=0.3)
        
        # Cumulative users
        ax = axes[1, 0]
        sorted_users = np.sort(users_per_item.values)[::-1]
        cumsum = np.cumsum(sorted_users)
        ax.plot(cumsum / cumsum[-1], linewidth=2, color='#2ca02c')
        ax.set_xlabel('Fraction of Items')
        ax.set_ylabel('Fraction of Interactions')
        ax.set_title('Cumulative Users Contribution')
        ax.grid(alpha=0.3)
        
        # Cumulative items
        ax = axes[1, 1]
        sorted_items = np.sort(items_per_user.values)[::-1]
        cumsum = np.cumsum(sorted_items)
        ax.plot(cumsum / cumsum[-1], linewidth=2, color='#d62728')
        ax.set_xlabel('Fraction of Users')
        ax.set_ylabel('Fraction of Interactions')
        ax.set_title('Cumulative Items Contribution')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / '01_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_sparsity(self, viz_dir: Path) -> None:
        """Visualization 2: User-Item Matrix Sparsity."""
        n_users = self.data['user_id'].max() + 1
        n_items = self.data['item_id'].max() + 1
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('User-Item Matrix Sparsity', fontsize=16, fontweight='bold')
        
        sparsity = 1 - (len(self.data) / (n_users * n_items))
        
        # Sparsity metric
        ax = axes[0]
        categories = ['Filled', 'Empty']
        values = [1 - sparsity, sparsity]
        colors = ['#2ca02c', '#d62728']
        ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Fraction')
        ax.set_title(f'Matrix Sparsity: {sparsity:.4f}')
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Matrix size info
        ax = axes[1]
        ax.axis('off')
        info_text = f"""
        Matrix Dimensions: {n_users} × {n_items}
        Total Elements: {n_users * n_items:,}
        Filled Elements: {len(self.data):,}
        Empty Elements: {n_users * n_items - len(self.data):,}
        Sparsity Ratio: {sparsity:.6f}
        Density: {1 - sparsity:.6f}
        """
        ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '02_sparsity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_rating_distribution(self, viz_dir: Path) -> None:
        """Visualization 3: Rating Distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Rating Distribution', fontsize=16, fontweight='bold')
        
        # Histogram
        ax = axes[0]
        ax.hist(self.data['rating'], bins=30, edgecolor='black', color='#1f77b4', alpha=0.7)
        ax.set_xlabel('Rating Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Rating Values Histogram')
        ax.grid(alpha=0.3)
        
        # Box plot
        ax = axes[1]
        data_train = self.train_data['rating'] if self.train_data is not None else self.data['rating']
        data_test = self.test_data['rating'] if self.test_data is not None else None
        
        if data_test is not None:
            bp = ax.boxplot([data_train, data_test], labels=['Train', 'Test'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2ca02c', '#ff7f0e']):
                patch.set_facecolor(color)
        else:
            bp = ax.boxplot(data_train, labels=['All Data'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ca02c')
        
        ax.set_ylabel('Rating Value')
        ax.set_title('Rating Distribution by Set')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '03_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_user_interactions(self, viz_dir: Path) -> None:
        """Visualization 4: User Interactions Distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('User Interactions Distribution', fontsize=16, fontweight='bold')
        
        user_counts = self.data.groupby('user_id').size()
        
        # CDF
        ax = axes[0, 0]
        sorted_counts = np.sort(user_counts.values)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax.plot(sorted_counts, cdf, linewidth=2, color='#1f77b4')
        ax.set_xlabel('Interactions per User')
        ax.set_ylabel('CDF')
        ax.set_title('Cumulative Distribution of User Interactions')
        ax.grid(alpha=0.3)
        
        # Histogram
        ax = axes[0, 1]
        ax.hist(user_counts, bins=50, edgecolor='black', color='#ff7f0e', alpha=0.7)
        ax.set_xlabel('Interactions per User')
        ax.set_ylabel('Frequency')
        ax.set_title('User Interactions Histogram')
        ax.grid(alpha=0.3)
        
        # Statistics
        ax = axes[1, 0]
        stats = {
            'Mean': user_counts.mean(),
            'Median': user_counts.median(),
            'Std': user_counts.std(),
            'Min': user_counts.min(),
            'Max': user_counts.max()
        }
        ax.axis('off')
        stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in stats.items()])
        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                family='monospace', fontweight='bold')
        ax.set_title('Statistics')
        
        # Top users
        ax = axes[1, 1]
        top_users = user_counts.nlargest(10)
        ax.barh(range(len(top_users)), top_users.values, color='#2ca02c', edgecolor='black')
        ax.set_yticks(range(len(top_users)))
        ax.set_yticklabels([f'User {uid}' for uid in top_users.index])
        ax.set_xlabel('Interactions')
        ax.set_title('Top 10 Most Active Users')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '04_user_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_item_interactions(self, viz_dir: Path) -> None:
        """Visualization 5: Item Interactions Distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Item Interactions Distribution', fontsize=16, fontweight='bold')
        
        item_counts = self.data.groupby('item_id').size()
        
        # CDF
        ax = axes[0, 0]
        sorted_counts = np.sort(item_counts.values)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax.plot(sorted_counts, cdf, linewidth=2, color='#d62728')
        ax.set_xlabel('Interactions per Item')
        ax.set_ylabel('CDF')
        ax.set_title('Cumulative Distribution of Item Interactions')
        ax.grid(alpha=0.3)
        
        # Histogram
        ax = axes[0, 1]
        ax.hist(item_counts, bins=50, edgecolor='black', color='#9467bd', alpha=0.7)
        ax.set_xlabel('Interactions per Item')
        ax.set_ylabel('Frequency')
        ax.set_title('Item Interactions Histogram')
        ax.grid(alpha=0.3)
        
        # Statistics
        ax = axes[1, 0]
        stats = {
            'Mean': item_counts.mean(),
            'Median': item_counts.median(),
            'Std': item_counts.std(),
            'Min': item_counts.min(),
            'Max': item_counts.max()
        }
        ax.axis('off')
        stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in stats.items()])
        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
                family='monospace', fontweight='bold')
        ax.set_title('Statistics')
        
        # Top items
        ax = axes[1, 1]
        top_items = item_counts.nlargest(10)
        ax.barh(range(len(top_items)), top_items.values, color='#17becf', edgecolor='black')
        ax.set_yticks(range(len(top_items)))
        ax.set_yticklabels([f'Item {iid}' for iid in top_items.index])
        ax.set_xlabel('Interactions')
        ax.set_title('Top 10 Most Popular Items')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '05_item_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_model_metrics(self, viz_dir: Path) -> None:
        """Visualization 6: Model Metrics Comparison."""
        if len(self.metrics) == 0:
            logger.warning("[Pipeline] No metrics available for visualization")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(self.metrics.keys())
        rmse_values = [self.metrics[m]['rmse'] for m in model_names]
        mae_values = [self.metrics[m]['mae'] for m in model_names]
        mse_values = [self.metrics[m]['mse'] for m in model_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        # RMSE
        ax = axes[0]
        ax.bar(model_names, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Squared Error')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(rmse_values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        
        # MAE
        ax = axes[1]
        ax.bar(model_names, mae_values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(mae_values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        
        # MSE
        ax = axes[2]
        ax.bar(model_names, mse_values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('MSE')
        ax.set_title('Mean Squared Error')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(mse_values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '06_model_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_prediction_errors(self, viz_dir: Path) -> None:
        """Visualization 7: Prediction Error Distribution."""
        if len(self.models) == 0 or self.test_data is None:
            logger.warning("[Pipeline] Cannot create prediction error visualization")
            return
        
        user_item_pairs = self.test_data[['user_id', 'item_id']].values
        ground_truth = self.test_data['rating'].values
        
        n_models = min(len(self.models), 4)  # Show up to 4 models
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(list(self.models.items())[:n_models]):
            predictions = model.predict(user_item_pairs, progress_bar=False)
            errors = predictions - ground_truth
            
            ax = axes[idx]
            ax.hist(errors, bins=40, edgecolor='black', color='#1f77b4', alpha=0.7)
            ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
            ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model_name} - Error Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '07_prediction_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_user_engagement(self, viz_dir: Path) -> None:
        """Visualization 8: User Engagement Patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('User Engagement Patterns', fontsize=16, fontweight='bold')
        
        # Engagement quartiles
        ax = axes[0, 0]
        user_counts = self.data.groupby('user_id').size()
        quartiles = pd.qcut(user_counts, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_counts = quartiles.value_counts().sort_index()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quartile_counts)))
        ax.bar(quartile_counts.index, quartile_counts.values, color=colors, edgecolor='black')
        ax.set_ylabel('Number of Users')
        ax.set_title('Users by Engagement Quartile')
        ax.grid(alpha=0.3, axis='y')
        
        # Interaction rate over time (if applicable)
        ax = axes[0, 1]
        daily_interactions = self.data.groupby('user_id').size().describe()
        ax.axis('off')
        desc_text = daily_interactions.to_string()
        ax.text(0.1, 0.5, desc_text, ha='left', va='center', fontsize=10,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_title('User Engagement Statistics')
        
        # Gini coefficient
        ax = axes[1, 0]
        sorted_counts = np.sort(user_counts.values)
        cum_counts = np.cumsum(sorted_counts)
        ax.plot(np.arange(len(sorted_counts)) / len(sorted_counts),
                cum_counts / cum_counts[-1],
                linewidth=2, color='#1f77b4', label='Lorenz Curve')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Equality')
        ax.fill_between(np.arange(len(sorted_counts)) / len(sorted_counts),
                        cum_counts / cum_counts[-1],
                        np.arange(len(sorted_counts)) / len(sorted_counts),
                        alpha=0.3, color='#1f77b4')
        ax.set_xlabel('Fraction of Users')
        ax.set_ylabel('Fraction of Interactions')
        ax.set_title('Interaction Inequality (Lorenz Curve)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # User satisfaction proxy
        ax = axes[1, 1]
        user_avg_ratings = self.data.groupby('user_id')['rating'].agg(['mean', 'std', 'count'])
        ax.scatter(user_avg_ratings['count'], user_avg_ratings['mean'], 
                  alpha=0.6, s=50, color='#ff7f0e', edgecolor='black')
        ax.set_xlabel('Number of Interactions')
        ax.set_ylabel('Average Rating')
        ax.set_title('User Satisfaction vs Activity')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / '08_user_engagement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_item_popularity(self, viz_dir: Path) -> None:
        """Visualization 9: Item Popularity Trends."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Item Popularity Analysis', fontsize=16, fontweight='bold')
        
        # Top vs long tail
        ax = axes[0, 0]
        item_counts = self.data.groupby('item_id').size().sort_values(ascending=False)
        threshold = item_counts.median()
        popular = (item_counts > threshold).sum()
        tail = (item_counts <= threshold).sum()
        ax.pie([popular, tail], labels=['Popular', 'Long Tail'], autopct='%1.1f%%',
               colors=['#ff7f0e', '#1f77b4'], startangle=90)
        ax.set_title('Popular vs Long Tail Items')
        
        # Power law
        ax = axes[0, 1]
        item_ranks = np.arange(1, len(item_counts) + 1)
        ax.loglog(item_ranks, item_counts.values, 'o', color='#2ca02c', markersize=4, alpha=0.6)
        ax.set_xlabel('Item Rank')
        ax.set_ylabel('Popularity (Number of Interactions)')
        ax.set_title('Item Popularity Power Law')
        ax.grid(alpha=0.3, which='both')
        
        # Rating vs popularity
        ax = axes[1, 0]
        item_stats = self.data.groupby('item_id').agg({'rating': ['mean', 'count']}).reset_index()
        item_stats.columns = ['item_id', 'avg_rating', 'popularity']
        ax.scatter(item_stats['popularity'], item_stats['avg_rating'],
                  alpha=0.6, s=50, color='#d62728', edgecolor='black')
        ax.set_xlabel('Popularity (Number of Interactions)')
        ax.set_ylabel('Average Rating')
        ax.set_title('Item Rating vs Popularity')
        ax.grid(alpha=0.3)
        
        # Rating distribution
        ax = axes[1, 1]
        item_avg_ratings = self.data.groupby('item_id')['rating'].mean()
        ax.hist(item_avg_ratings, bins=30, edgecolor='black', color='#9467bd', alpha=0.7)
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Number of Items')
        ax.set_title('Item Average Rating Distribution')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / '09_item_popularity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_train_test_split(self, viz_dir: Path) -> None:
        """Visualization 10: Train-Test Split Characteristics."""
        if self.train_data is None or self.test_data is None:
            logger.warning("[Pipeline] Cannot create train-test split visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Train-Test Split Analysis', fontsize=16, fontweight='bold')
        
        # Size comparison
        ax = axes[0, 0]
        sizes = [len(self.train_data), len(self.test_data)]
        colors = ['#2ca02c', '#ff7f0e']
        bars = ax.bar(['Train', 'Test'], sizes, color=colors, edgecolor='black', linewidth=2)
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(size):,}',
                   ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Number of Interactions')
        ax.set_title('Train-Test Split Size')
        ax.grid(alpha=0.3, axis='y')
        
        # User overlap
        ax = axes[0, 1]
        train_users = set(self.train_data['user_id'].unique())
        test_users = set(self.test_data['user_id'].unique())
        overlap_users = len(train_users & test_users)
        only_train = len(train_users - test_users)
        only_test = len(test_users - train_users)
        
        data = [only_train, overlap_users, only_test]
        labels = [f'Only Train\n({only_train})', f'Overlap\n({overlap_users})', f'Only Test\n({only_test})']
        ax.bar(labels, data, color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Users')
        ax.set_title('User Distribution')
        ax.grid(alpha=0.3, axis='y')
        
        # Item overlap
        ax = axes[1, 0]
        train_items = set(self.train_data['item_id'].unique())
        test_items = set(self.test_data['item_id'].unique())
        overlap_items = len(train_items & test_items)
        only_train_items = len(train_items - test_items)
        only_test_items = len(test_items - train_items)
        
        data = [only_train_items, overlap_items, only_test_items]
        labels = [f'Only Train\n({only_train_items})', f'Overlap\n({overlap_items})', f'Only Test\n({only_test_items})']
        ax.bar(labels, data, color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Items')
        ax.set_title('Item Distribution')
        ax.grid(alpha=0.3, axis='y')
        
        # Rating distribution
        ax = axes[1, 1]
        ax.hist(self.train_data['rating'], bins=20, alpha=0.6, label='Train', color='#2ca02c', edgecolor='black')
        ax.hist(self.test_data['rating'], bins=20, alpha=0.6, label='Test', color='#ff7f0e', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Rating Distribution Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / '10_train_test_split.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_predictions_correlation(self, viz_dir: Path) -> None:
        """Visualization 11: Model Predictions Correlation."""
        if len(self.models) < 2 or self.test_data is None:
            logger.warning("[Pipeline] Cannot create predictions correlation visualization")
            return
        
        user_item_pairs = self.test_data[['user_id', 'item_id']].values
        
        n_models = min(len(self.models), 4)
        fig, axes = plt.subplots(n_models, n_models, figsize=(14, 14))
        fig.suptitle('Model Predictions Correlation Matrix', fontsize=16, fontweight='bold')
        
        model_list = list(self.models.items())[:n_models]
        predictions_dict = {}
        
        for model_name, model in model_list:
            predictions_dict[model_name] = model.predict(user_item_pairs, progress_bar=False)
        
        for i, (name_i, model_i) in enumerate(model_list):
            for j, (name_j, model_j) in enumerate(model_list):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(predictions_dict[name_i], bins=20, color='#1f77b4', edgecolor='black', alpha=0.7)
                    ax.set_title(name_i, fontweight='bold')
                    ax.set_ylabel('Frequency')
                else:
                    ax.scatter(predictions_dict[name_j], predictions_dict[name_i],
                             alpha=0.3, s=10, color='#ff7f0e')
                    corr = np.corrcoef(predictions_dict[name_j], predictions_dict[name_i])[0, 1]
                    ax.set_title(f'corr={corr:.3f}', fontsize=10)
                
                if i == len(model_list) - 1:
                    ax.set_xlabel(name_j, fontsize=9)
                if j == 0:
                    ax.set_ylabel(name_i, fontsize=9)
                
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / '11_predictions_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_latent_factors(self, viz_dir: Path) -> None:
        """Visualization 12: Latent Factor Heatmaps."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Latent Factor Analysis', fontsize=16, fontweight='bold')
        
        # Try to get SVD or NMF factors
        latent_factors = None
        model_name = None
        
        if 'SVDBasedCF' in self.models:
            try:
                latent_factors = self.models['SVDBasedCF'].model.user_factors[:50, :20]
                model_name = 'SVD User Factors'
            except:
                pass
        
        if latent_factors is None and 'NMFBasedCF' in self.models:
            try:
                latent_factors = self.models['NMFBasedCF'].model.user_factors[:50, :20]
                model_name = 'NMF User Factors'
            except:
                pass
        
        if latent_factors is not None:
            # User factors heatmap
            ax = axes[0]
            sns.heatmap(latent_factors, cmap='viridis', ax=ax, cbar_kws={'label': 'Factor Value'})
            ax.set_title(f'{model_name}\n(First 50 Users × 20 Factors)')
            ax.set_xlabel('Latent Factor')
            ax.set_ylabel('User ID')
            
            # Correlation of factors
            ax = axes[1]
            factor_corr = np.corrcoef(latent_factors.T)
            sns.heatmap(factor_corr, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Latent Factor Correlation')
            ax.set_xlabel('Latent Factor')
            ax.set_ylabel('Latent Factor')
        else:
            for ax in axes:
                ax.axis('off')
                ax.text(0.5, 0.5, 'Latent factors not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(viz_dir / '12_latent_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self) -> None:
        """Save all trained models."""
        logger.info("[Pipeline] Saving models")
        
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in tqdm(self.models.items(), desc="Saving models", colour='green'):
            model_path = models_dir / f"{model_name}.pkl"
            model.save_model(str(model_path))
            logger.info(f"[Pipeline] Saved {model_name} to {model_path}")
    
    def save_metrics(self) -> None:
        """Save evaluation metrics."""
        logger.info("[Pipeline] Saving metrics")
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_path = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"[Pipeline] Metrics saved to {metrics_path}")
    
    def zip_visualizations(self) -> None:
        """Zip all visualizations."""
        logger.info("[Pipeline] Zipping visualizations")
        
        viz_dir = self.output_dir / "visualizations"
        zip_path = self.output_dir / "visualizations.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in viz_dir.glob("*.png"):
                zipf.write(file, file.name)
        
        logger.info(f"[Pipeline] Visualizations zipped to {zip_path}")
    
    def generate_report(self) -> None:
        """Generate markdown report."""
        logger.info("[Pipeline] Generating report")
        
        report_path = self.output_dir / "REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Collaborative Filtering Pipeline Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Data Summary\n\n")
            if self.data is not None:
                f.write(f"- **Total Interactions**: {len(self.data):,}\n")
                f.write(f"- **Total Users**: {self.data['user_id'].nunique():,}\n")
                f.write(f"- **Total Items**: {self.data['item_id'].nunique():,}\n")
                f.write(f"- **Sparsity**: {1 - len(self.data) / (self.data['user_id'].max() + 1) / (self.data['item_id'].max() + 1):.6f}\n\n")
            
            f.write("## Models Trained\n\n")
            for model_name in self.models_trained:
                f.write(f"- {model_name}\n")
            f.write("\n")
            
            f.write("## Evaluation Metrics\n\n")
            if self.metrics:
                f.write("| Model | RMSE | MAE | MSE |\n")
                f.write("|-------|------|-----|-----|\n")
                for model_name, metrics in self.metrics.items():
                    f.write(f"| {model_name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['mse']:.4f} |\n")
                f.write("\n")
            
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            viz_list = [
                "01_data_distribution.png - Overall data distribution",
                "02_sparsity.png - User-item matrix sparsity analysis",
                "03_rating_distribution.png - Rating values distribution",
                "04_user_interactions.png - User interaction patterns",
                "05_item_interactions.png - Item interaction patterns",
                "06_model_metrics.png - Model performance comparison",
                "07_prediction_errors.png - Prediction error distributions",
                "08_user_engagement.png - User engagement analysis",
                "09_item_popularity.png - Item popularity trends",
                "10_train_test_split.png - Train-test split analysis",
                "11_predictions_correlation.png - Model predictions correlation",
                "12_latent_factors.png - Latent factor analysis"
            ]
            for viz in viz_list:
                f.write(f"- {viz}\n")
            f.write("\n")
            
            f.write("## Output Files\n\n")
            f.write("- `models/` - Trained model files\n")
            f.write("- `visualizations/` - All visualization PNG files\n")
            f.write("- `visualizations.zip` - Compressed visualization archive\n")
            f.write("- `metrics.csv` - Evaluation metrics\n")
        
        logger.info(f"[Pipeline] Report saved to {report_path}")
    
    def run_full_pipeline(
        self,
        data_source: str,
        data_path: str,
        min_user_interactions: int = 2,
        min_item_interactions: int = 2,
        test_size: float = 0.2,
        mlflow_ui: bool = True
    ) -> None:
        """
        Run complete pipeline.
        
        Args:
            data_source: 'duckdb' or 'csv'
            data_path: Path to data
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            test_size: Test set fraction
            mlflow_ui: Whether to launch MLflow UI
        """
        logger.info("[Pipeline] Starting full pipeline")
        
        try:
            # Load data
            if data_source == 'duckdb':
                self.connect_duckdb()
                self.load_data_from_duckdb(data_path)
            elif data_source == 'csv':
                self.load_data_from_csv(data_path)
            else:
                raise ValueError(f"Unknown data_source: {data_source}")
            
            # Preprocess
            self.preprocess_data(
                min_user_interactions=min_user_interactions,
                min_item_interactions=min_item_interactions
            )
            
            # Split
            self.split_data(test_size=test_size)
            
            # Train
            self.train_models()
            
            # Evaluate
            self.evaluate_models()
            
            # Visualize
            self.create_visualizations()
            
            # Save
            self.save_models()
            self.save_metrics()
            self.zip_visualizations()
            self.generate_report()
            
            logger.info("[Pipeline] Full pipeline completed successfully")
            
            if mlflow_ui:
                logger.info("[Pipeline] Launching MLflow UI...")
                cmd = f"mlflow ui --backend-store-uri sqlite:///{self.output_dir}/mlflow.db"
                os.system(shlex.quote(cmd))
        
        except Exception as e:
            logger.error(f"[Pipeline] Pipeline failed: {e}")
            raise
        
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    # Example usage
    pipeline = CollaborativeFilteringPipeline(
        output_dir="./cf_outputs",
        mlflow_experiment="collaborative_filtering_demo"
    )
    
    # Example with CSV data
    # pipeline.run_full_pipeline(
    #     data_source='csv',
    #     data_path='path/to/data.csv',
    #     min_user_interactions=2,
    #     min_item_interactions=2,
    #     test_size=0.2
    # )