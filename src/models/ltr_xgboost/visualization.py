'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Comprehensive Visualization Module for XGBoost LTR Ranker
15+ production-grade plots for data and model analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class RankingVisualizer:
    """
    Advanced visualization for ranking and recommendation systems
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        """Initialize visualizer"""
        self.figsize = figsize
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_prediction_distribution(self, y_pred: np.ndarray, 
                                    y_true: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None):
        """Plot 1: Prediction score distribution"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(y_pred, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Predictions')
        if y_true is not None:
            ax.hist(y_true, bins=50, alpha=0.7, color='red', edgecolor='black', label='Actual')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray,
                                save_path: Optional[str] = None):
        """Plot 2: Actual vs Predicted scatter plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                               top_n: int = 15, save_path: Optional[str] = None):
        """Plot 3: Feature importance bar chart"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_imp)
        
        ax.barh(features, scores, color='steelblue', edgecolor='black')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot 4: Residual analysis"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Residuals histogram
        axes[0, 0].hist(residuals, bins=50, edgecolor='black', color='skyblue')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Residuals Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot (normal probability plot)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF plot simulation
        axes[1, 1].scatter(range(len(residuals[:100])), residuals[:100], alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_ndcg_by_rank(self, rankings: Dict[int, List[Tuple[int, float]]],
                         y_true: Dict[int, List[int]], 
                         save_path: Optional[str] = None):
        """Plot 5: NDCG@K curves"""
        ndcg_scores = []
        k_values = []
        
        for k in range(1, 11):
            total_ndcg = 0
            count = 0
            
            for group_id, ranks in rankings.items():
                y_pred = np.array([score for _, score in ranks[:k]])
                y_true_group = np.array(y_true.get(group_id, [0] * k)[:k])
                
                if len(y_pred) > 0 and len(y_true_group) > 0:
                    # Simplified NDCG calculation
                    dcg = sum((2 ** y_true_group[i] - 1) / np.log2(i + 2) for i in range(len(y_pred)))
                    total_ndcg += dcg
                    count += 1
            
            if count > 0:
                ndcg_scores.append(total_ndcg / count)
                k_values.append(k)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(k_values, ndcg_scores, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.fill_between(k_values, ndcg_scores, alpha=0.3)
        ax.set_xlabel('K')
        ax.set_ylabel('NDCG Score')
        ax.set_title('NDCG@K Performance Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_ranking_distribution(self, rankings: Dict[int, List[Tuple[int, float]]],
                                 save_path: Optional[str] = None):
        """Plot 6: Distribution of ranking scores"""
        all_scores = []
        for group_ranks in rankings.values():
            all_scores.extend([score for _, score in group_ranks])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(all_scores, bins=50, edgecolor='black', color='lightcoral', alpha=0.7)
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.3f}')
        ax.axvline(np.median(all_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_scores):.3f}')
        
        ax.set_xlabel('Ranking Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Ranking Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_cumulative_gain(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot 7: Cumulative gain chart"""
        sorted_indices = np.argsort(-y_pred)
        sorted_true = y_true[sorted_indices]
        
        cumulative_gains = np.cumsum(sorted_true)
        perfect_cumulative_gains = np.cumsum(np.sort(-y_true))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(cumulative_gains, label='Model', linewidth=2, marker='o', markersize=4)
        ax.plot(perfect_cumulative_gains, label='Perfect', linewidth=2, linestyle='--', marker='s', markersize=4)
        ax.fill_between(range(len(cumulative_gains)), cumulative_gains, alpha=0.3)
        
        ax.set_xlabel('Number of Items')
        ax.set_ylabel('Cumulative Gains')
        ax.set_title('Cumulative Gain Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot 8: ROC Curve (binary relevance)"""
        from sklearn.metrics import roc_curve, auc
        
        # Binarize predictions
        y_true_binary = (y_true > 0).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.fill_between(fpr, tpr, alpha=0.2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: Optional[str] = None):
        """Plot 9: Precision-Recall Curve"""
        from sklearn.metrics import precision_recall_curve, auc
        
        y_true_binary = (y_true > 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='green')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_correlation_heatmap(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """Plot 10: Feature correlation heatmap"""
        df_X = pd.DataFrame(X, columns=feature_names or [f'Feature_{i}' for i in range(X.shape[1])])
        corr_matrix = df_X.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                              n_bins: int = 10, save_path: Optional[str] = None):
        """Plot 11: Calibration curve"""
        from sklearn.calibration import calibration_curve
        
        y_true_binary = (y_true > 0).astype(int)
        prob_true, prob_pred = calibration_curve(y_true_binary, y_pred, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfectly calibrated')
        ax.fill_between(prob_pred, prob_true, alpha=0.3)
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_prediction_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          save_path: Optional[str] = None):
        """Plot 12: Prediction error distribution"""
        errors = np.abs(y_true - y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        axes[0].hist(errors, bins=50, edgecolor='black', color='purple', alpha=0.7)
        axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by percentiles
        percentiles = [np.percentile(errors, i) for i in range(0, 101, 10)]
        axes[1].bar(range(len(percentiles)), percentiles, edgecolor='black', color='lightblue')
        axes[1].set_xlabel('Percentile')
        axes[1].set_ylabel('Error Value')
        axes[1].set_title('Error by Percentile')
        axes[1].set_xticks(range(len(percentiles)))
        axes[1].set_xticklabels([f'{i}%' for i in range(0, 101, 10)])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                            save_path: Optional[str] = None):
        """Plot 13: Learning curves"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_scores) + 1)
        ax.plot(epochs, train_scores, marker='o', linewidth=2, label='Training', markersize=6)
        ax.plot(epochs, val_scores, marker='s', linewidth=2, label='Validation', markersize=6)
        ax.fill_between(epochs, train_scores, alpha=0.2)
        ax.fill_between(epochs, val_scores, alpha=0.2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             threshold: float = 0.5, save_path: Optional[str] = None):
        """Plot 14: Confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.show()
    
    def plot_interactive_scatter(self, X: np.ndarray, y_pred: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """Plot 15: Interactive 3D scatter (plotly)"""
        if X.shape[1] < 3:
            logger.warning("Need at least 3 features for 3D scatter plot")
            return
        
        df = pd.DataFrame(X[:, :3], columns=feature_names[:3] if feature_names else ['F1', 'F2', 'F3'])
        df['Prediction'] = y_pred
        
        fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2],
                           color='Prediction', size='Prediction',
                           hover_data=df.columns,
                           title='Interactive 3D Prediction Visualization')
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved: {save_path}")
        
        fig.show()