'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Explainable AI Module using SHAP
Feature importance, SHAP values, and model interpretability
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    SHAP-based model explanation and interpretability
    """
    
    def __init__(self, background_data: Optional[np.ndarray] = None,
                 max_samples: int = 100):
        """
        Initialize explainability engine
        
        Parameters
        ----------
        background_data : ndarray, optional
            Background data for SHAP
        max_samples : int
            Maximum samples for SHAP computation
        """
        self.background_data = background_data
        self.max_samples = max_samples
        self.explainer = None
        self.shap_values = None
        logger.info("ExplainabilityEngine initialized")
    
    def compute_shap(self, model, X: np.ndarray, 
                    use_background: bool = True) -> np.ndarray:
        """
        Compute SHAP values
        
        Parameters
        ----------
        model : object
            Trained model
        X : ndarray
            Feature matrix
        use_background : bool
            Whether to use background data
            
        Returns
        -------
        ndarray
            SHAP values
        """
        logger.info(f"Computing SHAP values for {X.shape[0]} samples")
        
        try:
            # Use a subset for faster computation
            X_sample = X[:min(self.max_samples, X.shape[0])]
            
            # Create explainer
            if hasattr(model, 'predict'):
                self.explainer = shap.TreeExplainer(model)
            else:
                logger.error("Model type not supported for SHAP")
                return None
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            logger.info("SHAP values computed successfully")
            return self.shap_values
        
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def plot_shap_summary(self, feature_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
        """
        Plot SHAP summary plot
        
        Parameters
        ----------
        feature_names : list, optional
            Feature names
        save_path : str, optional
            Path to save plot
        """
        if self.shap_values is None:
            logger.error("SHAP values not computed")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(self.shap_values, plot_type="bar",
                            feature_names=feature_names, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            plt.show()
        
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {e}")
    
    def plot_shap_dependence(self, feature_idx: int,
                            feature_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """
        Plot SHAP dependence for a feature
        
        Parameters
        ----------
        feature_idx : int
            Feature index
        feature_names : list, optional
            Feature names
        save_path : str, optional
            Path to save plot
        """
        if self.shap_values is None:
            logger.error("SHAP values not computed")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            feature_name = feature_names[feature_idx] if feature_names else f'Feature_{feature_idx}'
            shap.dependence_plot(feature_idx, self.shap_values, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP dependence plot saved to {save_path}")
            plt.show()
        
        except Exception as e:
            logger.error(f"Error plotting SHAP dependence: {e}")
    
    def interpret_predictions(self, shap_values: np.ndarray,
                             X: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             top_k: int = 5) -> Dict:
        """
        Generate interpretability report
        
        Parameters
        ----------
        shap_values : ndarray
            SHAP values
        X : ndarray
            Feature matrix
        feature_names : list, optional
            Feature names
        top_k : int
            Top K features to report
            
        Returns
        -------
        dict
            Interpretation results
        """
        logger.info("Generating interpretation report")
        
        interpretation = {
            'global_feature_importance': {},
            'sample_explanations': [],
            'feature_interactions': {}
        }
        
        # Global feature importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(-mean_abs_shap)[:top_k]
        
        for idx in top_features_idx:
            feature_name = feature_names[idx] if feature_names else f'Feature_{idx}'
            interpretation['global_feature_importance'][feature_name] = float(mean_abs_shap[idx])
        
        # Sample-level explanations (first 5 samples)
        for i in range(min(5, shap_values.shape[0])):
            sample_explanation = {
                'sample_id': i,
                'top_features': {}
            }
            
            top_features_sample = np.argsort(-np.abs(shap_values[i, :]))[:top_k]
            for rank, idx in enumerate(top_features_sample):
                feature_name = feature_names[idx] if feature_names else f'Feature_{idx}'
                sample_explanation['top_features'][feature_name] = {
                    'shap_value': float(shap_values[i, idx]),
                    'feature_value': float(X[i, idx]) if X is not None else None,
                    'rank': rank + 1
                }
            
            interpretation['sample_explanations'].append(sample_explanation)
        
        return interpretation
    
    def get_feature_interactions(self, X: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               top_k: int = 10) -> Dict:
        """
        Analyze feature interactions
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        feature_names : list, optional
            Feature names
        top_k : int
            Top K interactions
            
        Returns
        -------
        dict
            Feature interactions
        """
        if self.shap_values is None:
            logger.error("SHAP values not computed")
            return {}
        
        logger.info("Analyzing feature interactions")
        
        interactions = {}
        n_features = X.shape[1]
        
        # Compute interaction strength
        interaction_scores = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Simple interaction metric: correlation of SHAP values
                corr = np.corrcoef(self.shap_values[:, i], self.shap_values[:, j])[0, 1]
                if not np.isnan(corr):
                    interaction_scores.append({
                        'feature1_idx': i,
                        'feature2_idx': j,
                        'interaction_strength': abs(corr)
                    })
        
        # Sort and get top K
        interaction_scores = sorted(interaction_scores, 
                                   key=lambda x: x['interaction_strength'], 
                                   reverse=True)[:top_k]
        
        for inter in interaction_scores:
            i, j = inter['feature1_idx'], inter['feature2_idx']
            f1_name = feature_names[i] if feature_names else f'Feature_{i}'
            f2_name = feature_names[j] if feature_names else f'Feature_{j}'
            
            interactions[f"{f1_name} x {f2_name}"] = inter['interaction_strength']
        
        return interactions
    
    def generate_html_report(self, X: np.ndarray, y_pred: np.ndarray,
                            feature_names: Optional[List[str]] = None,
                            output_path: str = 'explainability_report.html'):
        """
        Generate HTML explainability report
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        y_pred : ndarray
            Model predictions
        feature_names : list, optional
            Feature names
        output_path : str
            Path to save report
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        interpretation = self.interpret_predictions(self.shap_values, X, feature_names)
        interactions = self.get_feature_interactions(X, feature_names)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XGBoost LTR Model Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #007bff; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .important {{ color: #d9534f; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>XGBoost LTR Model Explainability Report</h1>
            
            <h2>Global Feature Importance</h2>
            <table>
                <tr><th>Feature</th><th>SHAP Mean |value|</th></tr>
        """
        
        for feature, importance in interpretation['global_feature_importance'].items():
            html_content += f"<tr><td>{feature}</td><td>{importance:.6f}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Feature Interactions</h2>
            <table>
                <tr><th>Interaction</th><th>Strength</th></tr>
        """
        
        for interaction, strength in interactions.items():
            html_content += f"<tr><td>{interaction}</td><td>{strength:.6f}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Sample Explanations</h2>
        """
        
        for sample_exp in interpretation['sample_explanations']:
            html_content += f"<h3>Sample {sample_exp['sample_id']}</h3><table>"
            html_content += "<tr><th>Feature</th><th>SHAP Value</th><th>Feature Value</th></tr>"
            
            for feature, details in sample_exp['top_features'].items():
                html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{details['shap_value']:.6f}</td>
                    <td>{details['feature_value']:.6f if details['feature_value'] else 'N/A'}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")