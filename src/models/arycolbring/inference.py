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
inference.py
------------
Production inference pipeline for AryColBring collaborative filtering model.

Integrates:
- Model loading from saved embeddings
- Batch and real-time prediction APIs
- Inference metrics computation
- Production dashboard report generation
- Performance monitoring (latency, throughput)
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from ..inout.approximator import AryColBringPredictor
from .render_inference import generate_inference_report

logger = logging.getLogger(__name__)

__all__ = ["AryColBringInference", "InferenceService"]


class AryColBringInference:
    """
    Production-ready inference wrapper for AryColBring collaborative filtering model.
    
    Provides:
    - Fast batch predictions
    - Real-time single predictions
    - Top-N recommendations per user
    - Performance monitoring
    - Inference dashboard generation
    
    Parameters
    ----------
    model_path : str
        Path to the saved model (.npz file)
    num_threads : int
        Number of threads for parallel inference (default: 4)
    cache_enabled : bool
        Enable prediction caching (default: True)
    """
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        cache_enabled: bool = True,
    ) -> None:
        logger.info("Initializing AryColBringInference: model=%s threads=%d", 
                   model_path, num_threads)
        
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.cache_enabled = cache_enabled
        
        # Load model
        self.predictor = self._load_model()
        
        # Cache for predictions
        self.prediction_cache: Dict[str, float] = {} if cache_enabled else None
        
        # Inference statistics
        self.inference_stats = {
            "n_predictions": 0,
            "n_users_served": 0,
            "total_latency_ms": 0.0,
            "start_time": datetime.now().isoformat(),
        }
        
        self.config = self._get_config()
    
    def _load_model(self) -> AryColBringPredictor:
        """Load model from saved file."""
        logger.info("Loading model from: %s", self.model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        data = np.load(self.model_path, allow_pickle=True)
        
        config = json.loads(str(data["config"]))
        predictor = AryColBringPredictor(**config)
        
        predictor.item_embeddings = data["item_embeddings"]
        predictor.user_embeddings = data["user_embeddings"]
        predictor.item_biases = data["item_biases"]
        predictor.user_biases = data["user_biases"]
        
        logger.info("Model loaded successfully: users=%d items=%d",
                   predictor.user_embeddings.shape[0],
                   predictor.item_embeddings.shape[0])
        
        return predictor
    
    def _get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        try:
            data = np.load(self.model_path, allow_pickle=True)
            return json.loads(str(data["config"]))
        except Exception as e:
            logger.warning("Could not load config: %s", e)
            return {}
    
    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[int, List[int], np.ndarray],
        return_cache: bool = True,
    ) -> np.ndarray:
        """
        Predict scores for user-item pairs.
        
        Parameters
        ----------
        user_ids : int | list | ndarray
            User ID(s)
        item_ids : int | list | ndarray
            Item ID(s)
        return_cache : bool
            Use cached predictions if available (default: True)
        
        Returns
        -------
        ndarray
            Prediction scores (float32)
        """
        start_time = time.perf_counter()
        
        # Convert to arrays
        if isinstance(user_ids, int):
            user_ids = [user_ids]
        if isinstance(item_ids, int):
            item_ids = [item_ids]
        
        user_ids = np.array(user_ids, dtype=np.int32)
        item_ids = np.array(item_ids, dtype=np.int32)
        
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have same length")
        
        # Check cache
        if self.cache_enabled and return_cache:
            cached_results = []
            uncached_indices = []
            
            for i, (uid, iid) in enumerate(zip(user_ids, item_ids)):
                cache_key = f"{uid}_{iid}"
                if cache_key in self.prediction_cache:
                    cached_results.append((i, self.prediction_cache[cache_key]))
                else:
                    uncached_indices.append(i)
            
            if len(cached_results) == len(user_ids):
                # All cached
                results = np.zeros(len(user_ids), dtype=np.float32)
                for idx, score in cached_results:
                    results[idx] = score
                return results
            
            # Partial cache - need to compute uncached
            if uncached_indices:
                uncached_users = user_ids[uncached_indices]
                uncached_items = item_ids[uncached_indices]
                
                uncached_scores = self.predictor.predict(
                    user_ids=uncached_users,
                    item_ids=uncached_items,
                    num_threads=self.num_threads,
                )
                
                # Update cache
                for idx, score in zip(uncached_indices, uncached_scores):
                    cache_key = f"{user_ids[idx]}_{item_ids[idx]}"
                    self.prediction_cache[cache_key] = float(score)
                
                # Merge results
                results = np.zeros(len(user_ids), dtype=np.float32)
                for idx, score in cached_results:
                    results[idx] = score
                for idx, score in zip(uncached_indices, uncached_scores):
                    results[idx] = score
                
                predictions = results
            else:
                predictions = np.array([score for _, score in cached_results], dtype=np.float32)
        else:
            # No cache - compute all
            predictions = self.predictor.predict(
                user_ids=user_ids,
                item_ids=item_ids,
                num_threads=self.num_threads,
            )
        
        # Update stats
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.inference_stats["n_predictions"] += len(user_ids)
        self.inference_stats["total_latency_ms"] += latency_ms
        
        logger.debug("Predictions: n=%d latency=%.2fms", len(user_ids), latency_ms)
        
        return predictions
    
    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_items: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-N item recommendations for a user.
        
        Parameters
        ----------
        user_id : int
            User ID
        n_items : int
            Number of recommendations (default: 10)
        exclude_items : list of int, optional
            Items to exclude (e.g., already purchased)
        
        Returns
        -------
        list of tuples
            [(item_id, score), ...] sorted by score descending
        """
        logger.debug("Generating recommendations: user=%d n=%d", user_id, n_items)
        
        start_time = time.perf_counter()
        
        # Get number of items
        n_total_items = self.predictor.item_embeddings.shape[0]
        
        # Generate candidate items
        if exclude_items:
            candidates = [i for i in range(n_total_items) if i not in exclude_items]
        else:
            candidates = list(range(n_total_items))
        
        # Score all candidates
        user_array = np.full(len(candidates), user_id, dtype=np.int32)
        item_array = np.array(candidates, dtype=np.int32)
        
        scores = self.predictor.predict(
            user_ids=user_array,
            item_ids=item_array,
            num_threads=self.num_threads,
        )
        
        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n_items]
        
        recommendations = [
            (int(candidates[idx]), float(scores[idx]))
            for idx in top_indices
        ]
        
        # Update stats
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.inference_stats["n_predictions"] += len(candidates)
        self.inference_stats["n_users_served"] += 1
        self.inference_stats["total_latency_ms"] += latency_ms
        
        logger.info(
            "Recommendations generated: user=%d n=%d latency=%.2fms",
            user_id, n_items, latency_ms
        )
        
        return recommendations
    
    def batch_recommend(
        self,
        user_ids: List[int],
        n_items: int = 10,
        exclude_dict: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users in batch.
        
        Parameters
        ----------
        user_ids : list of int
            List of user IDs
        n_items : int
            Number of recommendations per user
        exclude_dict : dict, optional
            Mapping of user_id -> items to exclude
        
        Returns
        -------
        dict
            {user_id: [(item_id, score), ...], ...}
        """
        logger.info("Batch recommendations: users=%d n=%d", len(user_ids), n_items)
        
        start_time = time.perf_counter()
        
        results = {}
        for user_id in user_ids:
            exclude_items = exclude_dict.get(user_id) if exclude_dict else None
            results[user_id] = self.recommend(
                user_id=user_id,
                n_items=n_items,
                exclude_items=exclude_items,
            )
        
        total_latency = (time.perf_counter() - start_time) * 1000
        avg_latency = total_latency / len(user_ids) if user_ids else 0
        
        logger.info(
            "Batch complete: users=%d total=%.2fms avg=%.2fms",
            len(user_ids), total_latency, avg_latency
        )
        
        return results
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current inference performance metrics.
        
        Returns
        -------
        dict
            Performance metrics including latency, throughput, etc.
        """
        n_preds = self.inference_stats["n_predictions"]
        total_latency = self.inference_stats["total_latency_ms"]
        
        avg_latency = total_latency / n_preds if n_preds > 0 else 0
        throughput = n_preds / (total_latency / 1000) if total_latency > 0 else 0
        
        elapsed_time = (datetime.now() - datetime.fromisoformat(
            self.inference_stats["start_time"]
        )).total_seconds()
        
        qps = n_preds / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "n_predictions": n_preds,
            "n_users_served": self.inference_stats["n_users_served"],
            "avg_latency_ms": avg_latency,
            "throughput_preds_per_sec": throughput,
            "qps": qps,
            "elapsed_time_sec": elapsed_time,
        }
    
    def generate_inference_report(
        self,
        output_dir: Optional[str] = None,
        experiment_name: str = "Inference Run",
        metrics: Optional[Dict[str, float]] = None,
        predictions_sample: Optional[List[Dict[str, Any]]] = None,
        charts: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """
        Generate an inference performance dashboard report.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory for the report
        experiment_name : str
            Name of the experiment/run
        metrics : dict, optional
            Custom metrics to display
        predictions_sample : list, optional
            Sample predictions for the report
        charts : list, optional
            Custom charts to include
        
        Returns
        -------
        Path
            Path to the generated HTML report
        """
        logger.info("Generating inference report...")
        
        # Get default metrics
        default_metrics = self.get_metrics()
        
        # Merge with custom metrics
        if metrics:
            default_metrics.update(metrics)
        
        # Build context
        context_data = {
            "metrics": default_metrics,
            "inference_statistics": {
                "n_predictions": default_metrics["n_predictions"],
                "n_users_served": default_metrics["n_users_served"],
                "avg_latency_ms": default_metrics["avg_latency_ms"],
                "coverage": 0.75,  # Placeholder - calculate based on your data
            },
            "experiment_name": experiment_name,
            "model_version": "1.0.0",
            "batch_size": 100,
            "num_threads": self.num_threads,
            "predictions": predictions_sample or [],
            "charts": charts or [],
        }
        
        # Generate report
        report_path = generate_inference_report(
            context_data=context_data,
            output_dir=output_dir,
        )
        
        logger.info("Inference report generated: %s", report_path)
        return report_path
    
    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        if self.prediction_cache is not None:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")


class InferenceService:
    """
    High-level inference service for production deployment.
    
    Wraps AryColBringInference with additional features:
    - API endpoint simulation
    - Request batching
    - Health checks
    - Metrics export
    """
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        max_batch_size: int = 1000,
    ) -> None:
        logger.info("Initializing InferenceService: model=%s", model_path)
        
        self.inference = AryColBringInference(
            model_path=model_path,
            num_threads=num_threads,
            cache_enabled=True,
        )
        
        self.max_batch_size = max_batch_size
        self.request_count = 0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns
        -------
        dict
            Health status information
        """
        metrics = self.inference.get_metrics()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_path": str(self.inference.model_path),
            "num_threads": self.inference.num_threads,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
    
    def predict_api(
        self,
        user_id: int,
        item_ids: List[int],
    ) -> Dict[str, Any]:
        """
        Simulate API endpoint for prediction.
        
        Parameters
        ----------
        user_id : int
            User ID
        item_ids : list of int
            List of item IDs to score
        
        Returns
        -------
        dict
            Response with predictions and metadata
        """
        self.request_count += 1
        
        start_time = time.perf_counter()
        
        scores = self.inference.predict(
            user_ids=user_id,
            item_ids=item_ids,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "request_id": self.request_count,
            "user_id": user_id,
            "predictions": [
                {"item_id": iid, "score": float(score)}
                for iid, score in zip(item_ids, scores)
            ],
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }
    
    def recommend_api(
        self,
        user_id: int,
        n_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Simulate API endpoint for recommendations.
        
        Parameters
        ----------
        user_id : int
            User ID
        n_items : int
            Number of recommendations
        
        Returns
        -------
        dict
            Response with recommendations and metadata
        """
        self.request_count += 1
        
        start_time = time.perf_counter()
        
        recommendations = self.inference.recommend(
            user_id=user_id,
            n_items=n_items,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "request_id": self.request_count,
            "user_id": user_id,
            "recommendations": [
                {"item_id": iid, "score": float(score)}
                for iid, score in recommendations
            ],
            "n_items": n_items,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.
        
        Returns
        -------
        dict
            Service status and metrics
        """
        health = self.health_check()
        metrics = self.inference.get_metrics()
        
        return {
            "service": "AryColBring Inference Service",
            "version": "1.0.0",
            "requests_handled": self.request_count,
            "health": health,
            "performance_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Example usage
    print("AryColBring Inference Service")
    print("=" * 50)
    print("Use AryColBringInference or InferenceService class")
    print("for production deployment with dashboard reports.")
    print("\nExample:")
    print("  service = InferenceService(model_path='model.npz')")
    print("  recs = service.recommend_api(user_id=123, n_items=10)")
    print("  report = service.inference.generate_inference_report()")
