#!/usr/bin/env python3

"""
test_arycolbring_inference.py
_______________________________________________________________________________
Inference test script for AryColBring collaborative filtering model.

This script provides a comprehensive inference pipeline for the AryColBring model
with integrated logging, performance monitoring, and reporting.

Features:
- Load pre-trained AryColBring models
- Real-time single and batch predictions
- Top-N recommendation generation
- Performance metrics tracking (latency, throughput, QPS)
- Caching support for repeated predictions
- Comprehensive inference reports with dashboards

Author: Aryanto
Created: 2026-06-05
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from tqdm.auto import tqdm
import time

# Add source directory to path
SrcDir = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SrcDir))

from configs import logger, _cfg
from models.arycolbring.inference import AryColBringInference


class AryColBringInferencePipeline:
    """
    High-level inference pipeline wrapper for AryColBring model.
    Handles model loading, predictions, and performance monitoring.
    """

    def __init__(self,
                 model_path: Union[str, Path],
                 output_dir: Union[str, Path] = "artifacts/inference",
                 num_threads: int = 4,
                 cache_enabled: bool = True):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model (.npz file)
            output_dir: Directory for output reports
            num_threads: Number of threads for parallel inference
            cache_enabled: Enable prediction caching
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.num_threads = num_threads
        self.cache_enabled = cache_enabled

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.inference_service = self._load_model()
        logger.info("Inference pipeline initialized: model=%s, threads=%d, cache=%s",
                    self.model_path, num_threads, cache_enabled)

    def _load_model(self) -> AryColBringInference:
        """Load pre-trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info("Loading model from: %s", self.model_path)
        service = AryColBringInference(
            model_path=str(self.model_path),
            num_threads=self.num_threads,
            cache_enabled=self.cache_enabled
        )
        logger.info("Model loaded successfully")
        return service

    def predict_batch(self,
                      user_ids: List[int],
                      item_ids: List[int],
                      return_cache: bool = True) -> np.ndarray:
        """
        Predict scores for multiple user-item pairs.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            return_cache: Use cache if available

        Returns:
            Array of prediction scores
        """
        logger.info("Batch prediction: %d user-item pairs", len(user_ids))
        predictions = self.inference_service.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            return_cache=return_cache
        )
        logger.debug("Predictions completed: mean=%.4f, std=%.4f",
                     predictions.mean(), predictions.std())
        return predictions

    def recommend_batch(self,
                        user_ids: List[int],
                        n_items: int = 10,
                        exclude_dict: Optional[Dict[int, List[int]]] = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate top-N recommendations for multiple users.

        Args:
            user_ids: List of user IDs
            n_items: Number of recommendations per user
            exclude_dict: Items to exclude per user (optional)

        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples
        """
        logger.info("Batch recommendations: %d users, top-%d items", len(user_ids), n_items)

        recommendations = self.inference_service.batch_recommend(
            user_ids=user_ids,
            n_items=n_items,
            exclude_dict=exclude_dict
        )

        logger.info("Recommendations generated for %d users", len(recommendations))
        return recommendations

    def single_recommend(self,
                         user_id: int,
                         n_items: int = 10,
                         exclude_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a single user.

        Args:
            user_id: User ID
            n_items: Number of recommendations
            exclude_items: Items to exclude (optional)

        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        logger.debug("Single recommendation: user=%d, n=%d", user_id, n_items)

        recommendations = self.inference_service.recommend(
            user_id=user_id,
            n_items=n_items,
            exclude_items=exclude_items
        )

        return recommendations

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current inference performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = self.inference_service.get_metrics()
        logger.debug("Performance metrics: QPS=%.2f, avg_latency=%.2fms",
                     metrics.get("qps", 0), metrics.get("avg_latency_ms", 0))
        return metrics

    def generate_report(self,
                        experiment_name: str = "Inference Run",
                        predictions_sample: Optional[List[Dict]] = None) -> Path:
        """
        Generate inference performance report.

        Args:
            experiment_name: Name for the experiment
            predictions_sample: Sample predictions to include in report

        Returns:
            Path to generated HTML report
        """
        logger.info("Generating inference report: %s", experiment_name)

        metrics = self.get_performance_metrics()
        report_path = self.inference_service.generate_inference_report(
            output_dir=str(self.output_dir),
            experiment_name=experiment_name,
            metrics=metrics,
            predictions_sample=predictions_sample
        )

        logger.info("Report generated: %s", report_path)
        return report_path

    def run_performance_benchmark(self,
                                  n_predictions: int = 1000,
                                  batch_size: int = 100,
                                  n_users: Optional[int] = None,
                                  n_items: Optional[int] = None) -> Dict:
        """
        Run performance benchmark.

        Args:
            n_predictions: Total number of predictions to make
            batch_size: Batch size for predictions
            n_users: Number of unique users (random if not specified)
            n_items: Number of unique items (random if not specified)

        Returns:
            Dictionary with benchmark results
        """
        # Get model dimensions
        if n_users is None:
            n_users = self.inference_service.predictor.user_embeddings.shape[0]
        if n_items is None:
            n_items = self.inference_service.predictor.item_embeddings.shape[0]

        logger.info("Starting benchmark: %d predictions, batch_size=%d",
                    n_predictions, batch_size)

        # Clear cache
        if self.cache_enabled:
            self.inference_service.clear_cache()

        # Generate random user-item pairs
        user_ids = np.random.randint(0, n_users, n_predictions)
        item_ids = np.random.randint(0, n_items, n_predictions)

        # Benchmark
        start_time = time.perf_counter()
        predictions = self.predict_batch(user_ids, item_ids, return_cache=False)
        total_time = (time.perf_counter() - start_time) * 1000

        metrics = self.get_performance_metrics()

        results = {
            "n_predictions": n_predictions,
            "batch_size": batch_size,
            "total_time_ms": total_time,
            "predictions_per_sec": (n_predictions / total_time) * 1000,
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
            "qps": metrics.get("qps", 0),
            "n_users": n_users,
            "n_items": n_items
        }

        logger.info("Benchmark complete: %.2f predictions/sec, %.2fms avg latency",
                    results["predictions_per_sec"], results["avg_latency_ms"])

        return results

    def run_full_pipeline(self,
                          n_test_users: int = 100,
                          n_recommendations: int = 10,
                          run_benchmark: bool = True,
                          benchmark_predictions: int = 1000) -> dict:
        """
        Run complete inference pipeline.

        Args:
            n_test_users: Number of users to test
            n_recommendations: Number of recommendations per user
            run_benchmark: Whether to run performance benchmark
            benchmark_predictions: Number of predictions for benchmark

        Returns:
            Dictionary with results
        """
        logger.info("Starting full inference pipeline")

        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path)
        }

        # Get model dimensions
        n_total_users = self.inference_service.predictor.user_embeddings.shape[0]
        n_total_items = self.inference_service.predictor.item_embeddings.shape[0]

        logger.info("Model dimensions: users=%d, items=%d", n_total_users, n_total_items)

        # Test batch recommendations
        test_users = np.random.choice(n_total_users, min(n_test_users, n_total_users), replace=False)
        recommendations = self.recommend_batch(
            user_ids=test_users.tolist(),
            n_items=n_recommendations
        )
        results["batch_recommendations_completed"] = len(recommendations)

        # Generate sample predictions for report
        sample_predictions = []
        for user_id in list(test_users)[:5]:  # First 5 users
            recs = recommendations.get(user_id, [])
            sample_predictions.append({
                "user_id": int(user_id),
                "recommendations": [(int(item_id), float(score)) for item_id, score in recs]
            })

        # Generate report
        report_path = self.generate_report(
            experiment_name="Full Inference Pipeline",
            predictions_sample=sample_predictions
        )
        results["report_path"] = str(report_path)

        # Run benchmark
        if run_benchmark:
            benchmark_results = self.run_performance_benchmark(
                n_predictions=benchmark_predictions
            )
            results["benchmark"] = benchmark_results

        # Get final metrics
        metrics = self.get_performance_metrics()
        results["metrics"] = metrics

        logger.info("Pipeline completed successfully")
        return results


def main():
    """Main execution function for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run AryColBring inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="artifacts/inference", help="Output directory")
    parser.add_argument("--n-users", type=int, default=100, help="Number of test users")
    parser.add_argument("--n-recs", type=int, default=10, help="Number of recommendations per user")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-size", type=int, default=1000, help="Benchmark prediction count")

    args = parser.parse_args()

    # Run pipeline
    pipeline = AryColBringInferencePipeline(
        model_path=args.model,
        output_dir=args.output_dir,
        num_threads=args.threads,
        cache_enabled=not args.no_cache
    )

    results = pipeline.run_full_pipeline(
        n_test_users=args.n_users,
        n_recommendations=args.n_recs,
        run_benchmark=args.benchmark,
        benchmark_predictions=args.benchmark_size
    )

    # Print summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Model: {results['model_path']}")
    print(f"Batch recommendations: {results.get('batch_recommendations_completed', 0)}")
    print(f"Report: {results.get('report_path', 'N/A')}")

    if "metrics" in results:
        print("\nPerformance Metrics:")
        metrics = results["metrics"]
        print(f"  QPS: {metrics.get('qps', 0):.2f}")
        print(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.2f} ms")
        print(f"  Predictions: {metrics.get('n_predictions', 0)}")
        print(f"  Users Served: {metrics.get('n_users_served', 0)}")
        print(f"  Throughput: {metrics.get('throughput_preds_per_sec', 0):.2f} pred/sec")

    if "benchmark" in results:
        print("\nBenchmark Results:")
        bench = results["benchmark"]
        print(f"  Predictions: {bench['n_predictions']}")
        print(f"  Time: {bench['total_time_ms']:.2f} ms")
        print(f"  Throughput: {bench['predictions_per_sec']:.2f} pred/sec")
        print(f"  Avg Latency: {bench['avg_latency_ms']:.2f} ms")
        print(f"  QPS: {bench['qps']:.2f}")

    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Inference pipeline failed with error: %s", str(e))
        sys.exit(1)
