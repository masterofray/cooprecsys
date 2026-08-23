#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-04"


"""
t02_reasoner.py
___________________________________________________________________
Inference test script for AryColBring collaborative filtering model.
This script provides a comprehensive inference pipeline for the AryColBring
model with integrated logging, performance monitoring, and reporting.
"""

import sys
import time
import numpy  as np
import pandas as pd
from pathlib  import Path
from datetime import datetime
from typing   import Optional, Tuple, Union, List, Dict
from argparse import ArgumentParser

#LocDir = Path(__file__).resolve().parents[2] / 'src'
#sys.path.append(str(LocDir))
from src.cooprecsys.configs            import _cfg, logger
from src.cooprecsys.models.arycolbring import AryColBringInference


class AryColBring_Inference_Test:
    """
    High-level inference pipeline wrapper for AryColBring model.
    Handles model loading, predictions, and performance monitoring.
    """
    def __init__(self,
                 model_path    : Union[str, Path] = None,
                 output_dir    : Union[str, Path] = None,
                 num_threads   : Optional[int]    = None,
                 cache_enabled : bool             = True,
                ):
        if output_dir is None:
            self.output_dir = Path(_cfg.get('PATHS', 'output_dir',
                              fallback = 'artifacts')) / 'inference'
        else:
            self.output_dir = Path(output_dir)
        self.num_threads   = num_threads or _cfg.getint(
                             'model', 'num_threads', fallback = 4)
        self.cache_enabled = cache_enabled
        self._modelpath    = str()
        self._report_path  = str()

        self.output_dir.mkdir(parents = True, exist_ok = True)
        self.model_path = model_path

        self.inference_service = self._load_model()
        logger.info("Inference pipeline initialized: model = %s, "
                    "threads = %d, cache = %s",
                     self.model_path, self.num_threads, self.cache_enabled)

    @property
    def model_path(self) -> Path:
        return self._model_path

    @model_path.setter
    def model_path(self, value: Union[str, Path]) -> None:
        if not isinstance(value, (str, Path)):
            msg = f"Invalid model_path type: {type(value).__name__}." \
                   "Expected str or Path."
            logger.error(msg)
            raise TypeError(msg)
        path_obj = Path(value)
        if not path_obj.exists():
            msg = f"Validation failed: Model file does not exist at '{path_obj}'."
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not path_obj.is_file():
            msg = f"Validation failed: Target path is a "\
                  f"directory, not a file -> '{path_obj}'."
            logger.error(msg)
            raise ValueError(msg)
        if path_obj.suffix != '.npz':
            msg = f"Validation failed: Model file must be a '.npz' "\
                  f"archive, got '{path_obj.suffix}'."
            logger.error(msg)
            raise ValueError(msg)
        if path_obj.stat().st_size == 0:
            msg = f"Validation failed: The model file '{path_obj.name}' is empty."
            logger.error(msg)
            raise ValueError(msg)
        self._model_path = path_obj
        logger.debug("Successfully assigned model_path: '%s'.", path_obj)

    def _load_model(self) -> AryColBringInference:
        """Load pre-trained model."""
        logger.info("Loading model from: %s", self.model_path)
        service = AryColBringInference(
                  model_path    = str(self.model_path),
                  num_threads   = self.num_threads,
                  cache_enabled = self.cache_enabled)
        self._modelpath = str(self.model_path)
        logger.info("Model loaded successfully")
        return service

    def predict_batch(self,
                      user_ids     : List[int],
                      item_ids     : List[int],
                      return_cache : bool = True,
                     ) -> np.ndarray:
        """
        Predict scores for multiple user-item pairs.
        Its return be ndarray as Array of prediction scores.
        """
        logger.info("Batch prediction: %d user-item pairs", len(user_ids))
        predictions = self.inference_service.predict(
                      user_ids     = user_ids,
                      item_ids     = item_ids,
                      return_cache = return_cache)
        logger.debug("Predictions completed: mean = %.4f, std = %.4f",
                      predictions.mean(), predictions.std())
        return predictions

    def recommend_batch(self,
                        user_ids      : List[int],
                        purchase_data : pd.DataFrame,
                        n_items       : int = 10,
                        user_col      : Optional[str] = None,
                        item_col      : Optional[str] = None,
                       ) -> pd.DataFrame:
        """
        Generate PURCHASE-AWARE top-N recommendations for multiple users
        via AryColBringInference.batch_clean_recommend(): item yang sudah
        pernah dibeli user dibuang, ditambal dari kandidat cadangan model,
        lalu fallback Item-to-Item kalau masih kurang -- jadi hasilnya
        benar-benar rekomendasi yang "bagus"/usable, bukan rekomendasi
        mentah yang bisa saja berisi barang yang sudah dibeli user.

        Returns
        -------
        pandas.DataFrame gabungan semua user: user_id, rank, item_id,
        score, source ('model_topn' | 'model_patch' | 'item2item_fallback'),
        is_fallback.
        """
        logger.info("Clean batch recommendations: %d users, top-%d items",
                     len(user_ids), n_items)
        recommendations = self.inference_service.batch_clean_recommend(
                          user_ids      = user_ids,
                          purchase_data = purchase_data,
                          n_items       = n_items,
                          user_col      = user_col,
                          item_col      = item_col,
                          output_format = "dataframe")
        logger.info("Clean recommendations generated for %d user(s)",
                     recommendations["user_id"].nunique() if len(recommendations) else 0)
        return recommendations

    def clean_single_recommend(self,
                               user_id       : int,
                               purchase_data : pd.DataFrame,
                               n_items       : int = 10,
                               user_col      : Optional[str] = None,
                               item_col      : Optional[str] = None,
                              ) -> pd.DataFrame:
        """Versi single-user dari recommend_batch(), lewat clean_recommend()."""
        logger.debug("Clean single recommendation: user = %s, n = %d", user_id, n_items)
        return self.inference_service.clean_recommend(
               user_id       = user_id,
               purchase_data = purchase_data,
               n_items       = n_items,
               user_col      = user_col,
               item_col      = item_col,
               output_format = "dataframe")

    def synthesize_purchase_data(self,
                                 user_ids      : List[int],
                                 min_purchases : int = 3,
                                 max_purchases : int = 15,
                                 random_state  : Optional[int] = None,
                                ) -> pd.DataFrame:
        """
        Bikin riwayat pembelian SINTETIS untuk keperluan demo/benchmark,
        dipakai kalau tidak ada file transaksi nyata (--purchase-data
        tidak diisi). Tiap test user diberi sejumlah acak item yang
        ditandai "sudah dibeli", supaya jalur pembersihan
        clean_recommend() (buang item terbeli -> patch -> fallback
        Item-to-Item) benar-benar teruji dan terlihat efeknya di metric,
        bukan cuma lewat kosongan (yang bikin hasilnya identik dengan
        rekomendasi mentah).
        """
        n_total_items = self.inference_service.predictor.item_embeddings.shape[0]
        seed          = random_state if random_state is not None else _cfg.getint(
                        'model', 'random_state', fallback = 42)
        rng           = np.random.default_rng(seed)

        rows = list()
        for user_id in user_ids:
            n_purchase = int(rng.integers(min_purchases, max_purchases + 1))
            n_purchase = min(n_purchase, n_total_items)
            for item_id in rng.choice(n_total_items, size = n_purchase, replace = False):
                rows.append({"user_id": int(user_id), "item_id": int(item_id)})

        purchase_df = pd.DataFrame(rows)
        logger.info("Synthesized %d purchase record(s) for %d test user(s) "
                    "(%d-%d items/user) to exercise clean_recommend().",
                     len(purchase_df), len(user_ids), min_purchases, max_purchases)
        return purchase_df

    def single_recommend(self,
                         user_id       : int,
                         n_items       : int = 10,
                         exclude_items : Optional[List[int]] = None,
                        ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a single user, sorted by score descending.
        """
        logger.debug("Single recommendation: user = %d, n = %d", user_id, n_items)
        return self.inference_service.recommend(
               user_id       = user_id,
               n_items       = n_items,
               exclude_items = exclude_items)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current inference performance metrics."""
        metrics = self.inference_service.get_metrics()
        logger.debug("Performance metrics: QPS = %.2f, avg_latency = %.2fms",
                      metrics.get("qps", 0), metrics.get("avg_latency_ms", 0))
        return metrics

    def generate_report(self,
                        experiment_name    : str = "Inference Run",
                        predictions_sample : Optional[List[Dict]] = None,
                       ) -> Path:
        """Generate inference performance report."""
        logger.info("Generating inference report: %s", experiment_name)
        metrics = self.get_performance_metrics()
        self._report_path = self.inference_service.generate_inference_report(
                            output_dir          = str(self.output_dir),
                            experiment_name     = experiment_name,
                            metrics             = metrics,
                            predictions_sample  = predictions_sample)
        logger.info("Report generated: %s", self._report_path)
        return self._report_path

    def run_performance_benchmark(self,
                                  n_predictions : int = 1000,
                                  batch_size    : int = 100,
                                  n_users       : Optional[int] = None,
                                  n_items       : Optional[int] = None,
                                 ) -> Dict:
        """Run performance benchmark and return summary metrics."""
        if n_users is None:
            n_users = self.inference_service.predictor.user_embeddings.shape[0]
        if n_items is None:
            n_items = self.inference_service.predictor.item_embeddings.shape[0]

        logger.info("Starting benchmark: %d predictions, batch_size = %d",
                     n_predictions, batch_size)

        if self.cache_enabled:
            self.inference_service.clear_cache()

        user_ids   = np.random.randint(0, n_users, n_predictions)
        item_ids   = np.random.randint(0, n_items, n_predictions)

        start_time  = time.perf_counter()
        predictions = self.predict_batch(user_ids, item_ids, return_cache = False)
        total_time  = (time.perf_counter() - start_time) * 1000

        metrics = self.get_performance_metrics()
        results = {"n_predictions"      : n_predictions,
                   "batch_size"         : batch_size,
                   "total_time_ms"      : total_time,
                   "predictions_per_sec": (n_predictions / total_time) * 1000,
                   "avg_latency_ms"     : metrics.get("avg_latency_ms", 0),
                   "qps"                : metrics.get("qps", 0),
                   "n_users"            : n_users,
                   "n_items"            : n_items}

        logger.info("Benchmark complete: %.2f predictions/sec, %.2fms avg latency",
                     results["predictions_per_sec"], results["avg_latency_ms"])
        return results

    def run_full_pipeline(self,
                          n_test_users           : int  = 100,
                          n_recommendations      : int  = 10,
                          run_benchmark          : bool = True,
                          benchmark_predictions  : int  = 1000,
                          purchase_data          : Optional[pd.DataFrame] = None,
                         ) -> Dict:
        """Run the complete inference pipeline and return a results summary."""
        logger.debug("Starting full inference pipeline!")
        results = {"status"     : "SUCCESS",
                   "timestamp"  : datetime.now().isoformat(),
                   "model_path" : str(self.model_path)}

        n_total_users = self.inference_service.predictor.user_embeddings.shape[0]
        n_total_items = self.inference_service.predictor.item_embeddings.shape[0]
        logger.info("Model dimensions: users = %d, items = %d",
                     n_total_users, n_total_items)

        test_users = np.random.choice(n_total_users,
                     min(n_test_users, n_total_users), replace = False).tolist()

        used_synthetic_data = purchase_data is None or purchase_data.empty
        if used_synthetic_data:
            purchase_data = self.synthesize_purchase_data(user_ids = test_users)

        # synthesize_purchase_data() always emits columns literally named
        # "user_id"/"item_id" -- pass them explicitly instead of relying on
        # DetectReco_Identifier's heuristics, which are tuned for real-world
        # transaction exports and aren't guaranteed to fire reliably on a
        # small synthetic frame. Real files loaded via --purchase-data still
        # go through auto-detection (user_col/item_col left as None) so
        # arbitrary column naming keeps working as designed.
        synth_user_col = "user_id" if used_synthetic_data else None
        synth_item_col = "item_id" if used_synthetic_data else None

        # Purchase-aware batch recommendation (clean_recommend machinery).
        recommendations = self.recommend_batch(
                          user_ids      = test_users,
                          purchase_data = purchase_data,
                          n_items       = n_recommendations,
                          user_col      = synth_user_col,
                          item_col      = synth_item_col)
        results["batch_recommendations_completed"] = (
        int(recommendations["user_id"].nunique()) if len(recommendations) else 0)

        # Ringkasan kualitas rekomendasi: berapa banyak slot Top-N yang
        # murni dari model, berapa yang ditambal dari kandidat cadangan,
        # dan berapa yang jatuh ke fallback Item-to-Item -- ini yang
        # bikin "bagusnya" hasil clean_recommend kelihatan di metric.
        if len(recommendations):
            source_breakdown = recommendations["source"].value_counts().to_dict()
            fallback_ratio    = float(recommendations["is_fallback"].mean())
        else:
            source_breakdown = dict()
            fallback_ratio    = 0.0
        results["clean_recommend_summary"] = {
            "total_slots_filled" : int(len(recommendations)),
            "source_breakdown"   : source_breakdown,
            "fallback_ratio"     : fallback_ratio,
        }

        # Sample untuk laporan (5 user pertama).
        sample_predictions = list()
        for user_id in test_users[:5]:
            user_rows = recommendations[recommendations["user_id"] == user_id]
            recs      = list(zip(user_rows["item_id"].astype(int),
                                 user_rows["score"].astype(float)))
            sample_predictions.append({"user_id": int(user_id), "recommendations": recs})

        report_path = self.generate_report(
                      experiment_name     = "Full Inference Pipeline (Clean Recommend)",
                      predictions_sample  = sample_predictions)
        results["report_path"] = str(report_path)

        if run_benchmark:
            results["benchmark"] = self.run_performance_benchmark(
                                   n_predictions = benchmark_predictions)

        results["metrics"] = self.get_performance_metrics()
        results["metrics"]["fallback_ratio"] = fallback_ratio
        logger.info("Pipeline completed successfully")
        return results


def main() -> None:
    parser = ArgumentParser(description = "Run AryColBring inference")
    parser.add_argument("-m", "--model",
                        type     = str,
                        required = True,
                        help     = "Path to trained model (.npz)")
    parser.add_argument("-o", "--output-dir",
                        type     = str,
                        default  = None,
                        help     = "Directory to save inference reports")
    parser.add_argument("-u", "--n-users",
                        type     = int,
                        default  = 100,
                        help     = "Number of test users to sample")
    parser.add_argument("-r", "--n-recs",
                        type     = int,
                        default  = 10,
                        help     = "Number of recommendations per user")
    parser.add_argument("-t", "--threads",
                        type     = int,
                        default  = None,
                        help     = "Number of threads for parallel inference")
    parser.add_argument("--no-cache",
                        action   = "store_true",
                        help     = "Disable prediction caching")
    parser.add_argument("-b", "--benchmark",
                        action   = "store_true",
                        help     = "Run performance benchmark")
    parser.add_argument("-s", "--benchmark-size",
                        type     = int,
                        default  = 1000,
                        help     = "Number of predictions to run during benchmark")
    parser.add_argument("-p", "--purchase-data",
                        type     = str,
                        default  = None,
                        help     = "Path to a real transaction history file (.csv/.parquet) "
                                    "with a user-id and item-id column (any names -- "
                                    "auto-detected). If omitted, synthetic purchase "
                                    "history is generated so clean_recommend()'s "
                                    "remove/patch/fallback path still gets exercised.")
    args = parser.parse_args()

    pipeline = AryColBring_Inference_Test(
               model_path    = args.model,
               output_dir    = args.output_dir,
               num_threads   = args.threads,
               cache_enabled = not args.no_cache)

    purchase_data = None
    if args.purchase_data:
        pd_path = Path(args.purchase_data)
        if not pd_path.exists():
            logger.error("Purchase-data file not found: %s", pd_path)
            raise FileNotFoundError(f"Purchase-data file not found: {pd_path}")
        purchase_data = (pd.read_parquet(pd_path) if pd_path.suffix.lower() == ".parquet"
                         else pd.read_csv(pd_path))
        logger.info("Loaded purchase data from %s: %d row(s)", pd_path, len(purchase_data))

    results = pipeline.run_full_pipeline(
              n_test_users          = args.n_users,
              n_recommendations     = args.n_recs,
              run_benchmark         = args.benchmark,
              benchmark_predictions = args.benchmark_size,
              purchase_data         = purchase_data)

    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Model: {results['model_path']}")
    print(f"Batch recommendations: {results.get('batch_recommendations_completed', 0)}")
    print(f"Report: {results.get('report_path', 'N/A')}")

    if "clean_recommend_summary" in results:
        print("\nClean Recommendation Quality:")
        summary = results["clean_recommend_summary"]
        print(f"  Total slots filled: {summary.get('total_slots_filled', 0)}")
        for source, count in summary.get("source_breakdown", {}).items():
            print(f"    - {source:<20}: {count}")
        print(f"  Fallback ratio: {summary.get('fallback_ratio', 0):.2%}")

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

    print("=" * 80 + "\n")


if __name__ == "__main__":
    #Sample Command:
    #python -m test.arycolbring_tests.t02_reasoner -m ./artifacts/ACBmodel/20260604_models.npz -u 100 -r 10 -b -s 1000
    print("Running the AryColBring_Reasoner_Test")
    try:
        main()
    except Exception as arc:
        logger.warning("Try this: "
        'python -m test.arycolbring_tests.t02_reasoner -m '
        './artifacts/ACBmodel/20260604_models.npz -u 100 -r 10 -b -s 1000')
        logger.exception("Inference pipeline failed with error: %s", str(arc))
        sys.exit(1)
