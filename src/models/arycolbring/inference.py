#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__modified__   = "2026-07-18"


"""
inference.py
__________________________________________________
Production inference pipeline for AryColBring collaborative filtering model.
Integrates:
    - Model loading from saved embeddings
    - Batch and real-time prediction APIs
    - Inference metrics computation
    - Production dashboard report generation
    - Performance monitoring (latency, throughput)
"""

import json
import time
import inspect
import numpy  as np
import pandas as pd
from   pathlib    import Path
from   tqdm.auto  import tqdm
from   datetime   import datetime
from   typing     import (Any, Dict, List,
                          Optional, Tuple, Union)
from   .inout             import TheReasoner, AryInfFallBack
from   .narative          import genReasoner

#LocDir = Path(__file__).resolve()
#sys.path.append(str(LocDir.parents[2]))
from ...configs import _cfg, logger


class AryColBringInference:
    """
    Production-ready inference wrapper for AryColBring
    collaborative filtering model.
    Provides:
    - Fast batch predictions
    - Real-time single predictions
    - Top-N recommendations per user
    - Performance monitoring
    - Inference dashboard generation
    __________________________________________________
    Argument:
    - model_path    : str, Path to the saved model (.npz file)
    - num_threads   : int, Number of threads for parallel inference
    - cache_enabled : bool, Enable prediction caching
    """
    def __init__(self,
                 model_path    : str,
                 num_threads   : int  = 4,
                 cache_enabled : bool = True,
                ) -> None:
        logger.info("Initializing the parameter: model = %s threads = %d", 
                     model_path, num_threads)
        self.model_path    = Path(model_path)
        self.num_threads   = num_threads
        self.cache_enabled = cache_enabled
        self._archive       = self._read_archive()
        self.config         = self._get_config()
        self.predictor       = self._load_model()
        self.prediction_cache: Dict[str, float] = dict() if cache_enabled else None
        self.inference_stats = {"n_predictions"    : 0,
                                "n_users_served"   : 0,
                                "total_latency_ms" : 0.0,
                                "start_time"       : datetime.now().isoformat()}


    def _read_archive(self) -> np.lib.npyio.NpzFile:
        """Read the saved .npz archive once and reuse it for config + weights."""
        logger.info("Loading model from: %s", self.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return np.load(self.model_path, allow_pickle = False)


    def _load_model(self) -> TheReasoner:
        data = self._archive
        valid_params = set(inspect.signature(TheReasoner.__init__).parameters) - {"self"}
        model_kwargs  = {k: v for k, v in self.config.items() if k in valid_params}
        dropped_keys  = set(self.config) - valid_params
        if dropped_keys:
            logger.debug("Ignoring non-constructor config key(s): %s", sorted(dropped_keys))

        predictor = TheReasoner(**model_kwargs)
        predictor.item_embeddings = data["item_embeddings"]
        predictor.user_embeddings = data["user_embeddings"]
        predictor.item_biases     = data["item_biases"]
        predictor.user_biases     = data["user_biases"]
        predictor.item_embedding_gradients = np.zeros_like(predictor.item_embeddings)
        predictor.item_embedding_momentum  = np.zeros_like(predictor.item_embeddings)
        predictor.item_bias_gradients      = np.zeros_like(predictor.item_biases)
        predictor.item_bias_momentum       = np.zeros_like(predictor.item_biases)
        predictor.user_embedding_gradients = np.zeros_like(predictor.user_embeddings)
        predictor.user_embedding_momentum  = np.zeros_like(predictor.user_embeddings)
        predictor.user_bias_gradients      = np.zeros_like(predictor.user_biases)
        predictor.user_bias_momentum       = np.zeros_like(predictor.user_biases)

        logger.debug("Model loaded successfully: users = %d items = %d",
                      predictor.user_embeddings.shape[0],
                      predictor.item_embeddings.shape[0])
        return predictor


    def _get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        try:
            raw_config_bytes = self._archive["config"].tobytes()
            return json.loads(raw_config_bytes.decode('utf-8'))
        except Exception as e:
            logger.warning("Could not load config: %s", e)
            return dict()


    def predict(self,
                user_ids     : Union[int, List[int], np.ndarray],
                item_ids     : Union[int, List[int], np.ndarray],
                return_cache : bool = True,
               ) -> np.ndarray:
        """
        Predict scores for user-item pairs.
        Its return be ndarray as Prediction scores (float32)
        """
        start_time = time.perf_counter()
        if isinstance(user_ids, int):
            user_ids = np.array([user_ids], dtype = np.int32)
        elif not isinstance(user_ids, np.ndarray):
            user_ids = np.asarray(user_ids, dtype = np.int32)
        if isinstance(item_ids, int):
            item_ids = np.array([item_ids], dtype = np.int32)
        elif not isinstance(item_ids, np.ndarray):
            item_ids = np.asarray(item_ids, dtype = np.int32)
        if len(user_ids) != len(item_ids):
            logger.warning("user_ids and item_ids must have same length")
            #raise ValueError()

        if self.cache_enabled and return_cache:
            cached_results   = list()
            uncached_indices = list()
            for i, (uid, iid) in enumerate(tqdm(
                    zip(user_ids, item_ids), 
                    total       = len(user_ids), 
                    desc        = "Processing the Prediction",
                    colour      = _cfg.get('tqdm', 'colour'),
                    ncols       = _cfg.getint('tqdm', 'ncols'),
                    bar_format  = _cfg.get('tqdm', 'BarFormats'),
                    unit        = 'process',
                    mininterval = 0.1)):
                cache_key = f"{uid}_{iid}"
                if cache_key in self.prediction_cache:
                    cached_results.append((i,
                    self.prediction_cache[cache_key]))
                else:
                    uncached_indices.append(i)

            # Seed results with whatever was already cached.
            results = np.zeros(len(user_ids), dtype = np.float32)
            for idx, score in cached_results:
                results[idx] = score

            # Only compute the pairs that weren't cached.
            if uncached_indices:
                uncached_users  = user_ids[uncached_indices]
                uncached_items  = item_ids[uncached_indices]
                uncached_scores = self.predictor.predict(
                                  user_ids    = uncached_users,
                                  item_ids    = uncached_items,
                                  num_threads = self.num_threads)

                for idx, score in zip(uncached_indices, uncached_scores):
                    cache_key = f"{user_ids[idx]}_{item_ids[idx]}"
                    self.prediction_cache[cache_key] = float(score)
                    results[idx] = score

            predictions = results

        else:
            # No cache - compute all
            predictions = self.predictor.predict(
                          user_ids    = user_ids,
                          item_ids    = item_ids,
                          num_threads = self.num_threads)
        
        # Update stats (always, regardless of cache hit/miss path)
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.inference_stats["n_predictions"]    += len(user_ids)
        self.inference_stats["total_latency_ms"] += latency_ms
        logger.debug("Information about Predictions as n = %d with latency = %.2fms",
                      len(user_ids), latency_ms)
        return predictions

    
    def recommend(self,
                  user_id       : int,
                  n_items       : int = 10,
                  exclude_items : Optional[List[int]] = None,
                 ) -> List[Tuple[int, float]]:
        """
        Get top-N item recommendations for a user.
        user_id       : int, User ID
        n_items       : int, Number of recommendations (default: 10)
        exclude_items : list of int, optional
                        Items to exclude (e.g., already purchased)
        The return is list of tuples, e.g. [(item_id, score), ...]
        sorted by score descending
        """
        start_time     = time.perf_counter()
        n_total_items  = self.predictor.item_embeddings.shape[0]
        if exclude_items:
            candidates = [i for i in range(n_total_items) if i not in exclude_items]
        else:
            candidates = list(range(n_total_items))
        
        # Score all candidates
        user_array      = np.full(len(candidates), user_id, dtype = np.int32)
        item_array      = np.array(candidates, dtype = np.int32)
        scores          = self.predictor.predict(
                          user_ids    = user_array,
                          item_ids    = item_array,
                          num_threads = self.num_threads)
        top_indices     = np.argsort(scores)[::-1][:n_items]
        recommendations = [(int(candidates[idx]), float(scores[idx]))
                            for idx in top_indices]
        
        # Update stats
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.inference_stats["n_predictions"]    += len(candidates)
        self.inference_stats["n_users_served"]   += 1
        self.inference_stats["total_latency_ms"] += latency_ms
        return recommendations


    def batch_recommend(self,
                        user_ids     : List[int],
                        n_items      : int = 10,
                        exclude_dict : Optional[Dict[int, List[int]]] = None,
                       ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users in batch.
        user_ids     : list of int, List of user IDs
        n_items      : int, Number of recommendations per user
        exclude_dict : dict, Mapping of user_id -> items to exclude
        The return is dict type with {user_id: [(item_id, score), ...], ...}
        """
        logger.info("Batch recommendations with users = %d n = %d",
                     len(user_ids), n_items)
        start_time = time.perf_counter()

        results    = dict()
        for user_id in tqdm(
                user_ids,
                desc        = "Batch Prediction",
                colour      = _cfg.get('tqdm', 'colour'),
                ncols       = _cfg.getint('tqdm', 'ncols'),
                bar_format  = _cfg.get('tqdm', 'BarFormats'),
                unit        = 'userID',
                mininterval = 0.1):
            exclude_items    = exclude_dict.get(user_id) if exclude_dict else None
            results[user_id] = self.recommend(
                               user_id       = user_id,
                               n_items       = n_items,
                               exclude_items = exclude_items)
        total_latency = (time.perf_counter() - start_time) * 1000
        avg_latency   = total_latency / len(user_ids) if user_ids else 0

        logger.info("Batch complete with users = %d total = %.2fms avg = %.2fms",
                     len(user_ids), total_latency, avg_latency)
        return results


    def clean_recommend(self,
                        user_id         : int,
                        purchase_data   : pd.DataFrame,
                        n_items         : int = 10,
                        user_col        : Optional[str] = None,
                        item_col        : Optional[str] = None,
                        overscan_factor : int = 3,
                        output_format   : str = "dataframe",
                        db_path         : Optional[Union[str, Path]] = None,
                        table_name      : str = "clean_recommendations",
                       ) -> Union[pd.DataFrame, str]:
        """
        Rekomendasi Top-N yang sudah dibersihkan dari item yang pernah
        dibeli user_id, dengan fallback berlapis lewat AryInfFallBack:

        1. Cek riwayat pembelian user_id di purchase_data (kolom user/item
           terdeteksi otomatis, apapun nama kolomnya).
        2. Ambil rekomendasi Top-N dari model (di-overscan lebih besar
           dari n_items supaya ada kandidat cadangan).
        3. Buang item yang ternyata sudah pernah dibeli user_id.
        4. Tambal (patch) slot yang kosong dari kandidat cadangan
           model tadi (item-item peringkat di bawah Top-N asli).
        5. Kalau kandidat model masih belum cukup untuk menutup n_items,
           fallback ke algoritma Item-to-Item (cosine similarity atas
           embedding item hasil training), diseed dari riwayat pembelian.
        6. Kembalikan sebagai pandas.DataFrame ("dataframe"), atau
           ekspor ke DuckDB flat-file .db ("duckdb").

        purchase_data   : DataFrame riwayat transaksi (kolom bebas, akan
                          dideteksi otomatis). user_id/item_id di dalamnya
                          harus berada di ruang ID yang sama dengan yang
                          dipakai model (lihat catatan di AryInfFallBack).
        overscan_factor : Kelipatan n_items yang diminta ke model.predict
                          untuk membentuk kandidat cadangan (Point 4).
        output_format   : "dataframe" atau "duckdb".
        db_path/table_name : hanya dipakai kalau output_format = "duckdb".
        """
        if output_format not in ("dataframe", "duckdb"):
            msg = f"output_format must be 'dataframe' or 'duckdb', got '{output_format}'."
            logger.error(msg)
            raise ValueError(msg)

        fallback = AryInfFallBack(
                   purchase_data   = purchase_data,
                   item_embeddings = self.predictor.item_embeddings,
                   user_col        = user_col,
                   item_col        = item_col)

        n_total_items  = self.predictor.item_embeddings.shape[0]
        overscan_n     = min(max(n_items * max(overscan_factor, 1), n_items), n_total_items)
        candidate_pool = self.recommend(user_id = user_id, n_items = overscan_n)

        result_df = fallback.clean_recommendations(
                    user_id        = user_id,
                    candidate_pool = candidate_pool,
                    n_items        = n_items)

        if len(result_df) < n_items:
            logger.warning(
            "clean_recommend for user %s only produced %d/%d item(s) even "
            "after Item-to-Item fallback (catalog may be too small or "
            "user has bought almost the entire catalog).",
             user_id, len(result_df), n_items)

        if output_format == "duckdb":
            return fallback.export_duckdb(
                   result_df  = result_df,
                   db_path    = db_path,
                   table_name = table_name)
        return result_df


    def get_metrics(self) -> Dict[str, float]:
        """Get current inference performance metrics."""
        n_preds       = self.inference_stats["n_predictions"]
        total_latency = self.inference_stats["total_latency_ms"]
        avg_latency   = total_latency / n_preds if n_preds > 0 else 0
        throughput    = n_preds / (total_latency / 1000) if total_latency > 0 else 0
        elapsed_time  = (datetime.now() - datetime.fromisoformat(
                         self.inference_stats["start_time"])).total_seconds()
        qps           = n_preds / elapsed_time if elapsed_time > 0 else 0
        themetric     = {"qps"              : qps,
                         "n_predictions"    : n_preds,
                         "avg_latency_ms"   : avg_latency,
                         "elapsed_time_sec" : elapsed_time,
                         "n_users_served"   : self.inference_stats["n_users_served"],
                         "throughput_preds_per_sec" : throughput}
        return themetric


    def generate_inference_report(
            self,
            output_dir         : Optional[str]                  = None,
            experiment_name    : str                            = "Inference Run",
            metrics            : Optional[Dict[str, float]]     = None,
            predictions_sample : Optional[List[Dict[str, Any]]] = None,
            charts             : Optional[List[Dict[str, Any]]] = None,
        ) -> Path:
        """Generate an inference performance dashboard report."""
        logger.info("Generating inference report.")
        default_metrics = self.get_metrics()
        if metrics:
            default_metrics.update(metrics)
        context_data = {
            "metrics"             : default_metrics,
            "inference_statistics": {"n_predictions"  : default_metrics["n_predictions"],
                                     "n_users_served" : default_metrics["n_users_served"],
                                     "avg_latency_ms" : default_metrics["avg_latency_ms"],
                                     "coverage"       : 0.75,},  #Placeholder
            "experiment_name"     : experiment_name,
            "model_version"       : "0.0.1", #Placeholder
            "batch_size"          : 100,     #Placeholder
            "num_threads"         : self.num_threads,
            "predictions"         : predictions_sample or list(),
            "charts"              : charts or list(),
            }
        # Generate report
        SimpanPath  = genReasoner(
                      context_data = context_data,
                      output_dir   = output_dir)
        logger.debug("Inference report generated as %s", SimpanPath)
        return SimpanPath


    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        if self.prediction_cache is not None:
            self.prediction_cache.clear()
            logger.debug("Prediction cache cleared")


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
