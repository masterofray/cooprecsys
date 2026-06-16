#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-05"


import sys
import time
from   pathlib    import Path
from   tqdm.auto  import tqdm
from   datetime   import datetime
from   typing     import Any, Dict, List
from   .inference import AryColBringInference

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[2]))
from configs import logger


class InferenceService:
    """
    High-level inference service for production deployment.
    Wraps AryColBringInference with additional features:
    - API endpoint simulation
    - Request batching
    - Health checks
    - Metrics export
    """
    def __init__(self,
                 model_path     : str,
                 num_threads    : int = 4,
                 max_batch_size : int = 1000,
                ) -> None:
        logger.info("Initializing InferenceService: model = %s", model_path)
        self.inference = AryColBringInference(
                         model_path    = model_path,
                         num_threads   = num_threads,
                         cache_enabled = True)
        self.max_batch_size = max_batch_size
        self.request_count  = 0


    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        metrics = self.inference.get_metrics()
        data    = {"status"       : "healthy",
                   "model_loaded" : True,
                   "model_path"   : str(self.inference.model_path),
                   "num_threads"  : self.inference.num_threads,
                   "metrics"      : metrics,
                   "timestamp"    : datetime.now().isoformat()}
        return data


    def predict_api(self,
                    user_id  : int,
                    item_ids : List[int],
                   ) -> Dict[str, Any]:
        """Simulate API endpoint for prediction.
           The return is dict with Response with predictions and metadata
        """
        self.request_count += 1
        start_time = time.perf_counter()
        scores     = self.inference.predict(
                     user_ids = user_id,
                     item_ids = item_ids)
        latency    = (time.perf_counter() - start_time) * 1000
        context    = {"request_id"  : self.request_count,
                      "user_id"     : user_id,
                      "predictions" : [{"item_id": iid, 
                                        "score"  : float(score)}
                                        for iid, score in zip(
                                        item_ids, scores)],
                      "latency_ms"  : latency,
                      "timestamp"   : datetime.now().isoformat()}
        logger.debug(f'The context is {context}.')
        return context


    def recommend_api(self,
                      user_id : int,
                      n_items : int = 10,
                     ) -> Dict[str, Any]:
        """Simulate API endpoint for recommendations.
           The return is dict type as Response with 
           recommendations and metadata.
        """
        self.request_count += 1
        start_time      = time.perf_counter()
        recommendations = self.inference.recommend(
                          user_id = user_id,
                          n_items = n_items)
        latency         = (time.perf_counter() - start_time) * 1000
        context         = {"request_id"      : self.request_count,
                           "user_id"         : user_id,
                           "recommendations" : [{"item_id": iid, 
                                                 "score"  : float(score)}
                                                 for iid, score in 
                                                 recommendations],
                           "n_items"         : n_items,
                           "latency_ms"      : latency,
                           "timestamp"       : datetime.now().isoformat()}
        logger.debug(f'The context is {context}.')
        return context


    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        logger.debug('Try to get status.')
        health  = self.health_check()
        metrics = self.inference.get_metrics()
        return {"service"             : "AryColBring Inference Service",
                "version"             : "1.0.0",
                "requests_handled"    : self.request_count,
                "health"              : health,
                "performance_metrics" : metrics,
                "timestamp"           : datetime.now().isoformat()}


if __name__ == "__main__":
    print("AryColBring Inference Service")
    print("=" * 50)