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
Complete training pipeline for AryColBring collaborative filtering model.

Integrates:
- Model training using Cython-optimized kernels from CLproximity
- Evaluation metrics computation
- Training report dashboard generation
- Model serialization for production deployment
"""

import sys
import json
import numpy as np
import scipy.sparse as sp
from pathlib   import Path
from datetime  import datetime
from copy      import deepcopy
from typing    import Any, Dict, List, Optional, Tuple, Union
from .assist   import fileload_interactions, describe_interactions
from .inout    import TheAdvisor, TheReasoner
from .narative import genAdvisor, OUTPUT_DIR
from .eval     import (precision_at_k,
                       recall_at_k,
                       auc_score,
                       MRR_rank,
                       NDCG_rank,
                       CCC_k,
                       ILD_k,
                       Novelty_k)

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[2]))
from configs   import logger
from features  import Normalize_LargeSeries, Filter_TopN


def Adjusted_CSRshape(matrix     : sp.csr_matrix, 
                      expected_m : int, 
                      expected_n : int,
                     ) -> sp.csr_matrix:
            current_m, current_n = matrix.shape
            data                 = matrix.data
            indices              = matrix.indices
            indptr               = matrix.indptr
            # Jika BARIS (Users) kurang, pad indptr dengan nilai
            # terakhirnya agar ukurannya sesuai (M + 1)
            if current_m < expected_m:
                pad_size = expected_m - current_m
                indptr   = np.concatenate([indptr, np.full(
                           pad_size, indptr[-1], dtype = indptr.dtype)])
            TheResult    = sp.csr_matrix((data, indices, indptr),
                           shape = (expected_m, expected_n))
            return TheResult


class AryColBringModelTrainer:
    """
    High-level training wrapper for AryColBring 
    collaborative filtering model. Provides a unified 
    interface for:
    - Training with multiple loss functions
    - Automatic evaluation during training
    - Dashboard report generation
    - Model export for production
    Argument:
    _____________________________________________________________________
    no_components     : int, Number of latent factors (default: 32)
    loss              : str, Loss function: "logistic", "warp", 
                        "bpr", or "warp-kos" (default: "warp")
    learning_rate     : float, Base learning rate (default: 0.05)
    item_alpha        : float, L2 regularization for items (default: 0.0)
    user_alpha        : float, L2 regularization for users (default: 0.0)
    learning_schedule : str, Learning rate schedule: "adagrad" 
                        or "adadelta" (default: "adagrad")
    random_state      : int | None, Random seed for reproducibility
    """
    def __init__(self,
                 no_components     : int   = 32,
                 loss              : str   = "warp",
                 learning_rate     : float = 0.05,
                 item_alpha        : float = 0.0,
                 user_alpha        : float = 0.0,
                 learning_schedule : str   = "adagrad",
                 random_state      : Optional[int] = 4,
                ) -> None:
        logger.info(
        "Initializing class with loss = %s components = %d lr = %.4f",
        loss, no_components, learning_rate)
        loss              = str(loss).lower()
        learning_schedule = str(learning_schedule).lower()
        self._user_ids    = list()
        self._item_ids    = list()
        self.trainer = TheAdvisor(no_components     = no_components,
                                  loss              = loss,
                                  learning_rate     = learning_rate,
                                  item_alpha        = item_alpha,
                                  user_alpha        = user_alpha,
                                  learning_schedule = learning_schedule,
                                  random_state      = random_state)
        self.training_history: List[Dict[str, Any]]  = list()
        self.metrics_history: List[Dict[str, float]] = list()
        self.config  = {"no_components"     : no_components,
                        "loss"              : loss,
                        "learning_rate"     : learning_rate,
                        "item_alpha"        : item_alpha,
                        "user_alpha"        : user_alpha,
                        "learning_schedule" : learning_schedule,
                        "random_state"      : random_state,
                       }

    def fit(self,
            interactions    : Union[sp.spmatrix, str, 'pd.DataFrame'],
            user_features   : Optional[sp.spmatrix] = None,
            item_features   : Optional[sp.spmatrix] = None,
            sample_weight   : Optional[sp.spmatrix] = None,
            epochs          : int                   = 10,
            num_threads     : int                   = 4,
            verbose         : bool                  = True,
            validation_data : Optional[sp.spmatrix] = None,
            evaluate_every  : int                   = 1,
           ) -> "AryColBringModelTrainer":
        """
        Train the collaborative filtering model.
        interactions    : sparse matrix | str | DataFrame
                          Training interactions (COO/CSR matrix, 
                          CSV path, or DataFrame)
        epochs          : int, Number of training epochs (default: 10)
        num_threads     : int, Number of parallel threads (default: 4)
        verbose         : bool, Show progress bar (default: True)
        validation_data : sparse matrix, Validation interactions for evaluation
        evaluate_every  : int, Evaluate metrics every N epochs (default: 1)
        """
        start_time = datetime.now()
        logger.debug("Starting training: epochs = %d | threads = %d",
                      epochs, num_threads)
        if isinstance(interactions, str):
            logger.info("Loading interactions from flatfile: %s",
                         interactions)
            interactions, self._user_ids, self._item_ids = fileload_interactions(interactions)
        if not sp.isspmatrix_coo(interactions):
            interactions = interactions.tocoo()
        
        data_stats = describe_interactions(interactions)
        stats_row = data_stats.iloc[0]
        logger.debug(
        "Training data: users = %d | items = %d | interactions = %d | sparsity = %.4f",
        stats_row["n_users"], stats_row["n_items"],
        stats_row["nnz"],     stats_row["density"])
        
        # Aktifkan variabel di bawah ini:
        self.trainer.fit(interactions  = interactions,
                         user_features = user_features,
                         item_features = item_features,
                         sample_weight = sample_weight,
                         epochs        = epochs,
                         num_threads   = num_threads,
                         verbose       = verbose)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("Training completed in %.2f seconds", duration)
        self.training_history.append({
            "epochs"            : epochs,
            "num_threads"       : num_threads,
            "training_time_sec" : duration,
            "start_time"        : start_time.isoformat(),
            "end_time"          : datetime.now().isoformat(),
            })
        if user_features is not None:
            self._user_features = user_features
        if item_features is not None:
            self._item_features = item_features

        if validation_data is not None and epochs % evaluate_every == 0:
            logger.debug("Evaluating on validation data.")
            metrics = self.evaluate(validation_data, 
                                    train_interactions = interactions,
                                    user_features      = user_features,
                                    item_features      = item_features,
                                    num_threads        = num_threads)
            self.metrics_history.append(metrics)
            logger.debug(
            "Validation metrics: AUC = %.4f | Precision@10 = %.4f | Recall@10 = %.4f",
             metrics.get("auc", 0),
             metrics.get("precision_at_10", 0),
             metrics.get("recall_at_10", 0))
        return self


    def evaluate(self,
                 test_interactions  : sp.spmatrix,
                 train_interactions : Optional[sp.spmatrix] = None,
                 user_features      : sp.spmatrix           = None,
                 item_features      : sp.spmatrix           = None,
                 num_threads        : int                   = 4,
                 k_values           : List[int]             = None,
                ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        test_interactions  : sparse matrix, Test interaction matrix
        train_interactions : sparse matrix, Training interactions 
                             (to exclude from ranking)
        num_threads        : int, Number of threads for evaluation
        k_values           : list of int, K values for precision/recall
                             (default: [5, 10, 20])
        This will return Dictionary of evaluation metrics
        """
        if k_values is None:
            k_values = [5, 10, 20]
        logger.info("Evaluating model with k = %s", k_values)
        self._test_interactions = test_interactions

        test_interactions = test_interactions.tocsr()
        if train_interactions is not None:
            train_interactions = train_interactions.tocsr()

        # Ambil dimensi ekspektasi dari bobot model trainer
        if self.trainer.item_embeddings is not None:
            expected_items = self.trainer.item_embeddings.shape[0]
        else:
            expected_items = test_interactions.shape[1]
        if self.trainer.user_embeddings is not None:
            expected_users = self.trainer.user_embeddings.shape[0]
        else:
            expected_users = test_interactions.shape[0]

        # Eksekusi penyesuaian jika dimensi test/train
        # lebih kecil dari bobot model
        if test_interactions.shape[0] < expected_users \
        or test_interactions.shape[1] < expected_items:
            logger.debug("Adjusting test_interactions shape from %s to %s",
            test_interactions.shape, 
            (expected_users, expected_items))
            test_interactions = Adjusted_CSRshape(
                                matrix     = test_interactions, 
                                expected_m = expected_users,
                                expected_n = expected_items)

        if train_interactions is not None:
            if train_interactions.shape[0] < expected_users \
            or train_interactions.shape[1] < expected_items:
                logger.debug("Adjusting train_interactions shape from %s to %s",
                train_interactions.shape, 
                (expected_users, expected_items))
                train_interactions = Adjusted_CSRshape(
                                     matrix     = train_interactions,
                                     expected_m = expected_users,
                                     expected_n = expected_items)

        # Bypass Error Leakage
        if train_interactions is not None:
            overlap = test_interactions.multiply(train_interactions.astype(bool))
            if overlap.nnz > 0:
                logger.warning("Data Leakage Detected! "
                "Membuang %d interaksi overlap dari test_interactions.",
                overlap.nnz)
                test_interactions = test_interactions - overlap
                test_interactions.eliminate_zeros()

        # Make sure for item_features & user_features
        # have same rows ILD / Novelty
        if item_features is None or (
        item_features.shape[0] < expected_items):
            logger.warning("item_features kurang baris (%s) atau None."
            "Membuat fallback Identity Matrix untuk %d items.", 
            item_features.shape if item_features is not None else "None",
            expected_items)
            item_features = sp.eye(expected_items, format = "csr")
        if user_features is None or user_features.shape[0] < expected_users:
            user_features = sp.eye(expected_users, format = "csr")

        metrics       = dict()
        predictor     = TheReasoner(**self.config)
        initial_attrs = ["item_embeddings",
                         "item_embedding_gradients",
                         "item_embedding_momentum",
                         "item_biases",
                         "item_bias_gradients",
                         "item_bias_momentum",
                         "user_embeddings",
                         "user_embedding_gradients",
                         "user_embedding_momentum",
                         "user_biases",
                         "user_bias_gradients",
                         "user_bias_momentum"]
        for attr in initial_attrs:
            val = getattr(self.trainer, attr, None)
            setattr(predictor, attr, val)
        self._predictor = predictor
        
        #AUC
        try:
            res_auc        = auc_score(self._predictor, 
                                       test_interactions, 
                                       train_interactions = train_interactions, 
                                       num_threads        = num_threads)
            metrics["auc"] = float(res_auc.mean())
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)
            metrics["auc"] = 0.0

        #MRR
        try:
            res_mrr        = MRR_rank(self._predictor, 
                                      test_interactions, 
                                      train_interactions = train_interactions,
                                      num_threads        = num_threads)
            metrics["mrr"] = float(res_mrr.mean())
        except Exception as e:
            logger.warning("MRR computation failed: %s", e)
            metrics["mrr"] = 0.0


        for k in k_values:
            
            #Precision@K
            try:
                res_p = precision_at_k(
                        self._predictor, test_interactions, 
                        k                    = k,
                        train_interactions   = train_interactions,
                        num_threads          = num_threads)
                metrics[f"precision_at_{k}"] = float(res_p.mean())
            except Exception as e:
                logger.warning("Precision@%d computation failed: %s", k, e)
                metrics[f"precision_at_{k}"] = 0.0
            
            #Recall@K
            try:
                rec_p = recall_at_k(self._predictor,
                        test_interactions,
                        k                  = k,
                        train_interactions = train_interactions,
                        num_threads        = num_threads)
                metrics[f"recall_at_{k}"]  = float(rec_p.mean())
            except Exception as e:
                logger.warning("Recall@%d computation failed: %s", k, e)
                metrics[f"recall_at_{k}"]  = 0.0

            #NDCG@K
            try:
                ndcgk = NDCG_rank(model    = self._predictor,
                        test_interactions  = test_interactions,
                        train_interactions = train_interactions,
                        num_threads        = num_threads,
                        k                  = k)
                metrics[f"NDCG_at_{k}"] = float(ndcgk.mean())
            except Exception as e:
                logger.warning("NDCG@%d computation failed: %s", k, e)
                metrics[f"NDCG_at_{k}"] = 0.0


            #CCC@K
            try:
                metrics[f"CCC_at_{k}"] = float(
                CCC_k(model              = self._predictor,
                      test_interactions  = test_interactions,
                      train_interactions = train_interactions,
                      num_threads        = num_threads,
                      k                  = k))
            except Exception as e:
                logger.warning("CCC@%d computation failed: %s", k, e)
                metrics[f"CCC_at_{k}"] = 0.0

            #ILD@K
            try:
                ildk = ILD_k(model        = self._predictor,
                       test_interactions  = test_interactions,
                       train_interactions = train_interactions,
                       user_features      = user_features,
                       item_features      = item_features,
                       num_threads        = num_threads,
                       k                  = k)
                metrics[f"ILD_at_{k}"] = float(ildk.mean())
            except Exception as e:
                logger.warning("ILD@%d computation failed: %s", k, e)
                metrics[f"ILD_at_{k}"] = 0.0

            #Novelty@K
            try:
                novelk = Novelty_k(model    = self._predictor,
                         test_interactions  = test_interactions,
                         train_interactions = train_interactions,
                         user_features      = user_features,
                         item_features      = item_features,
                         num_threads        = num_threads,
                         k                  = k)
                metrics[f"Novelty_at_{k}"] = float(novelk.mean())
            except Exception as e:
                logger.warning("Novelty@%d computation failed: %s", k, e)
                metrics[f"Novelty_at_{k}"] = 0.0

        return metrics
    
    
    def generate_training_report(self,
            output_dir      : Optional[str] = None,
            experiment_name : str = "Default Experiment",
            charts          : Optional[List[Dict[str, Any]]] = None,
        ) -> Path:
        """
        Generate a comprehensive training dashboard
        report matching LTR layout. Splits telemetry 
        into Overviews, Ranking Quality, and 
        Diagnostic trends.
        """
        logger.debug("Generating comprehensive LTR-style training report.")
        n_interactions  = 0
        sparsity        = 0.0
        predictionDF    = self._predictor.predict(
                            user_ids      = self._user_ids,
                            item_ids      = self._item_ids,
                            item_features = self._item_features,
                            user_features = self._user_features,
                            num_threads   = 8)
        predictionDF    = Normalize_LargeSeries(predictionDF, 'score')
        predictionDF    = Filter_TopN(
                          Data      = predictionDF,
                          user_col  = 'user_id',
                          score_col = 'score')
        predictionDict  = predictionDF.to_dict(orient = 'records')

        current_metrics = self.metrics_history[-1] if \
                          self.metrics_history else dict()
        if self.trainer.item_embeddings is not None:
            n_items     = self.trainer.item_embeddings.shape[0]
        else:
            n_items     = self._test_interactions.shape[1]

        if self.trainer.user_embeddings is not None:
            n_users     = self.trainer.user_embeddings.shape[0]
        else:
            n_users     = self._test_interactions.shape[0]

        if hasattr(self, 'train_interactions') and self.train_interactions is not None:
            n_interactions = self.train_interactions.nnz
            total_elements = n_users * n_items
            if total_elements > 0:
                sparsity   = 1.0 - (n_interactions / total_elements)
        elif self.training_history:
            n_interactions = self.training_history[0].get("n_interactions", 0)
            sparsity       = self.training_history[0].get("sparsity", 0.0)

        generated_charts   = charts or list()
        if not generated_charts and self.training_history:
            ELP = [{"epoch": h.get("epoch", idx+1), 
                    "loss": h.get("loss_value", 0.0)}
                    for idx, h in enumerate(
                    self.training_history) if "loss_value" in h]
            if ELP:
                generated_charts.append({
                "type"  : "line",
                "title" : f"Convergence Curve ({self.config['loss'].upper()})",
                "label" : "Loss Value",
                "data"  : ELP,})
        Context = {
        "experiment_name"   : experiment_name,
        "loss"              : self.config["loss"],
        "epochs"            : self.training_history[-1]["epochs"] if \
                              self.training_history else 0,
        "no_components"     : self.config["no_components"],
        "learning_rate"     : self.config["learning_rate"],
        "item_alpha"        : self.config["item_alpha"],
        "user_alpha"        : self.config["user_alpha"],
        "learning_schedule" : self.config["learning_schedule"],
        "training_time_sec" : self.training_history[-1].get("training_time_sec",
                              0) if self.training_history else 0,
        "data_statistics"   : {"n_users"       : n_users,
                               "n_items"       : n_items,
                               "n_interactions": n_interactions,
                               "sparsity"      : sparsity,
                               "density"       : 1.0 - sparsity if sparsity > 0 else 0.0},
        "metrics"           : current_metrics,
        "charts"            : generated_charts,
        "history"           : {"training": self.training_history,
                               "metrics" : self.metrics_history,},
        "predictiondata"    : predictionDict,
        }

        OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
        today    = datetime.today().strftime('%Y%m%d')
        Contpath = OUTPUT_DIR / f"{today}_ACBcontext.json"
        with Contpath.open(mode = "w", encoding = "utf-8") as jfile:
            json.dump(Context,
                      fp           = jfile,
                      indent       = 2,
                      allow_nan    = False,
                      ensure_ascii = False)
        #sys.exit()
        
        RPath = genAdvisor(context_data = Context,
                           output_dir   = output_dir)
        logger.info("Training report successfully compiled "
                    "and aligned with LTR standards: %s", RPath)
        return RPath


    def save_model(self, path: str):
        """
        Serialize and save the trained model embeddings and configurations to disk.
        Safely packs weights, metadata, and hyper-parameters into a compressed NPZ archive.
        """
        logger.info("Initiating model serialization process to: %s", path)
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Enrich configurations with tracking metadata for model auditing
            exconf = deepcopy(self.config)
            exconf["serialized  _at"] = datetime.utcnow().isoformat()
            exconf["model_version"] = "0.0.1"
            
            # Safely encode JSON to a raw byte array without requiring pickle utilities
            config_bytes = json.dumps(exconf).encode('utf-8')
            config_array = np.frombuffer(config_bytes, dtype = np.uint8)

            # Construct the serialization dataset mapping
            dataset = {
                "item_embeddings" : self.trainer.item_embeddings,
                "user_embeddings" : self.trainer.user_embeddings,
                "item_biases"     : self.trainer.item_biases,
                "user_biases"     : self.trainer.user_biases,
                "config"          : config_array,}

            # Persist to disk using compressed archives with strict security parameters
            np.savez_compressed(path, allow_pickle=False, **dataset)
            logger.info("Model weights and structural metadata successfully written to disk.")
            
        except Exception as exc:
            logger.error("Failed to execute model saving pipeline "
            "due to an unhandled exception.", exc_info = True)
            raise IOError() from exc


    def load_model(self, path: str) -> "AryColBringModelTrainer":
        """
        Load, validate, and reconstruct a trained model checkpoint from a compressed NPZ file.
        Enforces strict structural validation and secure non-pickle array decoding.
        """
        logger.info("Attempting to load model checkpoint from path: %s", path)
        path = Path(path)
        
        if not path.exists():
            logger.error("Checkpoint file path does not exist: %s", path)
            raise FileNotFoundError(f"Model checkpoint file not found at location: {path}")

        try:
            # Enforce allow_pickle=False to strictly mitigate arbitrary code execution risks
            with np.load(path, allow_pickle=False) as data:
                
                # 1. Structural Integrity Check: Verify presence of all required model blocks
                required_keys = ["item_embeddings", "user_embeddings", "item_biases", "user_biases", "config"]
                for key in required_keys:
                    if key not in data:
                        raise KeyError(f"Corrupted checkpoint state: Missing required structural key '{key}'")

                # 2. Safe JSON Metadata Reconstruction
                try:
                    raw_config_bytes = data["config"].tobytes()
                    loaded_config = json.loads(raw_config_bytes.decode('utf-8'))
                except Exception as json_exc:
                    raise ValueError("Failed to deserialize model hyper-parameters from binary configuration block.") from json_exc

                # Extract weights for dimensional cross-verification
                item_embeds = data["item_embeddings"]
                user_embeds = data["user_embeddings"]
                item_biases = data["item_biases"]
                user_biases = data["user_biases"]

                # 3. Dimensional Verification Check
                expected_components = loaded_config.get("no_components", self.config.get("no_components"))
                if item_embeds.ndim != 2 or item_embeds.shape[1] != expected_components:
                    raise ValueError(f"Shape mismatch: item_embeddings shape {item_embeds.shape} "
                                     f"incompatible with no_components={expected_components}")
                
                if user_embeds.ndim != 2 or user_embeds.shape[1] != expected_components:
                    raise ValueError(f"Shape mismatch: user_embeddings shape {user_embeds.shape} "
                                     f"incompatible with no_components={expected_components}")

                if item_biases.ndim != 1 or item_biases.shape[0] != item_embeds.shape[0]:
                    raise ValueError(f"Shape mismatch: item_biases size {item_biases.shape[0]} "
                                     f"does not align with item cardinality {item_embeds.shape[0]}")

                if user_biases.ndim != 1 or user_biases.shape[0] != user_embeds.shape[0]:
                    raise ValueError(f"Shape mismatch: user_biases size {user_biases.shape[0]} "
                                     f"does not align with user cardinality {user_embeds.shape[0]}")

                # 4. Atomic Updates: Commit state arrays back to internal objects once validation passes
                self.config = loaded_config
                self.trainer.item_embeddings = item_embeds
                self.trainer.user_embeddings = user_embeds
                self.trainer.item_biases     = item_biases
                self.trainer.user_biases     = user_biases

            # Extract tracking statistics for analytical diagnostics
            serialized_time = self.config.get("serialized_at", "Unknown Timestamp")
            logger.info("Model checkpoint successfully restored. Metrics generation time context: %s", serialized_time)
            logger.info("Loaded weights summary: Users=%d, Items=%d, Latent Components=%d", 
                        user_embeds.shape[0], item_embeds.shape[0], expected_components)
            
            return self

        except Exception as exc:
            logger.error("Failed to safely load model state from target NPZ archive.", exc_info=True)
            raise RuntimeError("Model reconstruction aborted due to structural or validation anomalies.") from exc

    
    def get_predictor(self) -> TheReasoner:
        """
        Get a predictor instance for production inference.
        The return is TheReasoner, Ready-to-use predictor
        with loaded embeddings
        """
        predictor                 = TheReasoner(**self.config)
        predictor.item_embeddings = self.trainer.item_embeddings
        predictor.user_embeddings = self.trainer.user_embeddings
        predictor.item_biases     = self.trainer.item_biases
        predictor.user_biases     = self.trainer.user_biases
        logger.debug('Done for get_predictor in reasoner model !!!')
        return predictor


def RunTrainer(
        train_data      : Union[sp.spmatrix, str],
        epochs          : int   = 10,
        no_components   : int   = 32,
        loss            : str   = "warp",
        learning_rate   : float = 0.05,
        num_threads     : int   = 4,
        output_dir      : Optional[str]         = None,
        experiment_name : str                   = "Training Run",
        validation_data : Optional[sp.spmatrix] = None,
        save_model_path : Optional[str]         = None,
    ) -> Tuple[AryColBringModelTrainer, Path]:
    """
    Convenience function to train a model and generate a report.
       - train_data       : sparse matrix | str, Training data (matrix or CSV path)
       - epochs          : int, Number of training epochs
       - no_components   : int, Number of latent factors
       - loss            : str, Loss function ("logistic", "warp", "bpr", "warp-kos")
       - learning_rate   : float, Learning rate
       - num_threads     : int, Number of parallel threads
       - output_dir      : str, Directory for reports
       - experiment_name : str, Experiment name for the report
       - validation_data : sparse matrix, Validation data for evaluation
       - save_model_path : str, Path to save the trained model
    """
    logger.info("Starting training pipeline: %s", experiment_name)
    model = AryColBringModelTrainer(no_components = no_components,
                                    loss          = loss,
                                    learning_rate = learning_rate)
    model.fit(interactions    = train_data,
              epochs          = epochs,
              num_threads     = num_threads,
              validation_data = validation_data)
    report_path = model.generate_training_report(
                  output_dir      = output_dir,
                  experiment_name = experiment_name)
    if save_model_path:
        model.save_model(save_model_path)
    logger.info("Training pipeline completed successfully")
    return model, report_path


if __name__ == "__main__":
    # Example usage
    print("AryColBring Model Trainer")
    print("=" * 50)
    print("Use AryColBringModelTrainer class or train_and_report function")
    print("to train collaborative filtering models with dashboard reports.")
