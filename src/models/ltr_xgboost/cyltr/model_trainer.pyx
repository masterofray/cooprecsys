# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Model Training Module - Core Cython Implementation
Handles XGBoost LTR model training and ranking metrics computation
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport log, sqrt, exp, fabs
from libc.stdio cimport fprintf, stderr
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

DTYPE = np.float64
ITYPE = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RankingMetricsCalculator:
    """
    High-performance ranking metrics computation using Cython
    """
    cdef public int k
    cdef public dict metrics_history
    
    def __cinit__(self, int k=10):
        self.k = k
        self.metrics_history = {}
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_ndcg(self, int[:] y_true, double[:] y_pred):
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain)
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
            
        Returns
        -------
        double
            NDCG@K score
        """
        cdef int n = len(y_pred)
        cdef int k = min(self.k, n)
        cdef int i, j
        cdef double dcg = 0.0, idcg = 0.0
        cdef int[:] indices = np.argsort(-y_pred)[:k].astype(np.int32)
        cdef int[:] true_indices = np.argsort(-y_true)[:k].astype(np.int32)
        cdef int[:] true_sorted = np.sort(y_true)[::-1][:k].astype(np.int32)
        
        fprintf(stderr, b"[RankingMetrics] Computing NDCG@%d\n", k)
        
        with nogil:
            # Compute DCG
            for i in range(k):
                dcg += (2 ** y_true[indices[i]] - 1) / log(i + 2)
            
            # Compute IDCG
            for i in range(k):
                idcg += (2 ** true_sorted[i] - 1) / log(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_map(self, int[:] y_true, double[:] y_pred):
        """
        Compute MAP@K (Mean Average Precision)
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores (binary or graded)
        y_pred : ndarray
            Predicted scores
            
        Returns
        -------
        double
            MAP@K score
        """
        cdef int n = len(y_pred)
        cdef int k = min(self.k, n)
        cdef int i, j
        cdef double ap = 0.0
        cdef int relevant_count = 0
        cdef int total_relevant = 0
        cdef int[:] indices = np.argsort(-y_pred)[:k].astype(np.int32)
        
        fprintf(stderr, b"[RankingMetrics] Computing MAP@%d\n", k)
        
        with nogil:
            # Count total relevant items
            for j in range(n):
                if y_true[j] > 0:
                    total_relevant += 1
            
            # Compute AP
            for i in range(k):
                if y_true[indices[i]] > 0:
                    relevant_count += 1
                    ap += <double>relevant_count / (i + 1)
        
        if total_relevant == 0:
            return 0.0
        
        return ap / min(total_relevant, k)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_mrr(self, int[:] y_true, double[:] y_pred):
        """
        Compute MRR (Mean Reciprocal Rank)
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
            
        Returns
        -------
        double
            MRR score
        """
        cdef int n = len(y_pred)
        cdef int k = min(self.k, n)
        cdef int i
        cdef int[:] indices = np.argsort(-y_pred)[:k].astype(np.int32)
        
        fprintf(stderr, b"[RankingMetrics] Computing MRR@%d\n", k)
        
        with nogil:
            for i in range(k):
                if y_true[indices[i]] > 0:
                    return 1.0 / (i + 1)
        
        return 0.0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_precision(self, int[:] y_true, double[:] y_pred):
        """
        Compute Precision@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
            
        Returns
        -------
        double
            Precision@K score
        """
        cdef int n = len(y_pred)
        cdef int k = min(self.k, n)
        cdef int i
        cdef int relevant_count = 0
        cdef int[:] indices = np.argsort(-y_pred)[:k].astype(np.int32)
        
        fprintf(stderr, b"[RankingMetrics] Computing Precision@%d\n", k)
        
        with nogil:
            for i in range(k):
                if y_true[indices[i]] > 0:
                    relevant_count += 1
        
        return <double>relevant_count / k
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_recall(self, int[:] y_true, double[:] y_pred):
        """
        Compute Recall@K
        
        Parameters
        ----------
        y_true : ndarray
            True relevance scores
        y_pred : ndarray
            Predicted scores
            
        Returns
        -------
        double
            Recall@K score
        """
        cdef int n = len(y_pred)
        cdef int k = min(self.k, n)
        cdef int i, j
        cdef int relevant_at_k = 0
        cdef int total_relevant = 0
        cdef int[:] indices = np.argsort(-y_pred)[:k].astype(np.int32)
        
        fprintf(stderr, b"[RankingMetrics] Computing Recall@%d\n", k)
        
        with nogil:
            for j in range(n):
                if y_true[j] > 0:
                    total_relevant += 1
            
            for i in range(k):
                if y_true[indices[i]] > 0:
                    relevant_at_k += 1
        
        if total_relevant == 0:
            return 0.0
        
        return <double>relevant_at_k / total_relevant


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class XGBoostLTRTrainer:
    """
    XGBoost Learning-to-Rank model trainer
    """
    cdef public object model
    cdef public object feature_names
    cdef public dict best_params
    cdef public dict training_history
    cdef public RankingMetricsCalculator metrics_calc
    
    def __cinit__(self):
        self.model = None
        self.feature_names = []
        self.best_params = {}
        self.training_history = {}
        self.metrics_calc = RankingMetricsCalculator(k=10)
    
    def set_params(self, dict params):
        """
        Set model hyperparameters
        
        Parameters
        ----------
        params : dict
            Hyperparameters for XGBoost
        """
        self.best_params = params
        fprintf(stderr, b"[XGBoostLTRTrainer] Parameters set\n")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              int epochs=100, int early_stopping_rounds=10):
        """
        Train XGBoost LTR model
        
        Parameters
        ----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training labels
        X_val : ndarray, optional
            Validation features
        y_val : ndarray, optional
            Validation labels
        epochs : int
            Number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
            
        Returns
        -------
        object
            Trained model
        """
        fprintf(stderr, b"[XGBoostLTRTrainer] Starting model training with %d rounds\n", epochs)
        
        # Prepare training data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        dval = None
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Train model
        evals_result = {}
        self.model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=epochs,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        
        self.training_history = evals_result
        fprintf(stderr, b"[XGBoostLTRTrainer] Training completed\n")
        
        return self.model
    
    def get_feature_importance(self, str importance_type='gain'):
        """
        Get feature importance scores
        
        Parameters
        ----------
        importance_type : str
            'gain', 'weight', 'cover', 'total_gain', 'total_cover'
            
        Returns
        -------
        dict
            Feature importance dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        fprintf(stderr, b"[XGBoostLTRTrainer] Computing feature importance (%s)\n", importance_type.encode())
        
        importance = self.model.get_score(importance_type=importance_type)
        return importance
    
    def save_model(self, str filepath):
        """
        Save trained model
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(filepath)
        fprintf(stderr, b"[XGBoostLTRTrainer] Model saved to %s\n", filepath.encode())
    
    def load_model(self, str filepath):
        """
        Load pre-trained model
        
        Parameters
        ----------
        filepath : str
            Path to model file
        """
        self.model = xgb.Booster(model_file=filepath)
        fprintf(stderr, b"[XGBoostLTRTrainer] Model loaded from %s\n", filepath.encode())