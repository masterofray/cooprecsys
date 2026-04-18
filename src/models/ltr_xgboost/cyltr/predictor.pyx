'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Prediction Module - Core Cython Implementation
Handles batch prediction, ranking, and result formatting
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy
import xgboost as xgb

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

DTYPE = np.float64
ITYPE = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BatchPredictor:
    """
    High-performance batch prediction using Cython
    """
    cdef public object model
    cdef public int batch_size
    cdef public dict prediction_cache
    
    def __cinit__(self, int batch_size=1000):
        self.model = None
        self.batch_size = batch_size
        self.prediction_cache = {}
    
    def set_model(self, object model):
        """
        Set XGBoost model
        
        Parameters
        ----------
        model : xgboost.Booster
            Trained XGBoost model
        """
        self.model = model
        fprintf(stderr, b"[BatchPredictor] Model set for prediction\n")
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_batch(self, double[:, ::1] X):
        """
        Batch prediction with caching
        
        Parameters
        ----------
        X : ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        ndarray
            Prediction scores
        """
        cdef int n_samples = X.shape[0]
        cdef int n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        cdef int i, batch_idx, start, end
        cdef double[:] y_pred = np.zeros(n_samples, dtype=DTYPE)
        cdef object batch_preds
        
        fprintf(stderr, b"[BatchPredictor] Starting batch prediction for %d samples\n", n_samples)
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, n_samples)
            
            batch_X = np.asarray(X[start:end, :])
            dtest = xgb.DMatrix(batch_X)
            batch_preds = self.model.predict(dtest)
            
            for i in range(end - start):
                y_pred[start + i] = batch_preds[i]
        
        fprintf(stderr, b"[BatchPredictor] Batch prediction completed\n")
        return np.asarray(y_pred)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_with_ranking(self, double[:, ::1] X, int[:] group_ids, int top_k=10):
        """
        Predict and rank within groups
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        group_ids : ndarray
            Group IDs for each sample
        top_k : int
            Top K results per group
            
        Returns
        -------
        dict
            Ranking results by group
        """
        cdef int n_samples = X.shape[0]
        cdef int n_groups = len(np.unique(group_ids))
        cdef int i, j, k
        cdef double[:] y_pred = self.predict_batch(X)
        cdef dict rankings = {}
        cdef list group_scores
        cdef list sorted_indices
        cdef int group_id
        
        fprintf(stderr, b"[BatchPredictor] Ranking predictions (top_k=%d) for %d groups\n", top_k, n_groups)
        
        # Group predictions by group_id
        for i in range(n_samples):
            group_id = group_ids[i]
            if group_id not in rankings:
                rankings[group_id] = []
            rankings[group_id].append((i, y_pred[i]))
        
        # Sort and rank within each group
        for group_id in rankings:
            group_scores = rankings[group_id]
            sorted_scores = sorted(group_scores, key=lambda x: x[1], reverse=True)
            rankings[group_id] = sorted_scores[:top_k]
        
        fprintf(stderr, b"[BatchPredictor] Ranking completed\n")
        return rankings
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_proba(self, double[:, ::1] X):
        """
        Predict probabilities (applies sigmoid to raw scores)
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
            
        Returns
        -------
        ndarray
            Probability scores [0, 1]
        """
        cdef int n_samples = X.shape[0]
        cdef int i
        cdef double[:] y_pred = self.predict_batch(X)
        cdef double[:] y_proba = np.zeros(n_samples, dtype=DTYPE)
        
        fprintf(stderr, b"[BatchPredictor] Computing probability scores\n")
        
        with nogil:
            for i in prange(n_samples, schedule='dynamic'):
                y_proba[i] = 1.0 / (1.0 + np.exp(-y_pred[i]))
        
        return np.asarray(y_proba)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RankingFormatter:
    """
    Format and structure ranking results
    """
    cdef public int top_k
    
    def __cinit__(self, int top_k=10):
        self.top_k = top_k
    
    def format_rankings(self, dict raw_rankings, list item_ids, list group_ids):
        """
        Format raw rankings into structured output
        
        Parameters
        ----------
        raw_rankings : dict
            Raw ranking results by group
        item_ids : list
            Item identifiers
        group_ids : list
            Group identifiers
            
        Returns
        -------
        dict
            Formatted ranking results
        """
        cdef dict formatted_results = {}
        cdef int rank
        cdef list group_rankings
        
        fprintf(stderr, b"[RankingFormatter] Formatting ranking results\n")
        
        for group_id, group_rankings in raw_rankings.items():
            formatted_results[group_id] = []
            for rank, (item_idx, score) in enumerate(group_rankings, 1):
                formatted_results[group_id].append({
                    'rank': rank,
                    'item_id': item_ids[item_idx],
                    'score': score,
                    'group_id': group_id
                })
        
        return formatted_results
    
    def to_dataframe(self, dict formatted_rankings):
        """
        Convert formatted rankings to pandas DataFrame
        
        Parameters
        ----------
        formatted_rankings : dict
            Formatted ranking results
            
        Returns
        -------
        DataFrame
            Ranking results as DataFrame
        """
        import pandas as pd
        
        cdef list rows = []
        
        fprintf(stderr, b"[RankingFormatter] Converting rankings to DataFrame\n")
        
        for group_id, rankings in formatted_rankings.items():
            for ranking in rankings:
                rows.append(ranking)
        
        return pd.DataFrame(rows)