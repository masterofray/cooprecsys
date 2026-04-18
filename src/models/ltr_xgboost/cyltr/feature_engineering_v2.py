# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Feature Engineering Module - Core Cython Implementation
Handles all feature extraction, transformation, and normalization
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport log, sqrt, exp, fabs
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport malloc, free
import warnings

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

DTYPE = np.float64
ITYPE = np.int32

cdef extern from "math.h":
    double log(double x)
    double sqrt(double x)
    double exp(double x)
    double fabs(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FeatureEngineer:
    """
    High-performance feature engineering using Cython
    """
    cdef public int n_features
    cdef public dict feature_stats
    cdef public bint fitted
    
    def __cinit__(self):
        self.n_features = 0
        self.feature_stats = {}
        self.fitted = False
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, double[:, ::1] X):
        """
        Compute feature statistics for normalization
        
        Parameters
        ----------
        X : ndarray
            Feature matrix (n_samples, n_features)
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double min_val, max_val, mean_val, sum_val, sum_sq
        cdef double val
        
        fprintf(stderr, b"[FeatureEngineer] Fitting feature statistics for %d features\n", n_features)
        
        self.n_features = n_features
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                min_val = X[0, j]
                max_val = X[0, j]
                sum_val = 0.0
                sum_sq = 0.0
                
                for i in range(n_samples):
                    val = X[i, j]
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
                    sum_val += val
                    sum_sq += val * val
                
                mean_val = sum_val / n_samples
                
                with gil:
                    self.feature_stats[j] = {
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': sqrt((sum_sq / n_samples) - (mean_val * mean_val))
                    }
        
        self.fitted = True
        fprintf(stderr, b"[FeatureEngineer] Feature fitting completed\n")
        return self
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normalize_minmax(self, double[:, ::1] X):
        """
        Min-Max normalization: (x - min) / (max - min)
        
        Parameters
        ----------
        X : ndarray
            Feature matrix to normalize
            
        Returns
        -------
        ndarray
            Normalized features
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double min_val, max_val, range_val
        cdef double[:, ::1] X_norm = np.zeros((n_samples, n_features), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Applying Min-Max normalization\n")
        
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before normalization")
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                min_val = self.feature_stats[j]['min']
                max_val = self.feature_stats[j]['max']
                range_val = max_val - min_val
                
                if fabs(range_val) < 1e-10:
                    range_val = 1.0
                
                for i in range(n_samples):
                    X_norm[i, j] = (X[i, j] - min_val) / range_val
        
        return np.asarray(X_norm)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normalize_zscore(self, double[:, ::1] X):
        """
        Z-score normalization: (x - mean) / std
        
        Parameters
        ----------
        X : ndarray
            Feature matrix to normalize
            
        Returns
        -------
        ndarray
            Normalized features
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double mean_val, std_val
        cdef double[:, ::1] X_norm = np.zeros((n_samples, n_features), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Applying Z-score normalization\n")
        
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before normalization")
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                mean_val = self.feature_stats[j]['mean']
                std_val = self.feature_stats[j]['std']
                
                if fabs(std_val) < 1e-10:
                    std_val = 1.0
                
                for i in range(n_samples):
                    X_norm[i, j] = (X[i, j] - mean_val) / std_val
        
        return np.asarray(X_norm)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_interaction_features(self, double[:, ::1] X, list feature_pairs):
        """
        Create interaction features from feature pairs
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        feature_pairs : list
            List of tuples (i, j) for feature interactions
            
        Returns
        -------
        ndarray
            Interaction features
        """
        cdef int n_samples = X.shape[0]
        cdef int n_pairs = len(feature_pairs)
        cdef int i, p, idx_i, idx_j
        cdef double[:, ::1] X_inter = np.zeros((n_samples, n_pairs), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Creating %d interaction features\n", n_pairs)
        
        with nogil:
            for p in prange(n_pairs, schedule='dynamic'):
                idx_i, idx_j = feature_pairs[p]
                for i in range(n_samples):
                    X_inter[i, p] = X[i, idx_i] * X[i, idx_j]
        
        fprintf(stderr, b"[FeatureEngineer] Interaction features completed\n")
        return np.asarray(X_inter)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_polynomial_features(self, double[:, ::1] X, int degree=2):
        """
        Create polynomial features
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        degree : int
            Polynomial degree
            
        Returns
        -------
        ndarray
            Polynomial features
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int n_poly_features = n_features * degree
        cdef int i, j, d
        cdef double[:, ::1] X_poly = np.zeros((n_samples, n_poly_features), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Creating polynomial features (degree=%d)\n", degree)
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                for i in range(n_samples):
                    for d in range(degree):
                        X_poly[i, j * degree + d] = X[i, j] ** (d + 1)
        
        return np.asarray(X_poly)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_log_features(self, double[:, ::1] X, double epsilon=1e-10):
        """
        Create log-transformed features
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        epsilon : double
            Small constant to avoid log(0)
            
        Returns
        -------
        ndarray
            Log-transformed features
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double val
        cdef double[:, ::1] X_log = np.zeros((n_samples, n_features), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Creating log-transformed features\n")
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                for i in range(n_samples):
                    val = X[i, j]
                    if val > 0:
                        X_log[i, j] = log(val + epsilon)
                    else:
                        X_log[i, j] = log(epsilon)
        
        return np.asarray(X_log)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def create_statistical_features(self, double[:, ::1] X, int window_size=5):
        """
        Create statistical features (rolling statistics)
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        window_size : int
            Window size for rolling statistics
            
        Returns
        -------
        ndarray
            Statistical features (mean, std, min, max)
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int stat_features = n_features * 4  # mean, std, min, max
        cdef int i, j, stat_idx, start, end, k
        cdef double sum_val, sum_sq, min_val, max_val, val
        cdef double mean_val, std_val
        cdef double[:, ::1] X_stat = np.zeros((n_samples, stat_features), dtype=DTYPE)
        
        fprintf(stderr, b"[FeatureEngineer] Creating statistical features (window=%d)\n", window_size)
        
        with nogil:
            for i in prange(n_samples, schedule='dynamic'):
                for j in range(n_features):
                    start = max(0, i - window_size // 2)
                    end = min(n_samples, i + window_size // 2 + 1)
                    
                    sum_val = 0.0
                    sum_sq = 0.0
                    min_val = X[start, j]
                    max_val = X[start, j]
                    
                    for k in range(start, end):
                        val = X[k, j]
                        sum_val += val
                        sum_sq += val * val
                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                    
                    stat_idx = j * 4
                    mean_val = sum_val / (end - start)
                    std_val = sqrt((sum_sq / (end - start)) - (mean_val * mean_val))
                    
                    X_stat[i, stat_idx] = mean_val
                    X_stat[i, stat_idx + 1] = std_val
                    X_stat[i, stat_idx + 2] = min_val
                    X_stat[i, stat_idx + 3] = max_val
        
        return np.asarray(X_stat)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def handle_missing_values(self, double[:, ::1] X, str strategy='mean'):
        """
        Handle missing values (NaN)
        
        Parameters
        ----------
        X : ndarray
            Feature matrix with potential NaN values
        strategy : str
            'mean', 'median', or 'zero'
            
        Returns
        -------
        ndarray
            Feature matrix with imputed values
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double[:, ::1] X_filled = np.array(X, dtype=DTYPE)
        cdef double fill_val
        
        fprintf(stderr, b"[FeatureEngineer] Handling missing values with strategy: %s\n", strategy.encode())
        
        with nogil:
            for j in prange(n_features, schedule='dynamic'):
                if strategy == 'mean':
                    fill_val = self.feature_stats[j]['mean']
                elif strategy == 'zero':
                    fill_val = 0.0
                else:
                    fill_val = 0.0
                
                for i in range(n_samples):
                    # Check for NaN
                    if X_filled[i, j] != X_filled[i, j]:  # NaN check
                        X_filled[i, j] = fill_val
        
        return np.asarray(X_filled)