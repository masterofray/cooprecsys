'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: sources = []
# distutils: include_dirs = []

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdio cimport fprintf, stderr
from libc.math cimport sqrt, log, fabs, isnan, isinf
from libc.string cimport memcpy

cdef extern from "math.h":
    double isfinite(double)

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t

cdef class FeatureEngineer:
    """Cython-based feature engineering for learning-to-rank models."""
    
    cdef public int n_features
    cdef public dict feature_stats
    cdef public bint is_fitted
    cdef public double[:] feature_means
    cdef public double[:] feature_stds
    cdef public double[:] feature_mins
    cdef public double[:] feature_maxs
    
    def __cinit__(self):
        self.n_features = 0
        self.is_fitted = False
        self.feature_stats = {}
    
    def fit(self, double[:, ::1] X):
        """Fit feature statistics for normalization."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        self.n_features = n_features
        
        # Allocate memory for statistics
        self.feature_means = np.zeros(n_features, dtype=np.float64)
        self.feature_stds = np.zeros(n_features, dtype=np.float64)
        self.feature_mins = np.zeros(n_features, dtype=np.float64)
        self.feature_maxs = np.zeros(n_features, dtype=np.float64)
        
        cdef int i, j
        cdef double mean_val, std_val, min_val, max_val, val
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                # Calculate mean
                mean_val = 0.0
                for i in range(n_samples):
                    val = X[i, j]
                    if isfinite(val):
                        mean_val += val
                mean_val /= n_samples
                
                # Calculate std, min, max
                std_val = 0.0
                min_val = X[0, j]
                max_val = X[0, j]
                
                for i in range(n_samples):
                    val = X[i, j]
                    if isfinite(val):
                        std_val += (val - mean_val) ** 2
                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                
                std_val = sqrt(std_val / n_samples)
                
                self.feature_means[j] = mean_val
                self.feature_stds[j] = std_val
                self.feature_mins[j] = min_val
                self.feature_maxs[j] = max_val
                
                fprintf(stderr, b"Feature %d - Mean: %.6f, Std: %.6f, Min: %.6f, Max: %.6f\n",
                       j, mean_val, std_val, min_val, max_val)
        
        self.is_fitted = True
    
    def normalize(self, double[:, ::1] X):
        """Z-score normalization with NaN handling."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        
        cdef double[:, ::1] X_normalized = np.array(X, dtype=np.float64, copy=True)
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double val, std_val, mean_val
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                mean_val = self.feature_means[j]
                std_val = self.feature_stds[j]
                
                if std_val > 1e-10:
                    for i in range(n_samples):
                        val = X_normalized[i, j]
                        if isfinite(val):
                            X_normalized[i, j] = (val - mean_val) / std_val
                        else:
                            X_normalized[i, j] = 0.0
                
                fprintf(stderr, b"Normalized feature %d\n", j)
        
        return np.asarray(X_normalized)
    
    def min_max_scale(self, double[:, ::1] X):
        """Min-Max scaling to [0, 1] range."""
        cdef double[:, ::1] X_scaled = np.array(X, dtype=np.float64, copy=True)
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double val, range_val, min_val
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                min_val = self.feature_mins[j]
                range_val = self.feature_maxs[j] - min_val
                
                if range_val > 1e-10:
                    for i in range(n_samples):
                        val = X_scaled[i, j]
                        if isfinite(val):
                            X_scaled[i, j] = (val - min_val) / range_val
                        else:
                            X_scaled[i, j] = 0.0
        
        return np.asarray(X_scaled)


cdef class InteractionFeatures:
    """Generate interaction features efficiently in Cython."""
    
    def compute_pairwise_interactions(self, double[:, ::1] X, list feature_indices=None):
        """Compute pairwise feature interactions."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int n_interactions = 0
        cdef int i, j, k, idx
        cdef double val
        
        if feature_indices is None:
            # Use all features for interactions (limit to prevent explosion)
            if n_features > 10:
                feature_indices = list(range(10))
            else:
                feature_indices = list(range(n_features))
        
        cdef int n_interact_features = len(feature_indices)
        n_interactions = (n_interact_features * (n_interact_features - 1)) // 2
        
        # Allocate output array
        cdef double[:, ::1] interactions = np.zeros(
            (n_samples, n_interactions), dtype=np.float64
        )
        
        idx = 0
        with nogil:
            for i in prange(n_interact_features, schedule='static'):
                for j in range(i + 1, n_interact_features):
                    for k in range(n_samples):
                        interactions[k, idx] = X[k, feature_indices[i]] * X[k, feature_indices[j]]
                    idx += 1
                    fprintf(stderr, b"Computed interaction feature %d\n", idx)
        
        return np.asarray(interactions)
    
    def compute_polynomial_features(self, double[:, ::1] X, int degree=2):
        """Generate polynomial features up to specified degree."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int d, i, j, k
        cdef double val
        cdef int n_poly_features = n_features * degree
        
        cdef double[:, ::1] poly_features = np.zeros(
            (n_samples, n_poly_features), dtype=np.float64
        )
        
        with nogil:
            for d in prange(1, degree + 1, schedule='static'):
                for j in range(n_features):
                    for i in range(n_samples):
                        val = X[i, j]
                        if isfinite(val):
                            poly_features[i, (d-1)*n_features + j] = val ** d
                        else:
                            poly_features[i, (d-1)*n_features + j] = 0.0
                    
                    if d == 1 and j % 5 == 0:
                        fprintf(stderr, b"Computed polynomial degree %d feature %d\n", d, j)
        
        return np.asarray(poly_features)
    
    def compute_statistical_features(self, double[:, ::1] X):
        """Compute statistical features like log, sqrt, reciprocal."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j
        cdef double val
        cdef double[:, ::1] stat_features = np.zeros(
            (n_samples, n_features * 3), dtype=np.float64
        )
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                for i in range(n_samples):
                    val = X[i, j]
                    
                    # Log feature (with small offset to avoid log(0))
                    if val >= 0:
                        stat_features[i, j] = log(val + 1.0)
                    else:
                        stat_features[i, j] = log(fabs(val) + 1.0)
                    
                    # Sqrt feature
                    if val >= 0:
                        stat_features[i, n_features + j] = sqrt(val)
                    else:
                        stat_features[i, n_features + j] = -sqrt(fabs(val))
                    
                    # Reciprocal feature
                    if fabs(val) > 1e-10:
                        stat_features[i, 2*n_features + j] = 1.0 / val
                    else:
                        stat_features[i, 2*n_features + j] = 0.0
                
                fprintf(stderr, b"Computed statistical features for feature %d\n", j)
        
        return np.asarray(stat_features)
    
    def compute_rolling_statistics(self, double[:, ::1] X, int window_size=3):
        """Compute rolling mean and std within groups."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j, k, start, end
        cdef double sum_val, mean_val, std_val
        cdef double[:, ::1] rolling_features = np.zeros(
            (n_samples, n_features * 2), dtype=np.float64
        )
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                for i in range(n_samples):
                    start = max(0, i - window_size)
                    end = min(n_samples, i + 1)
                    
                    # Rolling mean
                    sum_val = 0.0
                    for k in range(start, end):
                        sum_val += X[k, j]
                    rolling_features[i, j] = sum_val / (end - start)
                    
                    # Rolling std
                    mean_val = rolling_features[i, j]
                    std_val = 0.0
                    for k in range(start, end):
                        std_val += (X[k, j] - mean_val) ** 2
                    rolling_features[i, n_features + j] = sqrt(std_val / (end - start))
                
                fprintf(stderr, b"Computed rolling statistics for feature %d\n", j)
        
        return np.asarray(rolling_features)


cdef class FeatureSelection:
    """Feature selection based on variance and correlation."""
    
    def variance_threshold(self, double[:, ::1] X, double threshold=0.01):
        """Select features with variance above threshold."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int j, i
        cdef double mean_val, var_val, val
        cdef list selected_indices = []
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                # Calculate variance
                mean_val = 0.0
                for i in range(n_samples):
                    mean_val += X[i, j]
                mean_val /= n_samples
                
                var_val = 0.0
                for i in range(n_samples):
                    val = X[i, j]
                    var_val += (val - mean_val) ** 2
                var_val /= n_samples
                
                if var_val > threshold:
                    fprintf(stderr, b"Feature %d selected (variance: %.6f)\n", j, var_val)
        
        return selected_indices
    
    def correlation_filter(self, double[:, ::1] X, double corr_threshold=0.95):
        """Remove highly correlated features."""
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int i, j, k
        cdef double mean_i, mean_j, std_i, std_j, corr, cov
        cdef list to_remove = []
        
        # Compute means and stds
        cdef double[:] means = np.zeros(n_features, dtype=np.float64)
        cdef double[:] stds = np.zeros(n_features, dtype=np.float64)
        
        with nogil:
            for j in prange(n_features, schedule='static'):
                means[j] = 0.0
                for i in range(n_samples):
                    means[j] += X[i, j]
                means[j] /= n_samples
                
                stds[j] = 0.0
                for i in range(n_samples):
                    stds[j] += (X[i, j] - means[j]) ** 2
                stds[j] = sqrt(stds[j] / n_samples)
        
        # Check correlations
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue
                
                cov = 0.0
                for k in range(n_samples):
                    cov += (X[k, i] - means[i]) * (X[k, j] - means[j])
                cov /= n_samples
                
                if stds[i] > 1e-10 and stds[j] > 1e-10:
                    corr = cov / (stds[i] * stds[j])
                    if fabs(corr) > corr_threshold:
                        to_remove.append(j)
                        fprintf(stderr, b"Removing feature %d (corr with %d: %.6f)\n", j, i, corr)
        
        return to_remove