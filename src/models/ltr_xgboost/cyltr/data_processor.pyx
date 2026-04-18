'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cython.parallel import prange, parallel
from libc.math cimport sqrt, log, exp, isnan, isinf
from libc.stdio cimport printf, fprintf, FILE, fopen, fclose
from libc.stdlib cimport malloc, free, calloc, realloc
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF
cimport numpy as np
import numpy as np
from numpy cimport ndarray, float64_t, int64_t, int32_t
import cython

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t LONG_t
ctypedef np.int32_t INT_t

cdef extern from "math.h":
    double isnan(double x) nogil

cdef class DataProcessor:
    """High-performance Cython-based data processor for LTR models"""
    
    cdef public int n_threads
    cdef public int batch_size
    cdef public double min_value
    cdef public double max_value
    
    def __init__(self, int n_threads=4, int batch_size=10000):
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.min_value = -1e10
        self.max_value = 1e10
    
    def normalize_features(self, ndarray[DTYPE_t, ndim=2] features):
        """Normalize features using z-score normalization with parallelization"""
        cdef int n_samples = features.shape[0]
        cdef int n_features = features.shape[1]
        cdef ndarray[DTYPE_t, ndim=2] result = np.zeros((n_samples, n_features), dtype=np.float64)
        cdef ndarray[DTYPE_t, ndim=1] means = np.zeros(n_features, dtype=np.float64)
        cdef ndarray[DTYPE_t, ndim=1] stds = np.zeros(n_features, dtype=np.float64)
        cdef int i, j
        cdef DTYPE_t mean_val, std_val, val
        cdef FILE* debug_file
        
        debug_file = fopen("debug_normalize.log", "w")
        fprintf(debug_file, "Starting feature normalization for %d samples, %d features\n", n_samples, n_features)
        
        # Compute means
        with nogil, parallel(num_threads=self.n_threads):
            for j in prange(n_features):
                mean_val = 0.0
                for i in range(n_samples):
                    mean_val += features[i, j]
                means[j] = mean_val / n_samples
        
        # Compute standard deviations
        with nogil, parallel(num_threads=self.n_threads):
            for j in prange(n_features):
                std_val = 0.0
                for i in range(n_samples):
                    std_val += (features[i, j] - means[j]) ** 2
                stds[j] = sqrt(std_val / n_samples)
        
        # Normalize
        with nogil, parallel(num_threads=self.n_threads):
            for j in prange(n_features):
                if stds[j] > 1e-10:
                    for i in range(n_samples):
                        result[i, j] = (features[i, j] - means[j]) / stds[j]
                else:
                    for i in range(n_samples):
                        result[i, j] = 0.0
        
        fprintf(debug_file, "Normalization completed. Mean range: [%.6f, %.6f], Std range: [%.6f, %.6f]\n", 
                np.min(means), np.max(means), np.min(stds), np.max(stds))
        fclose(debug_file)
        
        return result, means, stds
    
    def remove_outliers(self, ndarray[DTYPE_t, ndim=2] features, double threshold=3.0):
        """Remove outliers based on z-score with vectorized operations"""
        cdef int n_samples = features.shape[0]
        cdef int n_features = features.shape[1]
        cdef ndarray[DTYPE_t, ndim=1] z_scores = np.zeros(n_samples, dtype=np.float64)
        cdef ndarray[np.uint8_t, ndim=1] mask = np.ones(n_samples, dtype=np.uint8)
        cdef int i, j, valid_count = 0
        cdef DTYPE_t z_val, mean_val, std_val
        cdef FILE* debug_file
        
        debug_file = fopen("debug_outliers.log", "w")
        fprintf(debug_file, "Starting outlier removal with threshold=%.2f\n", threshold)
        
        # Compute z-scores in parallel
        with nogil, parallel(num_threads=self.n_threads):
            for i in prange(n_samples):
                z_val = 0.0
                for j in range(n_features):
                    mean_val = 0.0
                    std_val = 0.0
                    # Compute mean and std for feature j
                    for k in range(n_samples):
                        mean_val += features[k, j]
                    mean_val /= n_samples
                    for k in range(n_samples):
                        std_val += (features[k, j] - mean_val) ** 2
                    std_val = sqrt(std_val / n_samples)
                    
                    if std_val > 1e-10:
                        z_val += ((features[i, j] - mean_val) / std_val) ** 2
                
                z_scores[i] = sqrt(z_val / n_features)
                if z_scores[i] > threshold:
                    mask[i] = 0
        
        # Count valid samples
        for i in range(n_samples):
            if mask[i]:
                valid_count += 1
        
        fprintf(debug_file, "Outliers detected: %d/%d (%.2f%%)\n", 
                n_samples - valid_count, n_samples, 
                100.0 * (n_samples - valid_count) / n_samples)
        fclose(debug_file)
        
        return mask.astype(bool)
    
    def compute_group_statistics(self, ndarray[LONG_t, ndim=1] group_ids, 
                                 ndarray[DTYPE_t, ndim=1] values):
        """Compute statistics per group efficiently"""
        cdef int n_samples = group_ids.shape[0]
        cdef int max_group_id = np.max(group_ids) + 1
        cdef dict group_stats = {}
        cdef int i
        cdef LONG_t group_id
        cdef list group_values
        
        # Group values by group_id
        for i in range(n_samples):
            group_id = group_ids[i]
            if group_id not in group_stats:
                group_stats[group_id] = []
            group_stats[group_id].append(values[i])
        
        # Compute statistics per group
        cdef dict result = {}
        for group_id, vals in group_stats.items():
            vals_array = np.array(vals, dtype=np.float64)
            result[group_id] = {
                'mean': np.mean(vals_array),
                'std': np.std(vals_array),
                'min': np.min(vals_array),
                'max': np.max(vals_array),
                'count': len(vals)
            }
        
        return result
