# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language=c++
"""
Cython-optimized Collaborative Filtering algorithms with nogil support.
Multiple variants: User-Based, Item-Based, SVD, and NMF.
"""

from libc.math cimport sqrt, fabs
from libc.stdio cimport fprintf, stdout, stderr
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import csr_matrix, csc_matrix
import warnings

ctypedef np.float64_t DTYPE_f
ctypedef np.int32_t DTYPE_i

# User-based Collaborative Filtering
cdef class UserBasedCF:
    """
    User-Based Collaborative Filtering using Cosine Similarity.
    Compiled with Cython for nogil optimization.
    """
    
    cdef public np.ndarray user_item_matrix
    cdef public np.ndarray similarity_matrix
    cdef public int n_users
    cdef public int n_items
    cdef public int n_neighbors
    cdef public np.ndarray user_means
    
    def __cinit__(self, np.ndarray[DTYPE_f, ndim=2] user_item_matrix, int n_neighbors=10):
        """Initialize user-based CF."""
        self.user_item_matrix = np.asarray(user_item_matrix, dtype=np.float64)
        self.n_users = user_item_matrix.shape[0]
        self.n_items = user_item_matrix.shape[1]
        self.n_neighbors = min(n_neighbors, self.n_users - 1)
        self.similarity_matrix = np.zeros((self.n_users, self.n_users), dtype=np.float64)
        self.user_means = np.zeros(self.n_users, dtype=np.float64)
        fprintf(stdout, "[UserBasedCF] Initialized with %d users, %d items\n", 
                self.n_users, self.n_items)
    
    def compute_user_means(self):
        """Compute mean rating for each user."""
        cdef int u
        cdef double sum_val
        cdef int count
        cdef int i
        
        with nogil:
            for u in prange(self.n_users, num_threads=8):
                sum_val = 0.0
                count = 0
                for i in range(self.n_items):
                    if self.user_item_matrix[u, i] > 0:
                        sum_val += self.user_item_matrix[u, i]
                        count += 1
                if count > 0:
                    self.user_means[u] = sum_val / count
                else:
                    self.user_means[u] = 0.0
        
        fprintf(stdout, "[UserBasedCF] User means computed\n")
    
    cdef double compute_similarity(self, int u1, int u2) nogil:
        """Compute cosine similarity between two users."""
        cdef double dot_product = 0.0
        cdef double norm1 = 0.0
        cdef double norm2 = 0.0
        cdef double val1, val2
        cdef int i
        
        for i in range(self.n_items):
            val1 = self.user_item_matrix[u1, i] - self.user_means[u1]
            val2 = self.user_item_matrix[u2, i] - self.user_means[u2]
            
            dot_product += val1 * val2
            norm1 += val1 * val1
            norm2 += val2 * val2
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (sqrt(norm1) * sqrt(norm2))
        return 0.0
    
    def compute_similarities(self):
        """Compute similarity matrix between all users."""
        cdef int u1, u2
        
        with nogil:
            for u1 in prange(self.n_users, num_threads=8):
                for u2 in range(u1 + 1, self.n_users):
                    self.similarity_matrix[u1, u2] = self.compute_similarity(u1, u2)
                    self.similarity_matrix[u2, u1] = self.similarity_matrix[u1, u2]
        
        fprintf(stdout, "[UserBasedCF] Similarity matrix computed\n")
    
    def predict_ratings(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs):
        """Predict ratings for given user-item pairs."""
        cdef int n_pairs = user_item_pairs.shape[0]
        cdef np.ndarray[DTYPE_f, ndim=1] predictions = np.zeros(n_pairs, dtype=np.float64)
        cdef int idx, u, i, neighbor_u
        cdef double weighted_sum, sim_sum
        cdef np.ndarray neighbor_indices
        
        with nogil:
            for idx in prange(n_pairs, num_threads=8):
                u = user_item_pairs[idx, 0]
                i = user_item_pairs[idx, 1]
                
                # Find top-k similar users
                weighted_sum = 0.0
                sim_sum = 0.0
                
                for neighbor_u in range(self.n_users):
                    if neighbor_u != u and self.similarity_matrix[u, neighbor_u] > 0 and self.user_item_matrix[neighbor_u, i] > 0:
                        weighted_sum += self.similarity_matrix[u, neighbor_u] * self.user_item_matrix[neighbor_u, i]
                        sim_sum += fabs(self.similarity_matrix[u, neighbor_u])
                
                if sim_sum > 0:
                    predictions[idx] = self.user_means[u] + (weighted_sum / sim_sum)
                else:
                    predictions[idx] = self.user_means[u]
        
        fprintf(stdout, "[UserBasedCF] Predicted %d ratings\n", n_pairs)
        return predictions


# Item-based Collaborative Filtering
cdef class ItemBasedCF:
    """Item-Based Collaborative Filtering."""
    
    cdef public np.ndarray user_item_matrix
    cdef public np.ndarray item_similarity_matrix
    cdef public int n_users
    cdef public int n_items
    cdef public int n_neighbors
    cdef public np.ndarray item_means
    
    def __cinit__(self, np.ndarray[DTYPE_f, ndim=2] user_item_matrix, int n_neighbors=10):
        """Initialize item-based CF."""
        self.user_item_matrix = np.asarray(user_item_matrix, dtype=np.float64)
        self.n_users = user_item_matrix.shape[0]
        self.n_items = user_item_matrix.shape[1]
        self.n_neighbors = min(n_neighbors, self.n_items - 1)
        self.item_similarity_matrix = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        self.item_means = np.zeros(self.n_items, dtype=np.float64)
        fprintf(stdout, "[ItemBasedCF] Initialized with %d users, %d items\n", 
                self.n_users, self.n_items)
    
    def compute_item_means(self):
        """Compute mean rating for each item."""
        cdef int i
        cdef double sum_val
        cdef int count
        cdef int u
        
        with nogil:
            for i in prange(self.n_items, num_threads=8):
                sum_val = 0.0
                count = 0
                for u in range(self.n_users):
                    if self.user_item_matrix[u, i] > 0:
                        sum_val += self.user_item_matrix[u, i]
                        count += 1
                if count > 0:
                    self.item_means[i] = sum_val / count
                else:
                    self.item_means[i] = 0.0
        
        fprintf(stdout, "[ItemBasedCF] Item means computed\n")
    
    cdef double compute_item_similarity(self, int i1, int i2) nogil:
        """Compute cosine similarity between two items."""
        cdef double dot_product = 0.0
        cdef double norm1 = 0.0
        cdef double norm2 = 0.0
        cdef double val1, val2
        cdef int u
        
        for u in range(self.n_users):
            val1 = self.user_item_matrix[u, i1] - self.item_means[i1]
            val2 = self.user_item_matrix[u, i2] - self.item_means[i2]
            
            dot_product += val1 * val2
            norm1 += val1 * val1
            norm2 += val2 * val2
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (sqrt(norm1) * sqrt(norm2))
        return 0.0
    
    def compute_similarities(self):
        """Compute similarity matrix between all items."""
        cdef int i1, i2
        
        with nogil:
            for i1 in prange(self.n_items, num_threads=8):
                for i2 in range(i1 + 1, self.n_items):
                    self.item_similarity_matrix[i1, i2] = self.compute_item_similarity(i1, i2)
                    self.item_similarity_matrix[i2, i1] = self.item_similarity_matrix[i1, i2]
        
        fprintf(stdout, "[ItemBasedCF] Item similarity matrix computed\n")
    
    def predict_ratings(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs):
        """Predict ratings for given user-item pairs."""
        cdef int n_pairs = user_item_pairs.shape[0]
        cdef np.ndarray[DTYPE_f, ndim=1] predictions = np.zeros(n_pairs, dtype=np.float64)
        cdef int idx, u, i, neighbor_i
        cdef double weighted_sum, sim_sum
        
        with nogil:
            for idx in prange(n_pairs, num_threads=8):
                u = user_item_pairs[idx, 0]
                i = user_item_pairs[idx, 1]
                
                weighted_sum = 0.0
                sim_sum = 0.0
                
                for neighbor_i in range(self.n_items):
                    if neighbor_i != i and self.item_similarity_matrix[i, neighbor_i] > 0 and self.user_item_matrix[u, neighbor_i] > 0:
                        weighted_sum += self.item_similarity_matrix[i, neighbor_i] * self.user_item_matrix[u, neighbor_i]
                        sim_sum += fabs(self.item_similarity_matrix[i, neighbor_i])
                
                if sim_sum > 0:
                    predictions[idx] = self.item_means[i] + (weighted_sum / sim_sum)
                else:
                    predictions[idx] = self.item_means[i]
        
        fprintf(stdout, "[ItemBasedCF] Predicted %d ratings\n", n_pairs)
        return predictions


# Matrix Factorization (SVD-based)
cdef class SVDBasedCF:
    """SVD-based Matrix Factorization Collaborative Filtering."""
    
    cdef public np.ndarray user_factors
    cdef public np.ndarray item_factors
    cdef public int n_users
    cdef public int n_items
    cdef public int latent_dim
    cdef public double learning_rate
    cdef public double regularization
    
    def __cinit__(self, int n_users, int n_items, int latent_dim=50, 
                  double learning_rate=0.01, double regularization=0.01):
        """Initialize SVD-based CF."""
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (n_users, latent_dim)).astype(np.float64)
        self.item_factors = np.random.normal(0, 0.01, (n_items, latent_dim)).astype(np.float64)
        
        fprintf(stdout, "[SVDBasedCF] Initialized with latent_dim=%d\n", latent_dim)
    
    def fit(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs, 
            np.ndarray[DTYPE_f, ndim=1] ratings, int epochs=10):
        """Fit SVD model using SGD."""
        cdef int epoch, idx, u, i
        cdef double pred, error, grad_u, grad_i
        cdef int k
        cdef int n_pairs = user_item_pairs.shape[0]
        
        for epoch in range(epochs):
            with nogil:
                for idx in prange(n_pairs, num_threads=8):
                    u = user_item_pairs[idx, 0]
                    i = user_item_pairs[idx, 1]
                    
                    # Compute prediction
                    pred = 0.0
                    for k in range(self.latent_dim):
                        pred += self.user_factors[u, k] * self.item_factors[i, k]
                    
                    # Compute error
                    error = ratings[idx] - pred
                    
                    # Update factors
                    for k in range(self.latent_dim):
                        grad_u = -error * self.item_factors[i, k] + self.regularization * self.user_factors[u, k]
                        grad_i = -error * self.user_factors[u, k] + self.regularization * self.item_factors[i, k]
                        
                        self.user_factors[u, k] -= self.learning_rate * grad_u
                        self.item_factors[i, k] -= self.learning_rate * grad_i
            
            fprintf(stdout, "[SVDBasedCF] Epoch %d/%d completed\n", epoch + 1, epochs)
    
    def predict_ratings(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs):
        """Predict ratings for given user-item pairs."""
        cdef int n_pairs = user_item_pairs.shape[0]
        cdef np.ndarray[DTYPE_f, ndim=1] predictions = np.zeros(n_pairs, dtype=np.float64)
        cdef int idx, u, i, k
        
        with nogil:
            for idx in prange(n_pairs, num_threads=8):
                u = user_item_pairs[idx, 0]
                i = user_item_pairs[idx, 1]
                
                predictions[idx] = 0.0
                for k in range(self.latent_dim):
                    predictions[idx] += self.user_factors[u, k] * self.item_factors[i, k]
        
        fprintf(stdout, "[SVDBasedCF] Predicted %d ratings\n", n_pairs)
        return predictions


# NMF-based Collaborative Filtering
cdef class NMFBasedCF:
    """Non-negative Matrix Factorization Collaborative Filtering."""
    
    cdef public np.ndarray user_factors
    cdef public np.ndarray item_factors
    cdef public int n_users
    cdef public int n_items
    cdef public int latent_dim
    cdef public double regularization
    
    def __cinit__(self, int n_users, int n_items, int latent_dim=50, 
                  double regularization=0.01):
        """Initialize NMF-based CF."""
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.regularization = regularization
        
        # Initialize factors with non-negative values
        np.random.seed(42)
        self.user_factors = np.abs(np.random.normal(0.5, 0.1, (n_users, latent_dim))).astype(np.float64)
        self.item_factors = np.abs(np.random.normal(0.5, 0.1, (n_items, latent_dim))).astype(np.float64)
        
        fprintf(stdout, "[NMFBasedCF] Initialized with latent_dim=%d\n", latent_dim)
    
    def fit(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs, 
            np.ndarray[DTYPE_f, ndim=1] ratings, int epochs=10):
        """Fit NMF model using multiplicative update rules."""
        cdef int epoch, idx, u, i, k
        cdef double pred, error, update_factor
        cdef int n_pairs = user_item_pairs.shape[0]
        
        for epoch in range(epochs):
            with nogil:
                for idx in prange(n_pairs, num_threads=8):
                    u = user_item_pairs[idx, 0]
                    i = user_item_pairs[idx, 1]
                    
                    # Compute prediction
                    pred = 0.0
                    for k in range(self.latent_dim):
                        pred += self.user_factors[u, k] * self.item_factors[i, k]
                    
                    # Avoid division by zero
                    if pred < 1e-9:
                        pred = 1e-9
                    
                    # Multiplicative update
                    error = ratings[idx] / pred
                    
                    for k in range(self.latent_dim):
                        self.user_factors[u, k] *= error * self.item_factors[i, k]
                        self.item_factors[i, k] *= error * self.user_factors[u, k]
            
            fprintf(stdout, "[NMFBasedCF] Epoch %d/%d completed\n", epoch + 1, epochs)
    
    def predict_ratings(self, np.ndarray[DTYPE_i, ndim=2] user_item_pairs):
        """Predict ratings for given user-item pairs."""
        cdef int n_pairs = user_item_pairs.shape[0]
        cdef np.ndarray[DTYPE_f, ndim=1] predictions = np.zeros(n_pairs, dtype=np.float64)
        cdef int idx, u, i, k
        
        with nogil:
            for idx in prange(n_pairs, num_threads=8):
                u = user_item_pairs[idx, 0]
                i = user_item_pairs[idx, 1]
                
                predictions[idx] = 0.0
                for k in range(self.latent_dim):
                    predictions[idx] += self.user_factors[u, k] * self.item_factors[i, k]
        
        fprintf(stdout, "[NMFBasedCF] Predicted %d ratings\n", n_pairs)
        return predictions