# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
# CLproximity/_cy_types.pyx

from libc.stdio cimport fprintf, stderr
import numpy as np
cimport numpy as cnp

# HAPUS BARIS INI: ctypedef float flt (sudah ada di _cy_types.pxd)


cdef class CSRMatrix:
    """
    Thin wrapper over a scipy CSR sparse matrix.
    Provides nogil-safe row slicing via get_row_start / get_row_end.
    """

    def __init__(self,
                 csr_matrix,
                 bint verbose = False,
                ):
        if verbose:
            fprintf(stderr, b"[DEBUG] CSRMatrix.__init__: wrapping sparse matrix\n")

        self.indices = np.ascontiguousarray(csr_matrix.indices, dtype=np.int32)
        self.indptr  = np.ascontiguousarray(csr_matrix.indptr, dtype=np.int32)
        self.data    = np.ascontiguousarray(csr_matrix.data, dtype=np.float32)

        self.rows = <int>csr_matrix.shape[0]
        self.cols = <int>csr_matrix.shape[1]
        self.nnz  = <int>len(self.data)

        if verbose:
            fprintf(stderr,
                    b"[DEBUG] CSRMatrix.__init__: rows=%d cols=%d nnz=%d\n",
                    <int>self.rows, <int>self.cols, <int>self.nnz)

    cdef int get_row_start(self, int row) noexcept nogil:
        return self.indptr[row]

    cdef int get_row_end(self, int row) noexcept nogil:
        return self.indptr[row + 1]


cdef class FastAryColBring:
    """
    Central model-state container.
    """

    def __init__(self,
                 flt[:, ::1] item_features,
                 flt[:, ::1] item_feature_gradients,
                 flt[:, ::1] item_feature_momentum,
                 flt[::1]    item_biases,
                 flt[::1]    item_bias_gradients,
                 flt[::1]    item_bias_momentum,
                 flt[:, ::1] user_features,
                 flt[:, ::1] user_feature_gradients,
                 flt[:, ::1] user_feature_momentum,
                 flt[::1]    user_biases,
                 flt[::1]    user_bias_gradients,
                 flt[::1]    user_bias_momentum,
                 int         no_components,
                 int         adadelta,
                 flt         learning_rate,
                 flt         rho,
                 flt         epsilon,
                 int         max_sampled,
                 bint        verbose = False,
                 ):
        if verbose:
            fprintf(stderr,
                    b"[DEBUG] no_components = %d | adadelta = %d\n",
                    no_components, adadelta)
        self.item_features           = item_features
        self.item_feature_gradients  = item_feature_gradients
        self.item_feature_momentum   = item_feature_momentum
        self.item_biases             = item_biases
        self.item_bias_gradients     = item_bias_gradients
        self.item_bias_momentum      = item_bias_momentum
        self.user_features           = user_features
        self.user_feature_gradients  = user_feature_gradients
        self.user_feature_momentum   = user_feature_momentum
        self.user_biases             = user_biases
        self.user_bias_gradients     = user_bias_gradients
        self.user_bias_momentum      = user_bias_momentum
        self.no_components           = no_components
        self.learning_rate           = learning_rate
        self.rho                     = rho
        self.eps                     = epsilon
        self.item_scale              = 1.0
        self.user_scale              = 1.0
        self.adadelta                = adadelta
        self.max_sampled             = max_sampled
        if verbose:
            fprintf(stderr, b"[DEBUG] FastAryColBring.__init__: done\n")