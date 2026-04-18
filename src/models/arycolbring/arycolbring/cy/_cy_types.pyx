#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_types.pyx
# Implementation of CSRMatrix and FastAryColBring cdef classes.

from libc.stdio cimport fprintf, stderr


cdef class CSRMatrix:
    """
    Thin wrapper over a scipy CSR sparse matrix.
    Provides nogil-safe row slicing via get_row_start / get_row_end.
    """

    def __init__(self, csr_matrix):
        fprintf(stderr, b"[DEBUG] CSRMatrix.__init__: wrapping sparse matrix\n")

        self.indices = csr_matrix.indices
        self.indptr  = csr_matrix.indptr
        self.data    = csr_matrix.data

        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)

        fprintf(stderr,
                b"[DEBUG] CSRMatrix.__init__: rows=%d cols=%d nnz=%d\n",
                self.rows, self.cols, self.nnz)

    cdef int get_row_start(self, int row) nogil:
        return self.indptr[row]

    cdef int get_row_end(self, int row) nogil:
        return self.indptr[row + 1]


cdef class FastAryColBring:
    """
    Central model-state container.
    All embedding matrices and their optimiser accumulators live here.
    Passed by reference to every Cython kernel so they can mutate
    the arrays in-place without any Python overhead.
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
                 int         max_sampled):

        fprintf(stderr,
                b"[DEBUG] FastAryColBring.__init__: no_components=%d adadelta=%d\n",
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

        self.no_components  = no_components
        self.learning_rate  = learning_rate
        self.rho            = rho
        self.eps            = epsilon
        self.item_scale     = 1.0
        self.user_scale     = 1.0
        self.adadelta       = adadelta
        self.max_sampled    = max_sampled

        fprintf(stderr, b"[DEBUG] FastAryColBring.__init__: done\n")
