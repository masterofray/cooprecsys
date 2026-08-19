# _cy_types.pxd
# cython: language_level=3

ctypedef float flt

cdef class CSRMatrix:
    cdef public int[::1] indices
    cdef public int[::1] indptr
    cdef public flt[::1] data

    cdef public int rows
    cdef public int cols
    cdef public int nnz

    cdef int get_row_start(self, int row) noexcept nogil
    cdef int get_row_end(self, int row) noexcept nogil


cdef class FastAryColBring:
    cdef public flt[:, ::1] item_features
    cdef public flt[:, ::1] item_feature_gradients
    cdef public flt[:, ::1] item_feature_momentum

    cdef public flt[::1] item_biases
    cdef public flt[::1] item_bias_gradients
    cdef public flt[::1] item_bias_momentum

    cdef public flt[:, ::1] user_features
    cdef public flt[:, ::1] user_feature_gradients
    cdef public flt[:, ::1] user_feature_momentum

    cdef public flt[::1] user_biases
    cdef public flt[::1] user_bias_gradients
    cdef public flt[::1] user_bias_momentum

    cdef public int no_components
    cdef public int adadelta
    cdef public flt learning_rate
    cdef public flt rho
    cdef public flt eps
    cdef public int max_sampled

    cdef public double item_scale
    cdef public double user_scale