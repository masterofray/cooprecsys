# _cy_types.pxd
# Shared type declarations for arycolbring Cython modules.

ctypedef float flt


cdef class CSRMatrix:
    """
    Lightweight wrapper around a scipy CSR matrix for nogil access.
    Exposes indices/indptr/data as typed memory views.
    """
    cdef public int[::1] indices
    cdef public int[::1] indptr
    cdef public flt[::1] data

    cdef public int rows
    cdef public int cols
    cdef public int nnz

    cdef int get_row_start(self, int row) nogil noexcept
    cdef int get_row_end(self, int row) nogil noexcept


cdef class FastAryColBring:
    """
    Holds all model state (embeddings, gradients, momentum) for
    the arycolbring collaborative filtering model.
    All fields are typed memory views for direct C-level access.
    """
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