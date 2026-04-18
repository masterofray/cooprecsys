# _cy_types.pxd
# Shared type declarations for arycolbring Cython modules.
# cimport this file in any .pyx that needs CSRMatrix or FastAryColBring.

ctypedef float flt


cdef class CSRMatrix:
    """
    Lightweight wrapper around a scipy CSR matrix for nogil access.
    Exposes indices/indptr/data as typed memory views.
    """
    cdef int[::1] indices
    cdef int[::1] indptr
    cdef flt[::1] data

    cdef int rows
    cdef int cols
    cdef int nnz

    cdef int get_row_start(self, int row) nogil
    cdef int get_row_end(self, int row) nogil


cdef class FastAryColBring:
    """
    Holds all model state (embeddings, gradients, momentum) for
    the arycolbring collaborative filtering model.
    All fields are typed memory views for direct C-level access.
    """
    cdef flt[:, ::1] item_features
    cdef flt[:, ::1] item_feature_gradients
    cdef flt[:, ::1] item_feature_momentum

    cdef flt[::1] item_biases
    cdef flt[::1] item_bias_gradients
    cdef flt[::1] item_bias_momentum

    cdef flt[:, ::1] user_features
    cdef flt[:, ::1] user_feature_gradients
    cdef flt[:, ::1] user_feature_momentum

    cdef flt[::1] user_biases
    cdef flt[::1] user_bias_gradients
    cdef flt[::1] user_bias_momentum

    cdef int no_components
    cdef int adadelta
    cdef flt learning_rate
    cdef flt rho
    cdef flt eps
    cdef int max_sampled

    cdef double item_scale
    cdef double user_scale
