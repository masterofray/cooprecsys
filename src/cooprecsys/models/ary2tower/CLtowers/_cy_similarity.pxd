# cython: language_level=3
# _cy_similarity.pxd
# C-level declarations for fast similarity kernels (paired user-item representations).

from ._cy_types cimport flt

cdef void c_dot_product(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    bint        verbose
) nogil

cdef void c_cosine_similarity(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    flt         eps,
    bint        verbose
) nogil