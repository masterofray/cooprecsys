# cython: language_level=3
# _cy_forward.pxd
# C-level declarations for fast inter-module forward pass.

from ._cy_types cimport flt

cdef void c_tower_forward(
    int[::1]    ids,
    flt[:, ::1] embeddings,
    flt[:, ::1] w1,
    flt[::1]    b1,
    flt[:, ::1] w2,
    flt[::1]    b2,
    flt[:, ::1] hidden_out,
    flt[:, ::1] tower_out,
    int         num_threads,
    bint        verbose
) noexcept nogil