#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_similarity.pyx
# Similarity kernels between paired user-tower / item-tower outputs.
# Both are parallelised over the batch dimension with OpenMP prange.

from libc.stdio  cimport fprintf, stderr
from libc.math   cimport sqrt
from cython.parallel cimport prange

from ._cy_types cimport flt


def dot_product(flt[:, ::1] user_out,
                 flt[:, ::1] item_out,
                 flt[::1]    scores,
                 int         num_threads,
                 bint        verbose = False,
                ):
    """scores[i] = user_out[i] . item_out[i], for each row i (paired,
    NOT cross-joined -- user_out and item_out must have the same number
    of rows)."""
    cdef int n = user_out.shape[0]
    cdef int dim = user_out.shape[1]
    cdef int i, k
    cdef flt acc

    if verbose:
        fprintf(stderr, b"[DEBUG] dot_product: batch=%d dim=%d\n", n, dim)

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        acc = 0.0
        for k in range(dim):
            acc = acc + user_out[i, k] * item_out[i, k]
        scores[i] = acc


def cosine_similarity(flt[:, ::1] user_out,
                       flt[:, ::1] item_out,
                       flt[::1]    scores,
                       int         num_threads,
                       flt         eps = 1e-8,
                       bint        verbose = False,
                      ):
    """scores[i] = cosine_similarity(user_out[i], item_out[i])."""
    cdef int n = user_out.shape[0]
    cdef int dim = user_out.shape[1]
    cdef int i, k
    cdef flt dot, norm_u, norm_i

    if verbose:
        fprintf(stderr, b"[DEBUG] cosine_similarity: batch=%d dim=%d\n", n, dim)

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        dot = 0.0
        norm_u = 0.0
        norm_i = 0.0
        for k in range(dim):
            dot = dot + user_out[i, k] * item_out[i, k]
            norm_u = norm_u + user_out[i, k] * user_out[i, k]
            norm_i = norm_i + item_out[i, k] * item_out[i, k]
        scores[i] = dot / (sqrt(norm_u) * sqrt(norm_i) + eps)
