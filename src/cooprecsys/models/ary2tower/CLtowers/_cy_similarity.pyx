#!python
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_similarity.pyx
# Similarity kernels between paired user-tower / item-tower outputs.

from libc.stdio  cimport fprintf, stderr
from libc.math   cimport sqrt
from cython.parallel cimport prange

from cooprecsys.models.ary2tower.CLtowers._cy_types cimport flt


# ------------------------------------------------------------------
# C-level Kernels (Pure C, nogil-safe)
# ------------------------------------------------------------------

cdef void c_dot_product(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    bint        verbose
) noexcept nogil:
    """
    Pure C-level Dot Product implementation.
    scores[i] = user_out[i] . item_out[i]
    """
    cdef int n = user_out.shape[0]
    cdef int dim = user_out.shape[1]
    cdef int i, k
    cdef flt acc

    if verbose:
        fprintf(stderr, b"[DEBUG] c_dot_product: batch=%d dim=%d num_threads=%d\n", n, dim, num_threads)

    if n == 0:
        return

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        acc = 0.0
        for k in range(dim):
            acc = acc + user_out[i, k] * item_out[i, k]
        scores[i] = acc


cdef void c_cosine_similarity(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    flt         eps,
    bint        verbose
) noexcept nogil:
    """
    Pure C-level Cosine Similarity implementation.
    scores[i] = (user_out[i] . item_out[i]) / (||user_out[i]|| * ||item_out[i]|| + eps)
    """
    cdef int n = user_out.shape[0]
    cdef int dim = user_out.shape[1]
    cdef int i, k
    cdef flt dot, norm_u, norm_i

    if verbose:
        fprintf(stderr, b"[DEBUG] c_cosine_similarity: batch=%d dim=%d num_threads=%d\n", n, dim, num_threads)

    if n == 0:
        return

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        dot = 0.0
        norm_u = 0.0
        norm_i = 0.0
        for k in range(dim):
            dot = dot + user_out[i, k] * item_out[i, k]
            norm_u = norm_u + user_out[i, k] * user_out[i, k]
            norm_i = norm_i + item_out[i, k] * item_out[i, k]
        scores[i] = dot / (sqrt(norm_u) * sqrt(norm_i) + eps)


# ------------------------------------------------------------------
# Python-facing Wrappers (with shape assertions)
# ------------------------------------------------------------------

def dot_product(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    bint        verbose = False,
):
    """
    Python wrapper for dot_product with guardrail shape checks.
    """
    cdef int n = user_out.shape[0]
    
    if item_out.shape[0] != n:
        raise ValueError(f"Batch size mismatch: user_out ({n}) != item_out ({item_out.shape[0]})")
    if item_out.shape[1] != user_out.shape[1]:
        raise ValueError(f"Dimension mismatch: user_out ({user_out.shape[1]}) != item_out ({item_out.shape[1]})")
    if scores.shape[0] != n:
        raise ValueError(f"Scores output buffer size ({scores.shape[0]}) != batch size ({n})")

    c_dot_product(user_out, item_out, scores, num_threads, verbose)


def cosine_similarity(
    flt[:, ::1] user_out,
    flt[:, ::1] item_out,
    flt[::1]    scores,
    int         num_threads,
    flt         eps = 1e-8,
    bint        verbose = False,
):
    """
    Python wrapper for cosine_similarity with guardrail shape checks.
    """
    cdef int n = user_out.shape[0]
    
    if item_out.shape[0] != n:
        raise ValueError(f"Batch size mismatch: user_out ({n}) != item_out ({item_out.shape[0]})")
    if item_out.shape[1] != user_out.shape[1]:
        raise ValueError(f"Dimension mismatch: user_out ({user_out.shape[1]}) != item_out ({item_out.shape[1]})")
    if scores.shape[0] != n:
        raise ValueError(f"Scores output buffer size ({scores.shape[0]}) != batch size ({n})")

    c_cosine_similarity(user_out, item_out, scores, num_threads, eps, verbose)