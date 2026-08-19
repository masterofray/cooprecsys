#!python
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_forward.pyx
# Two-tower forward-pass kernel: Embedding -> Dense -> ReLU -> Dense.

from libc.stdio cimport fprintf, stderr
from cython.parallel cimport prange

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
) noexcept nogil:
    """
    Pure C-level forward pass implementation.
    Dapat dipanggil langsung di dalam blok `nogil` / `prange` dari modul Cython lain (_cy_train.pyx).
    """
    cdef int n = ids.shape[0]
    cdef int embedding_dim = embeddings.shape[1]
    cdef int hidden_dim = w1.shape[1]
    cdef int output_dim = w2.shape[1]
    
    cdef int i, j, k, entity_id
    cdef flt acc

    if verbose:
        fprintf(stderr,
                b"[DEBUG] tower_forward: batch=%d embedding_dim=%d hidden_dim=%d "
                b"output_dim=%d num_threads=%d\n",
                n, embedding_dim, hidden_dim, output_dim, num_threads)

    if n == 0:
        if verbose:
            fprintf(stderr, b"[WARN] tower_forward: empty batch, nothing to do\n")
        return

    # Parallel loop over batch dimension
    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        entity_id = ids[i]

        # --- Layer 1: Dense (Embedding -> Hidden) + ReLU ---
        for j in range(hidden_dim):
            acc = b1[j]
            for k in range(embedding_dim):
                acc = acc + embeddings[entity_id, k] * w1[k, j]
            hidden_out[i, j] = acc if acc > 0.0 else 0.0

        # --- Layer 2: Dense (Hidden -> Output) ---
        for j in range(output_dim):
            acc = b2[j]
            for k in range(hidden_dim):
                acc = acc + hidden_out[i, k] * w2[k, j]
            tower_out[i, j] = acc

    if verbose:
        fprintf(stderr, b"[DEBUG] tower_forward: batch complete\n")


def tower_forward(
    int[::1]    ids,
    flt[:, ::1] embeddings,
    flt[:, ::1] w1,
    flt[::1]    b1,
    flt[:, ::1] w2,
    flt[::1]    b2,
    flt[:, ::1] hidden_out,
    flt[:, ::1] tower_out,
    int         num_threads,
    bint        verbose = False,
):
    """
    Python-facing wrapper function.
    Menerima panggilan dari Python runtime, lalu melemparkan eksekusi ke C-kernel `c_tower_forward`.
    """
    c_tower_forward(
        ids, embeddings, w1, b1, w2, b2,
        hidden_out, tower_out, num_threads, verbose
    )