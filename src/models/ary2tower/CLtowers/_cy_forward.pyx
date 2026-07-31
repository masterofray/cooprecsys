#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_forward.pyx
# Two-tower forward-pass kernel: Embedding -> Dense -> ReLU -> Dense.
# Parallelised over the batch dimension with OpenMP prange -- each
# sample's forward pass is independent (no shared-write hazards), so
# this is embarrassingly parallel, unlike the update kernel in
# _cy_update.pyx.

from libc.stdio cimport fprintf, stderr
from cython.parallel cimport prange

from ._cy_types cimport flt


def tower_forward(int[::1]    ids,
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
    Run one tower's forward pass for a batch of ids:

        embedding = embeddings[id]                         # (embedding_dim,)
        hidden    = ReLU(embedding @ w1 + b1)                # (hidden_dim,)
        output    = hidden @ w2 + b2                         # (output_dim,)

    This same kernel serves BOTH towers -- call it once with
    model.user_embeddings/user_w1/... and once with
    model.item_embeddings/item_w1/... .

    Parameters
    ----------
    ids         : int32 array, length N -- the batch of user (or item) ids.
    embeddings  : (n_entities, embedding_dim) float32
    w1, b1      : first dense layer weights/bias, (embedding_dim, hidden_dim) / (hidden_dim,)
    w2, b2      : second dense layer weights/bias, (hidden_dim, output_dim) / (output_dim,)
    hidden_out  : (N, hidden_dim) float32, OUTPUT -- post-ReLU hidden activations.
                  Written here (not just used as scratch) because the
                  backward pass needs them for the ReLU derivative.
    tower_out   : (N, output_dim) float32, OUTPUT -- final tower representation.
    num_threads : OpenMP thread count.
    verbose     : if True, print batch-size/shape diagnostics to stderr
                  (Cython has no logger access inside nogil code, so
                  this follows the same fprintf(stderr, ...) convention
                  as arycolbring/CLproximity -- not a Python callback,
                  which can't be safely invoked without the GIL).
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
        fprintf(stderr, b"[WARN] tower_forward: empty batch, nothing to do\n")
        return

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        entity_id = ids[i]

        # --- Dense 1 + ReLU ---
        for j in range(hidden_dim):
            acc = b1[j]
            for k in range(embedding_dim):
                acc = acc + embeddings[entity_id, k] * w1[k, j]
            hidden_out[i, j] = acc if acc > 0 else 0.0

        # --- Dense 2 ---
        for j in range(output_dim):
            acc = b2[j]
            for k in range(hidden_dim):
                acc = acc + hidden_out[i, k] * w2[k, j]
            tower_out[i, j] = acc

    if verbose:
        fprintf(stderr, b"[DEBUG] tower_forward: batch complete\n")
