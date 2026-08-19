#!python
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_predict.pyx
# Parallel inference kernels for two-tower architectures.
# Supports both paired (user_i, item_i) scoring and batch user vs item-catalog scoring.

from libc.stdlib cimport malloc, free
from libc.stdio  cimport fprintf, stderr
from cython.parallel cimport prange, threadid

from ._cy_types cimport TwoTowerModel, flt


# ------------------------------------------------------------------
# Inline Helper Functions (nogil)
# ------------------------------------------------------------------

cdef inline void tower_forward_single(
    flt[:, ::1] embeddings,
    flt[:, ::1] w1,
    flt[::1]    b1,
    flt[:, ::1] w2,
    flt[::1]    b2,
    int         entity_id,
    int         embedding_dim,
    int         hidden_dim,
    int         output_dim,
    flt*        hidden_scratch,
    flt*        out_scratch
) noexcept nogil:
    """Forward pass for ONE entity ID: Embedding -> Dense1 -> ReLU -> Dense2."""
    cdef int j, k
    cdef flt acc

    # --- Layer 1: Dense + ReLU ---
    for j in range(hidden_dim):
        acc = b1[j]
        for k in range(embedding_dim):
            acc = acc + embeddings[entity_id, k] * w1[k, j]
        hidden_scratch[j] = acc if acc > 0.0 else 0.0

    # --- Layer 2: Dense ---
    for j in range(output_dim):
        acc = b2[j]
        for k in range(hidden_dim):
            acc = acc + hidden_scratch[k] * w2[k, j]
        out_scratch[j] = acc


cdef inline flt c_dot(flt* a, flt* b, int dim) noexcept nogil:
    """Vector dot product."""
    cdef int k
    cdef flt acc = 0.0
    for k in range(dim):
        acc = acc + a[k] * b[k]
    return acc


# ------------------------------------------------------------------
# Pure C Execution Kernels (nogil)
# ------------------------------------------------------------------

cdef void c_predict_pairs(
    int[::1]      user_ids,
    int[::1]      item_ids,
    flt[::1]      scores,
    TwoTowerModel model,
    int           num_threads,
    flt*          scratch_pool,
    bint          verbose
) noexcept nogil:
    """
    Computes predicted affinity scores for explicit (user_id[i], item_id[i]) pairs on the fly.
    """
    cdef int n_pairs = user_ids.shape[0]
    cdef int embedding_dim = model.embedding_dim
    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim

    # Pre-extract C-memoryviews to prevent Python object access in GIL-free loop
    cdef flt[:, ::1] u_emb = model.user_embeddings
    cdef flt[:, ::1] u_w1  = model.user_w1
    cdef flt[::1]    u_b1  = model.user_b1
    cdef flt[:, ::1] u_w2  = model.user_w2
    cdef flt[::1]    u_b2  = model.user_b2

    cdef flt[:, ::1] i_emb = model.item_embeddings
    cdef flt[:, ::1] i_w1  = model.item_w1
    cdef flt[::1]    i_b1  = model.item_b1
    cdef flt[:, ::1] i_w2  = model.item_w2
    cdef flt[::1]    i_b2  = model.item_b2

    cdef int stride = 2 * hidden_dim + 2 * output_dim
    cdef int i, uid, iid, tid

    cdef flt* t_scratch
    cdef flt* user_hidden
    cdef flt* user_out
    cdef flt* item_hidden
    cdef flt* item_out

    if verbose:
        fprintf(stderr, b"[DEBUG] c_predict_pairs: n_pairs=%d num_threads=%d\n", n_pairs, num_threads)

    for i in prange(n_pairs, nogil=True, num_threads=num_threads, schedule='static'):
        tid = threadid()
        t_scratch = scratch_pool + tid * stride

        user_hidden = t_scratch
        user_out    = user_hidden + hidden_dim
        item_hidden = user_out + output_dim
        item_out    = item_hidden + hidden_dim

        uid = user_ids[i]
        iid = item_ids[i]

        tower_forward_single(u_emb, u_w1, u_b1, u_w2, u_b2,
                             uid, embedding_dim, hidden_dim, output_dim,
                             user_hidden, user_out)

        tower_forward_single(i_emb, i_w1, i_b1, i_w2, i_b2,
                             iid, embedding_dim, hidden_dim, output_dim,
                             item_hidden, item_out)

        scores[i] = c_dot(user_out, item_out, output_dim)


cdef void c_predict_user_items(
    int[::1]      user_ids,
    flt[:, ::1]   item_outputs,
    flt[:, ::1]   scores_out,
    TwoTowerModel model,
    int           num_threads,
    flt*          scratch_pool,
    bint          verbose
) noexcept nogil:
    """
    Computes user score matrix for target batch of user_ids against pre-computed item tower representations.
    `scores_out` shape: [n_users, n_items]
    """
    cdef int n_users = user_ids.shape[0]
    cdef int n_items = item_outputs.shape[0]
    cdef int embedding_dim = model.embedding_dim
    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim

    cdef flt[:, ::1] u_emb = model.user_embeddings
    cdef flt[:, ::1] u_w1  = model.user_w1
    cdef flt[::1]    u_b1  = model.user_b1
    cdef flt[:, ::1] u_w2  = model.user_w2
    cdef flt[::1]    u_b2  = model.user_b2

    cdef int stride = hidden_dim + output_dim
    cdef int u_idx, i_idx, uid, tid, k
    cdef flt acc

    cdef flt* t_scratch
    cdef flt* user_hidden
    cdef flt* user_out

    if verbose:
        fprintf(stderr, b"[DEBUG] c_predict_user_items: n_users=%d n_items=%d num_threads=%d\n",
                n_users, n_items, num_threads)

    for u_idx in prange(n_users, nogil=True, num_threads=num_threads, schedule='static'):
        tid = threadid()
        t_scratch = scratch_pool + tid * stride

        user_hidden = t_scratch
        user_out    = user_hidden + hidden_dim

        uid = user_ids[u_idx]

        # Compute User Tower representation on-the-fly once per user
        tower_forward_single(u_emb, u_w1, u_b1, u_w2, u_b2,
                             uid, embedding_dim, hidden_dim, output_dim,
                             user_hidden, user_out)

        # Dot product against all pre-computed item tower outputs
        for i_idx in range(n_items):
            acc = 0.0
            for k in range(output_dim):
                acc = acc + user_out[k] * item_outputs[i_idx, k]
            scores_out[u_idx, i_idx] = acc


# ------------------------------------------------------------------
# Python-facing API Wrappers (with shape assertions and safety checks)
# ------------------------------------------------------------------

def predict_pairs(
    int[::1]      user_ids,
    int[::1]      item_ids,
    flt[::1]      scores,
    TwoTowerModel model,
    int           num_threads,
    bint          verbose = False,
):
    """
    Python wrapper to score paired (user_id[i], item_id[i]) array.
    """
    cdef int n_pairs = user_ids.shape[0]
    if n_pairs == 0:
        return

    if item_ids.shape[0] != n_pairs:
        raise ValueError(f"Shape mismatch: user_ids length ({n_pairs}) != item_ids length ({item_ids.shape[0]})")
    if scores.shape[0] != n_pairs:
        raise ValueError(f"Shape mismatch: scores output length ({scores.shape[0]}) != n_pairs ({n_pairs})")

    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim
    cdef int stride = 2 * hidden_dim + 2 * output_dim

    # Contiguous thread-scratch pool allocation
    cdef flt* scratch_pool = <flt*>malloc(sizeof(flt) * num_threads * stride)
    if scratch_pool == NULL:
        raise MemoryError("Failed to allocate thread scratch pool for predict_pairs")

    try:
        c_predict_pairs(user_ids, item_ids, scores, model, num_threads, scratch_pool, verbose)
    finally:
        free(scratch_pool)


def predict_user_items(
    int[::1]      user_ids,
    flt[:, ::1]   item_outputs,
    flt[:, ::1]   scores_out,
    TwoTowerModel model,
    int           num_threads,
    bint          verbose = False,
):
    """
    Python wrapper to compute affinity scores between batch of users and precomputed item tower vectors.
    `scores_out` shape must be (len(user_ids), item_outputs.shape[0]).
    """
    cdef int n_users = user_ids.shape[0]
    cdef int n_items = item_outputs.shape[0]

    if n_users == 0 or n_items == 0:
        return

    if scores_out.shape[0] != n_users or scores_out.shape[1] != n_items:
        raise ValueError(
            f"Scores matrix shape ({scores_out.shape[0]}, {scores_out.shape[1]}) "
            f"must match (n_users={n_users}, n_items={n_items})"
        )
    if item_outputs.shape[1] != model.output_dim:
        raise ValueError(
            f"Item output representation dimension ({item_outputs.shape[1]}) "
            f"does not match model output_dim ({model.output_dim})"
        )

    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim
    cdef int stride = hidden_dim + output_dim

    cdef flt* scratch_pool = <flt*>malloc(sizeof(flt) * num_threads * stride)
    if scratch_pool == NULL:
        raise MemoryError("Failed to allocate thread scratch pool for predict_user_items")

    try:
        c_predict_user_items(user_ids, item_outputs, scores_out, model, num_threads, scratch_pool, verbose)
    finally:
        free(scratch_pool)