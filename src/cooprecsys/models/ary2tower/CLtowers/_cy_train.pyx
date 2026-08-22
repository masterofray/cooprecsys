#!python
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_train.pyx
# One epoch of pairwise (BPR-style) two-tower training with Hogwild!-style updates.

import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio  cimport fprintf, stderr
from libc.math   cimport exp
from cython.parallel cimport prange, threadid

from cooprecsys.models.ary2tower.CLtowers._cy_types cimport TwoTowerModel, flt


# ------------------------------------------------------------------
# Inline C Utilities & Math Kernels
# ------------------------------------------------------------------

cdef inline flt c_sigmoid(flt x) noexcept nogil:
    if x > 0.0:
        return <flt>(1.0 / (1.0 + exp(-x)))
    else:
        return <flt>(exp(x) / (1.0 + exp(x)))


cdef inline unsigned int c_rand_r(unsigned int *seed) noexcept nogil:
    """Minimal thread-local xorshift PRNG for nogil sampling."""
    seed[0] ^= seed[0] << 13
    seed[0] ^= seed[0] >> 17
    seed[0] ^= seed[0] << 5
    return seed[0]


cdef inline flt c_dot(flt* a, flt* b, int dim) noexcept nogil:
    cdef int k
    cdef flt acc = 0.0
    for k in range(dim):
        acc = acc + a[k] * b[k]
    return acc


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
    """Forward pass for ONE entity ID."""
    cdef int j, k
    cdef flt acc

    for j in range(hidden_dim):
        acc = b1[j]
        for k in range(embedding_dim):
            acc = acc + embeddings[entity_id, k] * w1[k, j]
        hidden_scratch[j] = acc if acc > 0.0 else 0.0

    for j in range(output_dim):
        acc = b2[j]
        for k in range(hidden_dim):
            acc = acc + hidden_scratch[k] * w2[k, j]
        out_scratch[j] = acc


cdef inline void tower_backward_update(
    flt[:, ::1] embeddings,
    flt[:, ::1] w1, flt[::1] b1,
    flt[:, ::1] w2, flt[::1] b2,
    flt[:, ::1] embeddings_momentum,
    flt[:, ::1] w1_momentum, flt[::1] b1_momentum,
    flt[:, ::1] w2_momentum, flt[::1] b2_momentum,
    int         entity_id,
    flt*        hidden,
    flt*        d_out,
    int         embedding_dim,
    int         hidden_dim,
    int         output_dim,
    double      learning_rate,
    double      momentum_coef,
    flt*        d_hidden_scratch
) noexcept nogil:
    """Backprop & SGD+Momentum update in-place for ONE entity."""
    cdef int j, k, m
    cdef flt grad, velocity, d_hidden_pre

    for k in range(hidden_dim):
        d_hidden_scratch[k] = 0.0
    for j in range(output_dim):
        grad = d_out[j]
        for k in range(hidden_dim):
            d_hidden_scratch[k] = d_hidden_scratch[k] + grad * w2[k, j]

    # --- Dense 2 update ---
    for j in range(output_dim):
        grad = d_out[j]
        for k in range(hidden_dim):
            velocity = <flt>(momentum_coef * w2_momentum[k, j] - learning_rate * grad * hidden[k])
            w2_momentum[k, j] = velocity
            w2[k, j] = w2[k, j] + velocity

        velocity = <flt>(momentum_coef * b2_momentum[j] - learning_rate * grad)
        b2_momentum[j] = velocity
        b2[j] = b2[j] + velocity

    # --- Dense 1 update (ReLU-masked) ---
    for k in range(hidden_dim):
        if hidden[k] <= 0.0:
            continue
        d_hidden_pre = d_hidden_scratch[k]

        for m in range(embedding_dim):
            velocity = <flt>(momentum_coef * w1_momentum[m, k] - learning_rate * d_hidden_pre * embeddings[entity_id, m])
            w1_momentum[m, k] = velocity
            w1[m, k] = w1[m, k] + velocity

        velocity = <flt>(momentum_coef * b1_momentum[k] - learning_rate * d_hidden_pre)
        b1_momentum[k] = velocity
        b1[k] = b1[k] + velocity

    # --- Embedding update ---
    for m in range(embedding_dim):
        grad = 0.0
        for k in range(hidden_dim):
            if hidden[k] > 0.0:
                grad = grad + d_hidden_scratch[k] * w1[m, k]
        velocity = <flt>(momentum_coef * embeddings_momentum[entity_id, m] - learning_rate * grad)
        embeddings_momentum[entity_id, m] = velocity
        embeddings[entity_id, m] = embeddings[entity_id, m] + velocity


# ------------------------------------------------------------------
# Pure C Execution Kernel (nogil)
# ------------------------------------------------------------------

cdef void c_fit_two_tower(
    int[::1]          user_ids,
    int[::1]          positive_item_ids,
    int[::1]          shuffle_indices,
    TwoTowerModel     model,
    double            learning_rate,
    double            momentum_coef,
    int               num_threads,
    unsigned int[::1] random_states,
    flt*              scratch_pool,
    bint              verbose
) noexcept nogil:
    cdef int no_examples = user_ids.shape[0]
    cdef int embedding_dim = model.embedding_dim
    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim
    cdef int n_items = model.item_embeddings.shape[0]

    # Pre-extract C-memoryviews to avoid Python object access inside nogil loop
    cdef flt[:, ::1] u_emb   = model.user_embeddings
    cdef flt[:, ::1] u_w1    = model.user_w1
    cdef flt[::1]    u_b1    = model.user_b1
    cdef flt[:, ::1] u_w2    = model.user_w2
    cdef flt[::1]    u_b2    = model.user_b2
    cdef flt[:, ::1] u_emb_m = model.user_embeddings_momentum
    cdef flt[:, ::1] u_w1_m  = model.user_w1_momentum
    cdef flt[::1]    u_b1_m  = model.user_b1_momentum
    cdef flt[:, ::1] u_w2_m  = model.user_w2_momentum
    cdef flt[::1]    u_b2_m  = model.user_b2_momentum

    cdef flt[:, ::1] i_emb   = model.item_embeddings
    cdef flt[:, ::1] i_w1    = model.item_w1
    cdef flt[::1]    i_b1    = model.item_b1
    cdef flt[:, ::1] i_w2    = model.item_w2
    cdef flt[::1]    i_b2    = model.item_b2
    cdef flt[:, ::1] i_emb_m = model.item_embeddings_momentum
    cdef flt[:, ::1] i_w1_m  = model.item_w1_momentum
    cdef flt[::1]    i_b1_m  = model.item_b1_momentum
    cdef flt[:, ::1] i_w2_m  = model.item_w2_momentum
    cdef flt[::1]    i_b2_m  = model.item_b2_momentum

    cdef int stride = 4 * hidden_dim + 6 * output_dim
    cdef int i, row, uid, pos_iid, neg_iid, j, tid
    cdef flt pos_score, neg_score, g

    cdef flt* t_scratch
    cdef flt* user_hidden
    cdef flt* user_out
    cdef flt* pos_hidden
    cdef flt* pos_out
    cdef flt* neg_hidden
    cdef flt* neg_out
    cdef flt* d_user_out
    cdef flt* d_pos_out
    cdef flt* d_neg_out
    cdef flt* d_hidden_scratch

    if verbose:
        fprintf(stderr,
                b"[DEBUG] c_fit_two_tower: no_examples=%d num_threads=%d "
                b"learning_rate=%.5f momentum=%.3f\n",
                no_examples, num_threads, learning_rate, momentum_coef)

    for i in prange(no_examples, nogil=True, num_threads=num_threads, schedule='dynamic'):
        tid = threadid()
        t_scratch = scratch_pool + tid * stride

        # Partition thread-local contiguous scratch space
        user_hidden      = t_scratch
        user_out         = user_hidden + hidden_dim
        pos_hidden       = user_out + output_dim
        pos_out          = pos_hidden + hidden_dim
        neg_hidden       = pos_out + output_dim
        neg_out          = neg_hidden + hidden_dim
        d_user_out       = neg_out + output_dim
        d_pos_out        = d_user_out + output_dim
        d_neg_out        = d_pos_out + output_dim
        d_hidden_scratch = d_neg_out + output_dim

        row = shuffle_indices[i]
        uid = user_ids[row]
        pos_iid = positive_item_ids[row]
        neg_iid = <int>(c_rand_r(&random_states[tid]) % n_items)

        # Forward passes
        tower_forward_single(u_emb, u_w1, u_b1, u_w2, u_b2,
                             uid, embedding_dim, hidden_dim, output_dim,
                             user_hidden, user_out)
        tower_forward_single(i_emb, i_w1, i_b1, i_w2, i_b2,
                             pos_iid, embedding_dim, hidden_dim, output_dim,
                             pos_hidden, pos_out)
        tower_forward_single(i_emb, i_w1, i_b1, i_w2, i_b2,
                             neg_iid, embedding_dim, hidden_dim, output_dim,
                             neg_hidden, neg_out)

        pos_score = c_dot(user_out, pos_out, output_dim)
        neg_score = c_dot(user_out, neg_out, output_dim)

        g = 1.0 - c_sigmoid(pos_score - neg_score)

        for j in range(output_dim):
            d_user_out[j] = -g * pos_out[j] + g * neg_out[j]
            d_pos_out[j]  = -g * user_out[j]
            d_neg_out[j]  = g * user_out[j]

        # Backward passes & Hogwild! updates
        tower_backward_update(u_emb, u_w1, u_b1, u_w2, u_b2,
                              u_emb_m, u_w1_m, u_b1_m, u_w2_m, u_b2_m,
                              uid, user_hidden, d_user_out,
                              embedding_dim, hidden_dim, output_dim,
                              learning_rate, momentum_coef, d_hidden_scratch)

        tower_backward_update(i_emb, i_w1, i_b1, i_w2, i_b2,
                              i_emb_m, i_w1_m, i_b1_m, i_w2_m, i_b2_m,
                              pos_iid, pos_hidden, d_pos_out,
                              embedding_dim, hidden_dim, output_dim,
                              learning_rate, momentum_coef, d_hidden_scratch)

        tower_backward_update(i_emb, i_w1, i_b1, i_w2, i_b2,
                              i_emb_m, i_w1_m, i_b1_m, i_w2_m, i_b2_m,
                              neg_iid, neg_hidden, d_neg_out,
                              embedding_dim, hidden_dim, output_dim,
                              learning_rate, momentum_coef, d_hidden_scratch)


# ------------------------------------------------------------------
# Python-facing API Wrapper
# ------------------------------------------------------------------

def fit_two_tower(
    int[::1]      user_ids,
    int[::1]      positive_item_ids,
    int[::1]      shuffle_indices,
    TwoTowerModel model,
    double        learning_rate,
    double        momentum_coef,
    int           num_threads,
    random_state,
    bint          verbose = False,
):
    """
    One epoch of pairwise two-tower training (BPR loss).
    Allocates thread-safe contiguous scratch space prior to parallel execution.
    """
    cdef int no_examples = user_ids.shape[0]
    if no_examples == 0:
        if verbose:
            fprintf(stderr, b"[WARN] fit_two_tower: no_examples=0, skipping\n")
        return

    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim
    cdef int stride = 4 * hidden_dim + 6 * output_dim

    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")

    # numpy.default_rng() exposes `integers`, while the legacy RandomState
    # exposes `randint`. Support both so Python-side RNG choice is not a
    # hidden failure point for the compiled training backend.
    if hasattr(random_state, "integers"):
        seed_values = random_state.integers(1, 2 ** 31 - 1, size=num_threads, dtype=np.uint32)
    else:
        seed_values = random_state.randint(1, 2 ** 31 - 1, size=num_threads).astype(np.uint32)

    cdef unsigned int[::1] random_states = seed_values

    # Contiguous scratch pool allocation (Single allocation, 0 thread locks)
    cdef flt* scratch_pool = <flt*>malloc(sizeof(flt) * num_threads * stride)
    if scratch_pool == NULL:
        raise MemoryError("Failed to allocate thread scratch buffer for fit_two_tower")

    try:
        c_fit_two_tower(
            user_ids, positive_item_ids, shuffle_indices, model,
            learning_rate, momentum_coef, num_threads,
            random_states, scratch_pool, verbose
        )
    finally:
        free(scratch_pool)