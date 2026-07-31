#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_train.pyx
# One epoch of pairwise (BPR-style) two-tower training.
#
# For each observed (user, positive_item) pair, one negative item is
# sampled uniformly and a sigmoid-weighted pairwise gradient is
# backpropagated through both towers' two dense layers and into the
# embedding tables, with an SGD+momentum update applied directly.
#
# PARALLELISM NOTE (read before touching this file): this repo's own
# existing fit_bpr()/fit_warp() kernels (arycolbring/CLproximity/
# _cy_fit_bpr.pyx) update the shared embedding/weight arrays from
# inside an unlocked `prange` loop -- i.e. they accept the well-known
# Hogwild!-style race condition where two threads may race on the same
# user/item's row within one epoch (a lock is only taken there for the
# rare regularization-overflow rescale, which this simpler kernel does
# not have). This kernel follows that same, already-established
# convention for consistency, rather than adding a stricter (and
# slower) locking scheme this codebase doesn't otherwise use for the
# per-sample update. Hogwild! SGD is a standard, published technique
# for sparse gradient updates -- this is a documented trade-off, not an
# oversight.

import numpy as np
from libc.stdlib     cimport malloc, free
from libc.stdio      cimport fprintf, stderr
from libc.math       cimport exp
from cython.parallel cimport prange, parallel, threadid

from ._cy_types cimport TwoTowerModel, flt


cdef inline flt c_sigmoid(flt x) nogil:
    if x > 0:
        return <flt>(1.0 / (1.0 + exp(-x)))
    else:
        # Numerically stable branch for very negative x.
        return <flt>(exp(x) / (1.0 + exp(x)))


cdef inline unsigned int c_rand_r(unsigned int *seed) nogil:
    """Minimal thread-local xorshift PRNG (avoids depending on libc's
    non-reentrant rand(), and avoids needing the GIL inside prange)."""
    seed[0] ^= seed[0] << 13
    seed[0] ^= seed[0] >> 17
    seed[0] ^= seed[0] << 5
    return seed[0]


cdef inline void tower_forward_single(flt[:, ::1] embeddings,
                                      flt[:, ::1] w1,
                                      flt[::1]    b1,
                                      flt[:, ::1] w2,
                                      flt[::1]    b2,
                                      int         entity_id,
                                      int         embedding_dim,
                                      int         hidden_dim,
                                      int         output_dim,
                                      flt*        hidden_scratch,
                                      flt*        out_scratch,
                                     ) nogil:
    """Forward pass for ONE id: embedding -> Dense(W1,b1) -> ReLU ->
    Dense(W2,b2). Writes into caller-owned scratch buffers (allocated
    once per thread, not per sample -- see fit_two_tower below). Same
    math as _cy_forward.pyx's batched tower_forward(), duplicated here
    because the training loop needs a single-id calling convention."""
    cdef int j, k
    cdef flt acc

    for j in range(hidden_dim):
        acc = b1[j]
        for k in range(embedding_dim):
            acc = acc + embeddings[entity_id, k] * w1[k, j]
        hidden_scratch[j] = acc if acc > 0 else 0.0

    for j in range(output_dim):
        acc = b2[j]
        for k in range(hidden_dim):
            acc = acc + hidden_scratch[k] * w2[k, j]
        out_scratch[j] = acc


cdef inline flt c_dot(flt* a, flt* b, int dim) nogil:
    cdef int k
    cdef flt acc = 0.0
    for k in range(dim):
        acc = acc + a[k] * b[k]
    return acc


cdef inline void tower_backward_update(flt[:, ::1] embeddings,
                                       flt[:, ::1] w1, flt[::1] b1,
                                       flt[:, ::1] w2, flt[::1] b2,
                                       flt[:, ::1] embeddings_momentum,
                                       flt[:, ::1] w1_momentum, flt[::1] b1_momentum,
                                       flt[:, ::1] w2_momentum, flt[::1] b2_momentum,
                                       int         entity_id,
                                       flt*        hidden,       # from tower_forward_single (post-ReLU)
                                       flt*        d_out,        # dLoss/d(tower_out), length output_dim
                                       int         embedding_dim,
                                       int         hidden_dim,
                                       int         output_dim,
                                       double      learning_rate,
                                       double      momentum_coef,
                                       flt*        d_hidden_scratch,   # length hidden_dim, caller-owned
                                      ) nogil:
    """Backprop Dense2 -> ReLU -> Dense1 -> embedding for ONE sample and
    apply an SGD+momentum update in-place (Hogwild!-style, see module
    docstring). `d_out` is the incoming gradient w.r.t. this tower's
    output for this sample (computed by the caller from the pairwise
    loss -- see fit_two_tower). d_hidden_scratch is computed from the
    ORIGINAL (pre-update) W2 first, before W2 itself is mutated, so the
    two passes below must stay in this order."""
    cdef int j, k, m
    cdef flt grad, velocity, d_hidden_pre

    for k in range(hidden_dim):
        d_hidden_scratch[k] = 0.0
    for j in range(output_dim):
        grad = d_out[j]
        for k in range(hidden_dim):
            d_hidden_scratch[k] = d_hidden_scratch[k] + grad * w2[k, j]

    # --- Dense 2 update: dW2[k,j] = hidden[k] * d_out[j], db2[j] = d_out[j] ---
    for j in range(output_dim):
        grad = d_out[j]
        for k in range(hidden_dim):
            velocity = <flt>(momentum_coef * w2_momentum[k, j]
                             - learning_rate * grad * hidden[k])
            w2_momentum[k, j] = velocity
            w2[k, j] = w2[k, j] + velocity

        velocity = <flt>(momentum_coef * b2_momentum[j] - learning_rate * grad)
        b2_momentum[j] = velocity
        b2[j] = b2[j] + velocity

    # --- Dense 1 update (ReLU-masked) ---
    for k in range(hidden_dim):
        if hidden[k] <= 0:
            continue  # ReLU derivative is 0 where the activation was clamped
        d_hidden_pre = d_hidden_scratch[k]

        for m in range(embedding_dim):
            velocity = <flt>(momentum_coef * w1_momentum[m, k]
                             - learning_rate * d_hidden_pre * embeddings[entity_id, m])
            w1_momentum[m, k] = velocity
            w1[m, k] = w1[m, k] + velocity

        velocity = <flt>(momentum_coef * b1_momentum[k] - learning_rate * d_hidden_pre)
        b1_momentum[k] = velocity
        b1[k] = b1[k] + velocity

    # --- Embedding update: d_embedding[m] = sum_k d_hidden_pre[k] * W1[m,k] ---
    # (uses the just-updated W1 -- a standard, accepted approximation
    # for this kind of fused single-pass SGD step, same spirit as
    # updating biases before/after weights elsewhere in this codebase.)
    for m in range(embedding_dim):
        grad = 0.0
        for k in range(hidden_dim):
            if hidden[k] > 0:
                grad = grad + d_hidden_scratch[k] * w1[m, k]
        velocity = <flt>(momentum_coef * embeddings_momentum[entity_id, m]
                         - learning_rate * grad)
        embeddings_momentum[entity_id, m] = velocity
        embeddings[entity_id, m] = embeddings[entity_id, m] + velocity


def fit_two_tower(int[::1]  user_ids,
                   int[::1]  positive_item_ids,
                   int[::1]  shuffle_indices,
                   TwoTowerModel model,
                   double    learning_rate,
                   double    momentum_coef,
                   int       num_threads,
                   random_state,
                   bint      verbose = False,
                  ):
    """
    One epoch of pairwise two-tower training (dot-product similarity,
    BPR-style pairwise sigmoid loss): for each (user, positive_item),
    sample a random negative item and take one SGD+momentum step that
    pulls the positive pair's score above the negative pair's score.

    See the module docstring re: the intentionally-unlocked parallel
    embedding update (Hogwild!-style, matching this repo's existing
    fit_bpr()/fit_warp() convention).
    """
    cdef int no_examples = user_ids.shape[0]
    cdef int embedding_dim = model.embedding_dim
    cdef int hidden_dim = model.hidden_dim
    cdef int output_dim = model.output_dim
    cdef int n_items = model.item_embeddings.shape[0]
    cdef int i, row, uid, pos_iid, neg_iid, j
    cdef flt pos_score, neg_score, g
    cdef unsigned int[::1] random_states

    # Thread-local scratch buffers -- declared here (function scope) so
    # each OpenMP thread gets its own private copy inside the
    # `parallel()` block below, matching the exact convention used in
    # arycolbring/CLproximity/_cy_predict.pyx.
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
                b"[DEBUG] fit_two_tower: no_examples=%d num_threads=%d "
                b"learning_rate=%.5f momentum=%.3f\n",
                no_examples, num_threads, learning_rate, momentum_coef)

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_two_tower: no_examples=0, skipping\n")
        return

    # Per-thread PRNG seeds (must be non-zero for xorshift).
    random_states = (random_state
                     .randint(1, 2 ** 31 - 1, size=num_threads)
                     .astype(np.uint32))

    with nogil, parallel(num_threads=num_threads):
        user_hidden = <flt*>malloc(sizeof(flt) * hidden_dim)
        user_out    = <flt*>malloc(sizeof(flt) * output_dim)
        pos_hidden  = <flt*>malloc(sizeof(flt) * hidden_dim)
        pos_out     = <flt*>malloc(sizeof(flt) * output_dim)
        neg_hidden  = <flt*>malloc(sizeof(flt) * hidden_dim)
        neg_out     = <flt*>malloc(sizeof(flt) * output_dim)
        d_user_out  = <flt*>malloc(sizeof(flt) * output_dim)
        d_pos_out   = <flt*>malloc(sizeof(flt) * output_dim)
        d_neg_out   = <flt*>malloc(sizeof(flt) * output_dim)
        d_hidden_scratch = <flt*>malloc(sizeof(flt) * hidden_dim)

        if (user_hidden == NULL or user_out == NULL or pos_hidden == NULL
                or pos_out == NULL or neg_hidden == NULL or neg_out == NULL
                or d_user_out == NULL or d_pos_out == NULL or d_neg_out == NULL
                or d_hidden_scratch == NULL):
            fprintf(stderr, b"[ERROR] fit_two_tower: malloc failed for thread buffer\n")
        else:
            for i in prange(no_examples, schedule='dynamic'):
                row = shuffle_indices[i]
                uid = user_ids[row]
                pos_iid = positive_item_ids[row]
                # Uniform negative sample over the whole catalogue (a
                # simpler sampler than fit_bpr()'s in_positives()-checked
                # rejection sampling -- occasionally sampling a true
                # positive as the "negative" is a standard, accepted
                # approximation at this scale, and keeps this kernel
                # lock-free with respect to the interactions matrix).
                neg_iid = <int>(c_rand_r(&random_states[threadid()]) % n_items)

                tower_forward_single(model.user_embeddings, model.user_w1, model.user_b1,
                                     model.user_w2, model.user_b2, uid,
                                     embedding_dim, hidden_dim, output_dim,
                                     user_hidden, user_out)
                tower_forward_single(model.item_embeddings, model.item_w1, model.item_b1,
                                     model.item_w2, model.item_b2, pos_iid,
                                     embedding_dim, hidden_dim, output_dim,
                                     pos_hidden, pos_out)
                tower_forward_single(model.item_embeddings, model.item_w1, model.item_b1,
                                     model.item_w2, model.item_b2, neg_iid,
                                     embedding_dim, hidden_dim, output_dim,
                                     neg_hidden, neg_out)

                pos_score = c_dot(user_out, pos_out, output_dim)
                neg_score = c_dot(user_out, neg_out, output_dim)

                # BPR loss: L = -log(sigmoid(pos_score - neg_score)).
                # dL/d(margin) = sigmoid(margin) - 1 (always <= 0: a
                # larger margin means lower loss). `g` below folds that
                # sign into a "pull strength" used directly as the
                # per-output-dim gradient below.
                g = 1.0 - c_sigmoid(pos_score - neg_score)

                for j in range(output_dim):
                    d_user_out[j] = -g * pos_out[j] + g * neg_out[j]
                    d_pos_out[j]  = -g * user_out[j]
                    d_neg_out[j]  = g * user_out[j]

                tower_backward_update(model.user_embeddings, model.user_w1, model.user_b1,
                                      model.user_w2, model.user_b2,
                                      model.user_embeddings_momentum,
                                      model.user_w1_momentum, model.user_b1_momentum,
                                      model.user_w2_momentum, model.user_b2_momentum,
                                      uid, user_hidden, d_user_out,
                                      embedding_dim, hidden_dim, output_dim,
                                      learning_rate, momentum_coef, d_hidden_scratch)
                tower_backward_update(model.item_embeddings, model.item_w1, model.item_b1,
                                      model.item_w2, model.item_b2,
                                      model.item_embeddings_momentum,
                                      model.item_w1_momentum, model.item_b1_momentum,
                                      model.item_w2_momentum, model.item_b2_momentum,
                                      pos_iid, pos_hidden, d_pos_out,
                                      embedding_dim, hidden_dim, output_dim,
                                      learning_rate, momentum_coef, d_hidden_scratch)
                tower_backward_update(model.item_embeddings, model.item_w1, model.item_b1,
                                      model.item_w2, model.item_b2,
                                      model.item_embeddings_momentum,
                                      model.item_w1_momentum, model.item_b1_momentum,
                                      model.item_w2_momentum, model.item_b2_momentum,
                                      neg_iid, neg_hidden, d_neg_out,
                                      embedding_dim, hidden_dim, output_dim,
                                      learning_rate, momentum_coef, d_hidden_scratch)

        free(user_hidden); free(user_out)
        free(pos_hidden);  free(pos_out)
        free(neg_hidden);  free(neg_out)
        free(d_user_out);  free(d_pos_out); free(d_neg_out)
        free(d_hidden_scratch)

    if verbose:
        fprintf(stderr, b"[DEBUG] fit_two_tower: epoch complete\n")
