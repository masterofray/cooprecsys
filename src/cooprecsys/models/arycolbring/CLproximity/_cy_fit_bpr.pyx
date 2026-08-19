#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_fit_bpr.pyx
# BPR (Bayesian Personalised Ranking) training kernel for arycolbring.

import numpy as np
from libc.stdlib  cimport malloc, free
from libc.stdio   cimport fprintf, stderr
from cython.parallel cimport prange, parallel, threadid

from _cy_types          cimport CSRMatrix, FastAryColBring, flt
from _cy_math           cimport rand_r, in_positives, sigmoid
from _cy_representation cimport compute_representation, compute_prediction_from_repr
from _cy_update         cimport warp_update
from _cy_regularize     cimport regularize, locked_regularize, omp_lock_t, \
                                 omp_init_lock, omp_destroy_lock

cdef double MAX_REG_SCALE = 1000000.0


def fit_bpr(CSRMatrix item_features,
            CSRMatrix user_features,
            CSRMatrix interactions,
            int[::1] user_ids,
            int[::1] item_ids,
            flt[::1] Y,
            flt[::1] sample_weight,
            int[::1] shuffle_indices,
            FastAryColBring model,
            double learning_rate,
            double item_alpha,
            double user_alpha,
            int num_threads,
            random_state,
            bint verbose = False):
    """
    One epoch of BPR-loss collaborative filtering.

    For each positive, one hard negative is sampled and a pairwise
    gradient update (same kernel as WARP) is applied with a sigmoid loss.
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] fit_bpr: no_examples=%d num_threads=%d\n",
                Y.shape[0], num_threads)

    cdef int i, j, no_examples, user_id, positive_item_id, negative_item_id
    cdef int sampled, row
    cdef double positive_prediction, negative_prediction
    cdef flt weight
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef unsigned int[::1] random_states
    cdef omp_lock_t reg_lock

    random_states = (random_state
                     .randint(0, np.iinfo(np.int32).max, size=num_threads)
                     .astype(np.uint32))

    no_examples = Y.shape[0]

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_bpr: no_examples=0, skipping\n")
        return

    omp_init_lock(&reg_lock)
    fprintf(stderr, b"[DEBUG] fit_bpr: OMP lock initialised\n")

    with nogil, parallel(num_threads=num_threads):
        user_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        pos_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        neg_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))

        if user_repr == NULL or pos_it_repr == NULL or neg_it_repr == NULL:
            fprintf(stderr, b"[ERROR] fit_bpr: malloc failed for thread buffer\n")
        else:
            for i in prange(no_examples, schedule='dynamic'):
                row = shuffle_indices[i]

                if not Y[row] > 0:
                    continue

                weight           = sample_weight[row]
                user_id          = user_ids[row]
                positive_item_id = item_ids[row]

                # Sample a negative item (not in user's positives)
                for j in range(no_examples):
                    negative_item_id = item_ids[
                        rand_r(&random_states[threadid()]) % no_examples]
                    if not in_positives(negative_item_id, user_id, interactions):
                        break

                compute_representation(user_features,
                                       model.user_features, model.user_biases,
                                       model, user_id, model.user_scale, user_repr)
                compute_representation(item_features,
                                       model.item_features, model.item_biases,
                                       model, positive_item_id, model.item_scale,
                                       pos_it_repr)
                compute_representation(item_features,
                                       model.item_features, model.item_biases,
                                       model, negative_item_id, model.item_scale,
                                       neg_it_repr)

                positive_prediction = compute_prediction_from_repr(
                    user_repr, pos_it_repr, model.no_components)
                negative_prediction = compute_prediction_from_repr(
                    user_repr, neg_it_repr, model.no_components)

                warp_update(
                    weight * (1.0 - sigmoid(<flt>(positive_prediction
                                                  - negative_prediction))),
                    item_features, user_features,
                    user_id, positive_item_id, negative_item_id,
                    user_repr, pos_it_repr, neg_it_repr,
                    model, item_alpha, user_alpha)

                if model.item_scale > MAX_REG_SCALE or model.user_scale > MAX_REG_SCALE:
                    locked_regularize(model, item_alpha, user_alpha,
                                      &reg_lock, MAX_REG_SCALE)

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)

    omp_destroy_lock(&reg_lock)

    regularize(model, item_alpha, user_alpha)
    if verbose:
        fprintf(stderr, b"[DEBUG] fit_bpr: epoch complete\n")
