#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_fit_warp.pyx
# WARP-loss kernel; no prange reduction on `sampled` / `sampled_local`.

import numpy as np
cimport numpy as np

from libc.stdlib  cimport malloc, free, rand
from libc.stdio   cimport fprintf, stderr

from ._cy_types          cimport CSRMatrix, FastAryColBring, flt
from ._cy_math           cimport in_positives
from ._cy_representation cimport compute_representation, compute_prediction_from_repr
from ._cy_update         cimport warp_update
from ._cy_regularize     cimport regularize, locked_regularize, omp_lock_t, omp_init_lock, omp_destroy_lock

cdef extern from "math.h" nogil:
    double log(double)
    double floor(double)

cdef double MAX_REG_SCALE = 1000000.0


def fit_warp(
        CSRMatrix item_features,
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
    One epoch of WARP-loss collaborative filtering.

    We keep the outer loop sequential; inner WARP loop is `nogil` but not `prange`.
    This avoids reduction‑variable errors on `sampled` and `sampled_local`.
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] fit_warp: no_examples=%d num_threads=%d\n",
                Y.shape[0], num_threads)

    cdef int i, no_examples, user_id, positive_item_id, negative_item_id
    cdef int sampled, sampled_local, row, local_seed_index
    cdef double positive_prediction, negative_prediction, loss
    cdef double MAX_LOSS = 10.0
    cdef flt weight
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef unsigned int[::1] random_states
    cdef omp_lock_t reg_lock

    random_states = (random_state
                     .randint(0, np.iinfo(np.int32).max, size=num_threads)
                     .astype(np.uintc))

    no_examples = Y.shape[0]

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_warp: no_examples=0, skipping\n")
        return

    omp_init_lock(&reg_lock)
    if verbose:
        fprintf(stderr, b"[DEBUG] fit_warp: OMP lock initialised\n")

    user_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
    pos_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
    neg_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))

    if user_repr == NULL or pos_it_repr == NULL or neg_it_repr == NULL:
        fprintf(stderr, b"[ERROR] fit_warp: malloc failed for thread buffer\n")
        if user_repr   != NULL: free(user_repr)
        if pos_it_repr != NULL: free(pos_it_repr)
        if neg_it_repr != NULL: free(neg_it_repr)
        omp_destroy_lock(&reg_lock)
        return

    # No `prange` here; keep outer loop sequential to avoid reduction warning.
    for i in range(no_examples):

        row              = shuffle_indices[i]
        user_id          = user_ids[row]
        positive_item_id = item_ids[row]

        if not Y[row] > 0:
            continue

        weight = sample_weight[row]

        compute_representation(user_features,
                               model.user_features, model.user_biases,
                               model, user_id, model.user_scale, user_repr)

        compute_representation(item_features,
                               model.item_features, model.item_biases,
                               model, positive_item_id, model.item_scale,
                               pos_it_repr)

        positive_prediction = compute_prediction_from_repr(
            user_repr, pos_it_repr, model.no_components)

        sampled = 0
        local_seed_index = i % num_threads
        while sampled < model.max_sampled:
            sampled += 1
            negative_item_id = rand() % item_features.rows

            compute_representation(item_features,
                                   model.item_features, model.item_biases,
                                   model, negative_item_id, model.item_scale,
                                   neg_it_repr)

            negative_prediction = compute_prediction_from_repr(
                user_repr, neg_it_repr, model.no_components)

            if negative_prediction > positive_prediction - 1:
                if in_positives(negative_item_id, user_id, interactions):
                    continue

                sampled_local = sampled
                loss = weight * log(
                    max(1.0,
                        floor((item_features.rows - 1) / <double>sampled_local)))

                if loss > MAX_LOSS:
                    loss = MAX_LOSS

                warp_update(loss, item_features, user_features,
                            user_id, positive_item_id, negative_item_id,
                            user_repr, pos_it_repr, neg_it_repr,
                            model, item_alpha, user_alpha)
                break

        if model.item_scale > MAX_REG_SCALE or model.user_scale > MAX_REG_SCALE:
            locked_regularize(model, item_alpha, user_alpha, &reg_lock, MAX_REG_SCALE)

    free(user_repr)
    free(pos_it_repr)
    free(neg_it_repr)
    omp_destroy_lock(&reg_lock)
    regularize(model, item_alpha, user_alpha)
    if verbose:
        fprintf(stderr, b"[DEBUG] fit_warp: epoch complete\n")

