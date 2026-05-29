#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_fit_warp_kos.pyx
# WARP k-OS kernel; no prange on outer loop; no reduction‑var error.

import numpy as np
cimport numpy as np

from libc.stdlib  cimport malloc, free, rand, qsort
from libc.stdio   cimport fprintf, stderr

from ._cy_types          cimport CSRMatrix, FastAryColBring, flt
from ._cy_math           cimport (sample_range, in_positives,
                                  int_min, Pair, reverse_pair_compare, qsort)
from ._cy_representation cimport compute_representation, compute_prediction_from_repr
from ._cy_update         cimport warp_update
from ._cy_regularize     cimport regularize, locked_regularize, omp_lock_t, omp_init_lock, omp_destroy_lock

cdef extern from "math.h" nogil:
    double log(double)
    double floor(double)

cdef double MAX_REG_SCALE = 1000000.0


def fit_warp_kos(
        CSRMatrix item_features,
        CSRMatrix user_features,
        CSRMatrix data,
        int[::1] user_ids,
        int[::1] shuffle_indices,
        FastAryColBring model,
        double learning_rate,
        double item_alpha,
        double user_alpha,
        int k,
        int n,
        int num_threads,
        random_state):
    """
    One epoch of WARP-kOS.

    Outer loop is `range`, not `prange`, to avoid reduction‑variable warnings
    on `sampled` and `sampled_local`.
    """
    fprintf(stderr,
            b"[DEBUG] fit_warp_kos: no_examples=%d k=%d n=%d num_threads=%d\n",
            user_ids.shape[0], k, n, num_threads)

    cdef int i, j, no_examples, user_id, positive_item_id, negative_item_id
    cdef int sampled, sampled_local, row, sampled_positive_item_id
    cdef int user_pids_start, user_pids_stop, no_positives, idx
    cdef double positive_prediction, negative_prediction, sampled_positive_prediction
    cdef double loss, MAX_LOSS = 10.0
    cdef flt *user_repr
    cdef flt *pos_it_repr
    cdef flt *neg_it_repr
    cdef Pair *pos_pairs
    cdef unsigned int[::1] random_states
    cdef omp_lock_t reg_lock

    random_states = (random_state
                     .randint(0, np.iinfo(np.int32).max, size=num_threads)
                     .astype(np.uintc))

    no_examples = user_ids.shape[0]

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_warp_kos: no_examples=0, skipping\n")
        return

    omp_init_lock(&reg_lock)
    fprintf(stderr, b"[DEBUG] fit_warp_kos: OMP lock initialised\n")

    user_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
    pos_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
    neg_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
    pos_pairs   = <Pair*>malloc(sizeof(Pair) * n)

    if user_repr == NULL or pos_it_repr == NULL or neg_it_repr == NULL or pos_pairs == NULL:
        fprintf(stderr, b"[ERROR] fit_warp_kos: malloc failed\n")
        if user_repr   != NULL: free(user_repr)
        if pos_it_repr != NULL: free(pos_it_repr)
        if neg_it_repr != NULL: free(neg_it_repr)
        if pos_pairs   != NULL: free(pos_pairs)
        omp_destroy_lock(&reg_lock)
        return

    # No `prange`; sequential outer loop avoids reduction warning.
    for i in range(no_examples):

        row     = shuffle_indices[i]
        user_id = user_ids[row]

        compute_representation(user_features,
                               model.user_features, model.user_biases,
                               model, user_id, model.user_scale, user_repr)

        user_pids_start = data.get_row_start(user_id)
        user_pids_stop  = data.get_row_end(user_id)

        if user_pids_stop == user_pids_start:
            continue

        no_positives = int_min(n, user_pids_stop - user_pids_start)
        for j in range(no_positives):
            idx = user_pids_start + rand() % (user_pids_stop - user_pids_start)
            sampled_positive_item_id = data.indices[idx]

            compute_representation(item_features,
                                   model.item_features, model.item_biases,
                                   model, sampled_positive_item_id,
                                   model.item_scale, pos_it_repr)

            sampled_positive_prediction = compute_prediction_from_repr(
                user_repr, pos_it_repr, model.no_components)

            pos_pairs[j].idx = sampled_positive_item_id
            pos_pairs[j].val = <flt>sampled_positive_prediction

        qsort(pos_pairs, no_positives, sizeof(Pair), reverse_pair_compare)

        positive_item_id    = pos_pairs[int_min(k, no_positives) - 1].idx
        positive_prediction = pos_pairs[int_min(k, no_positives) - 1].val

        compute_representation(item_features,
                               model.item_features, model.item_biases,
                               model, positive_item_id, model.item_scale,
                               pos_it_repr)

        sampled = 0
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
                if in_positives(negative_item_id, user_id, data):
                    continue

                sampled_local = sampled
                loss = log(floor((item_features.rows - 1) / <double>sampled_local))
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
    free(pos_pairs)

    omp_destroy_lock(&reg_lock)

    regularize(model, item_alpha, user_alpha)

    fprintf(stderr, b"[DEBUG] fit_warp_kos: epoch complete\n")