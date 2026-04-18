#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_fit_warp_kos.pyx
# WARP k-OS (k-th Order Statistic) training kernel for arycolbring.
# Selects the k-th highest-ranked positive item per user before applying WARP.

import numpy as np
from libc.stdlib  cimport malloc, free
from libc.stdio   cimport fprintf, stderr
from cython.parallel cimport prange, parallel, threadid

from _cy_types          cimport CSRMatrix, FastAryColBring, flt
from _cy_math           cimport (rand_r, sample_range, in_positives,
                                  int_min, Pair, reverse_pair_compare, qsort)
from _cy_representation cimport compute_representation, compute_prediction_from_repr
from _cy_update         cimport warp_update
from _cy_regularize     cimport regularize, locked_regularize, omp_lock_t, \
                                 omp_init_lock, omp_destroy_lock

cdef extern from "math.h" nogil:
    double log(double)
    double floor(double)

cdef double MAX_REG_SCALE = 1000000.0


def fit_warp_kos(CSRMatrix item_features,
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
    One epoch of WARP-kOS collaborative filtering.

    For each user, `n` positive items are sampled; the k-th highest-scored
    one is selected as the anchor. Then WARP negative sampling proceeds as
    normal from that anchor.
    """
    fprintf(stderr,
            b"[DEBUG] fit_warp_kos: no_examples=%d k=%d n=%d num_threads=%d\n",
            user_ids.shape[0], k, n, num_threads)

    cdef int i, j, no_examples, user_id, positive_item_id, negative_item_id
    cdef int sampled, row, sampled_positive_item_id
    cdef int user_pids_start, user_pids_stop, no_positives
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
                     .astype(np.uint32))

    no_examples = user_ids.shape[0]

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_warp_kos: no_examples=0, skipping\n")
        return

    omp_init_lock(&reg_lock)
    fprintf(stderr, b"[DEBUG] fit_warp_kos: OMP lock initialised\n")

    with nogil, parallel(num_threads=num_threads):
        user_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        pos_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        neg_it_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        pos_pairs   = <Pair*>malloc(sizeof(Pair) * n)

        if (user_repr == NULL or pos_it_repr == NULL
                or neg_it_repr == NULL or pos_pairs == NULL):
            fprintf(stderr, b"[ERROR] fit_warp_kos: malloc failed\n")
        else:
            for i in prange(no_examples, schedule='dynamic'):
                row     = shuffle_indices[i]
                user_id = user_ids[row]

                compute_representation(user_features,
                                       model.user_features, model.user_biases,
                                       model, user_id, model.user_scale, user_repr)

                user_pids_start = data.get_row_start(user_id)
                user_pids_stop  = data.get_row_end(user_id)

                if user_pids_stop == user_pids_start:
                    continue

                # Sample up to n positives and score them
                no_positives = int_min(n, user_pids_stop - user_pids_start)
                for j in range(no_positives):
                    sampled_positive_item_id = data.indices[
                        sample_range(user_pids_start, user_pids_stop,
                                     &random_states[threadid()])]

                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, sampled_positive_item_id,
                                           model.item_scale, pos_it_repr)

                    sampled_positive_prediction = compute_prediction_from_repr(
                        user_repr, pos_it_repr, model.no_components)

                    pos_pairs[j].idx = sampled_positive_item_id
                    pos_pairs[j].val = <flt>sampled_positive_prediction

                # Pick the k-th order statistic (descending sort, then take index k-1)
                qsort(pos_pairs, no_positives, sizeof(Pair), reverse_pair_compare)

                positive_item_id    = pos_pairs[int_min(k, no_positives) - 1].idx
                positive_prediction = pos_pairs[int_min(k, no_positives) - 1].val

                compute_representation(item_features,
                                       model.item_features, model.item_biases,
                                       model, positive_item_id, model.item_scale,
                                       pos_it_repr)

                # WARP negative sampling phase
                sampled = 0
                while sampled < model.max_sampled:
                    sampled += 1
                    negative_item_id = (rand_r(&random_states[threadid()])
                                        % item_features.rows)

                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, negative_item_id, model.item_scale,
                                           neg_it_repr)

                    negative_prediction = compute_prediction_from_repr(
                        user_repr, neg_it_repr, model.no_components)

                    if negative_prediction > positive_prediction - 1:
                        if in_positives(negative_item_id, user_id, data):
                            continue

                        loss = log(floor(
                            (item_features.rows - 1) / <double>sampled))
                        if loss > MAX_LOSS:
                            loss = MAX_LOSS

                        warp_update(loss, item_features, user_features,
                                    user_id, positive_item_id, negative_item_id,
                                    user_repr, pos_it_repr, neg_it_repr,
                                    model, item_alpha, user_alpha)
                        break

                if model.item_scale > MAX_REG_SCALE or model.user_scale > MAX_REG_SCALE:
                    locked_regularize(model, item_alpha, user_alpha,
                                      &reg_lock, MAX_REG_SCALE)

        free(user_repr)
        free(pos_it_repr)
        free(neg_it_repr)
        free(pos_pairs)

    omp_destroy_lock(&reg_lock)

    regularize(model, item_alpha, user_alpha)

    fprintf(stderr, b"[DEBUG] fit_warp_kos: epoch complete\n")
