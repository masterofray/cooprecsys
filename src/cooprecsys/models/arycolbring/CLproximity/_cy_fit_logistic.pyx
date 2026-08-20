#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_fit_logistic.pyx
# Logistic-loss SGD training kernel for arycolbring.
# Parallelised with OpenMP prange; each thread owns its own malloc'd buffers.

import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio  cimport fprintf, stderr
from cython.parallel cimport prange, parallel

from ._cy_types          cimport CSRMatrix, FastAryColBring, flt
from ._cy_math           cimport sigmoid
from ._cy_representation cimport compute_representation, compute_prediction_from_repr
from ._cy_update         cimport update
from ._cy_regularize     cimport regularize, locked_regularize, omp_lock_t, \
                                 omp_init_lock, omp_destroy_lock

cdef double MAX_REG_SCALE = 1000000.0


def fit_logistic(CSRMatrix item_features,
                 CSRMatrix user_features,
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
                 bint verbose=False):
    """
    One epoch of logistic-loss collaborative filtering.

    Each worker thread allocates its own (user_repr, it_repr) buffers.
    A shared OMP lock serialises the periodic full-regularisation flush.
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] fit_logistic: no_examples=%d num_threads=%d\n",
                Y.shape[0], num_threads)

    cdef int i, row, user_id, item_id, no_examples
    cdef double prediction, loss
    cdef int y
    cdef flt y_row, weight
    cdef flt *user_repr
    cdef flt *it_repr
    cdef omp_lock_t reg_lock

    no_examples = Y.shape[0]

    if no_examples == 0:
        fprintf(stderr, b"[WARN] fit_logistic: no_examples=0, skipping\n")
        return

    omp_init_lock(&reg_lock)
    if verbose:
        fprintf(stderr, b"[DEBUG] fit_logistic: OMP lock initialised\n")

    with nogil, parallel(num_threads=num_threads):
        user_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        it_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))

        if user_repr == NULL or it_repr == NULL:
            fprintf(stderr, b"[ERROR] fit_logistic: malloc failed for thread buffer\n")
        else:
            for i in prange(no_examples, schedule='dynamic'):
                row     = shuffle_indices[i]
                user_id = user_ids[row]
                item_id = item_ids[row]
                weight  = sample_weight[row]

                compute_representation(user_features,
                                       model.user_features,
                                       model.user_biases,
                                       model, user_id,
                                       model.user_scale, user_repr)

                compute_representation(item_features,
                                       model.item_features,
                                       model.item_biases,
                                       model, item_id,
                                       model.item_scale, it_repr)

                prediction = sigmoid(
                    compute_prediction_from_repr(user_repr, it_repr,
                                                 model.no_components))

                y_row = Y[row]
                y = 1 if y_row > 0 else 0

                loss = weight * (prediction - y)

                update(loss, item_features, user_features,
                       user_id, item_id, user_repr, it_repr,
                       model, item_alpha, user_alpha)

                if model.item_scale > MAX_REG_SCALE or model.user_scale > MAX_REG_SCALE:
                    locked_regularize(model, item_alpha, user_alpha,
                                      &reg_lock, MAX_REG_SCALE)

        free(user_repr)
        free(it_repr)

    omp_destroy_lock(&reg_lock)
    regularize(model, item_alpha, user_alpha)
    if verbose:
        fprintf(stderr, b"[DEBUG] fit_logistic: epoch complete\n")
