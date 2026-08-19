#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_evaluate.pyx
# ROC-AUC computation from pre-computed item ranks for arycolbring.
# Parallelised per-user with OpenMP prange.

from libc.stdio cimport fprintf, stderr
from cython.parallel cimport prange, parallel

from _cy_types cimport CSRMatrix, flt
from _cy_math cimport flt_compare, qsort


def calculate_auc_from_rank(CSRMatrix ranks,
                             int[::1] num_train_positives,
                             flt[::1] rank_data,
                             flt[::1] auc,
                             int num_threads,
                             bint verbose = False):
    """
    Convert per-user item ranks (from predict_ranks) to ROC-AUC scores.
    For each user the AUC is the probability that a randomly chosen
    positive example is ranked above a randomly chosen negative.
    A perfect model scores 1.0; random scoring gives 0.5.

    Parameters
    ----------
    ranks               : CSRMatrix — sparse rank matrix (users × items)
    num_train_positives : int32 array [n_users] — # train positives per user
                          (excluded from the negative count)
    rank_data           : float32 array — the .data buffer of `ranks`
                          (modified in-place: sorted per-user row)
    auc                 : float32 array [n_users] — output AUC per user
    num_threads         : int >= 1
    verbose             : bint - kontrol for fprint log debug (default: False)
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] calculate_auc_from_rank: n_users = %d | num_threads = %d\n",
                ranks.rows, num_threads)

    cdef int i, user_id, row_start, row_stop
    cdef int num_negatives, num_positives
    cdef flt rank
    cdef flt one = <flt>1.0
    cdef flt zero = <flt>0.0
    cdef flt half = <flt>0.5

    with nogil, parallel(num_threads = num_threads):
        for user_id in prange(ranks.rows, schedule='static'):
            row_start = ranks.get_row_start(user_id)
            row_stop = ranks.get_row_end(user_id)
            num_positives = row_stop - row_start
            num_negatives = ranks.cols - num_positives - num_train_positives[user_id]

            if num_positives == 0 or num_negatives <= 0:
                auc[user_id] = half
                continue

            # Sort positive ranks ascending so we can correct for ties
            qsort(&rank_data[row_start],
                  num_positives,
                  sizeof(flt),
                  flt_compare)

            auc[user_id] = zero
            for i in range(num_positives):
                rank = rank_data[row_start + i]
                rank = rank - <flt>i
                if rank < zero:
                    rank = zero

                # P(positive ranked above random negative) for this item
                auc[user_id] = auc[user_id] + (one - (rank / <flt>num_negatives))

            if num_positives != 0:
                auc[user_id] = auc[user_id] / <flt>num_positives

    if verbose:
        fprintf(stderr, b"[DEBUG] calculate_auc_from_rank: done\n")