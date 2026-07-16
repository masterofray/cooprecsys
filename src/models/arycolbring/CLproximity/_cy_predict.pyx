#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_predict.pyx
# Prediction kernels for arycolbring: pointwise scores and item ranks.
# Both are parallelised with OpenMP prange.

from libc.stdlib  cimport malloc, free
from libc.stdio   cimport fprintf, stderr
from cython.parallel cimport prange, parallel

from ._cy_types          cimport CSRMatrix, FastAryColBring, flt
from ._cy_math           cimport in_positives, int_max
from ._cy_representation cimport compute_representation, compute_prediction_from_repr


def predict_arycolbring(CSRMatrix item_features,
                        CSRMatrix user_features,
                        int[::1] user_ids,
                        int[::1] item_ids,
                        flt[::1] predictions,
                        FastAryColBring model,
                        int num_threads,
                        bint cross_join = False,
                        bint verbose = False,
                       ):
    """
    Compute prediction scores for users x items. Two indexing modes,
    selected by `cross_join`:

    Pairwise mode (cross_join=False, default) — requires
    len(user_ids) == len(item_ids) == len(predictions):
        predictions[i] is computed for (user_ids[i], item_ids[i]).
        user_ids[i] / item_ids[i] are used as LITERAL row indices into
        user_features / item_features (classic LightFM-style convention:
        the caller's feature matrix must already be sized/aligned to the
        raw id range, e.g. via sp.identity(max_id + 1)).

    Cross-join / hybrid mode (cross_join=True) — user_ids and item_ids may
    have different lengths N and M; predictions must have length N * M,
    laid out row-major (user 0 vs every item, then user 1 vs every item, ...).
        Row indices into user_features / item_features are POSITIONAL:
        the i-th entry of user_ids maps to row `i` of user_features, and the
        j-th entry of item_ids maps to row `j` of item_features — the actual
        integer VALUES stored in user_ids / item_ids are not read at all in
        this mode. This lets a caller hand in a compact feature matrix built
        for exactly the N users / M items it cares about (e.g. a reporting
        batch) without needing raw ids to be a dense 0..N-1 range. The
        caller is responsible for re-associating each output row with its
        original (raw) user/item label on the Python side, since the kernel
        itself only produces the N*M score grid.

    Parameters
    ----------
    item_features  : CSRMatrix  [n_item_features x n_item_feat_cols]
    user_features  : CSRMatrix  [n_user_features x n_user_feat_cols]
    user_ids       : int32 array, length N
    item_ids       : int32 array, length M
    predictions    : float32 array (output, written in-place).
                     Length must be N if cross_join is False (and N == M),
                     else N * M.
    model          : FastAryColBring  holding all embedding state
    num_threads    : int  ≥ 1
    cross_join     : bool, default False — see modes above.
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] predict_arycolbring: n_examples=%d num_threads=%d cross_join=%d\n",
                predictions.shape[0], num_threads, <int>cross_join)

    cdef int i, j, idx, no_examples, n_u, n_i
    cdef flt *user_repr
    cdef flt *it_repr

    if not cross_join:
        no_examples = predictions.shape[0]

        if no_examples == 0:
            fprintf(stderr, b"[WARN] predict_arycolbring: no_examples=0, nothing to score\n")
            return
        if user_ids.shape[0] != item_ids.shape[0]:
            fprintf(stderr,
                    b"[ERROR] predict_arycolbring: pairwise mode requires user_ids and "
                    b"item_ids of equal length (got %d vs %d); pass cross_join=True for "
                    b"differing lengths\n",
                    user_ids.shape[0], item_ids.shape[0])
            return

        with nogil, parallel(num_threads=num_threads):
            user_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
            it_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))

            if user_repr == NULL or it_repr == NULL:
                fprintf(stderr, b"[ERROR] predict_arycolbring: malloc failed\n")
            else:
                for i in prange(no_examples, schedule='static'):
                    compute_representation(user_features,
                                           model.user_features, model.user_biases,
                                           model, user_ids[i],
                                           model.user_scale, user_repr)
                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, item_ids[i],
                                           model.item_scale, it_repr)

                    predictions[i] = compute_prediction_from_repr(
                        user_repr, it_repr, model.no_components)

            free(user_repr)
            free(it_repr)

    else:
        n_u = user_ids.shape[0]
        n_i = item_ids.shape[0]
        no_examples = n_u * n_i

        if no_examples == 0:
            fprintf(stderr,
                    b"[WARN] predict_arycolbring: cross_join with 0 pairs (n_u=%d, n_i=%d), "
                    b"nothing to score\n", n_u, n_i)
            return
        if predictions.shape[0] != no_examples:
            fprintf(stderr,
                    b"[ERROR] predict_arycolbring: cross_join predictions buffer has length "
                    b"%d but n_u * n_i = %d * %d = %d\n",
                    predictions.shape[0], n_u, n_i, no_examples)
            return

        with nogil, parallel(num_threads=num_threads):
            user_repr = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
            it_repr   = <flt*>malloc(sizeof(flt) * (model.no_components + 1))

            if user_repr == NULL or it_repr == NULL:
                fprintf(stderr, b"[ERROR] predict_arycolbring: malloc failed\n")
            else:
                for idx in prange(no_examples, schedule='static'):
                    i = idx // n_i
                    j = idx % n_i
                    # Positional indexing on purpose — see cross-join mode
                    # docstring above. Raw user_ids[i] / item_ids[j] values
                    # are intentionally not used here.
                    compute_representation(user_features,
                                           model.user_features, model.user_biases,
                                           model, i,
                                           model.user_scale, user_repr)
                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, j,
                                           model.item_scale, it_repr)

                    predictions[idx] = compute_prediction_from_repr(
                        user_repr, it_repr, model.no_components)

            free(user_repr)
            free(it_repr)

    if verbose:
        fprintf(stderr, b"[DEBUG] predict_arycolbring: scoring complete\n")


def predict_ranks(CSRMatrix item_features,
                  CSRMatrix user_features,
                  CSRMatrix test_interactions,
                  CSRMatrix train_interactions,
                  flt[::1]  ranks,
                  FastAryColBring model,
                  int num_threads,
                  bint verbose = False,
                 ):
    """
    Compute the rank of every test-positive item for each user.

    For each user the routine:
      1. Scores the user's test-positive items.
      2. Scores ALL catalogue items (excluding train positives).
      3. Counts how many catalogue items outscore each test positive.
         That count is the item's rank (lower = better recommendation).

    Parameters
    ----------
    item_features     : CSRMatrix
    user_features     : CSRMatrix
    test_interactions : CSRMatrix about positives to rank
    train_interactions: CSRMatrix about positives to skip during ranking
    ranks             : float32 array matching test_interactions.data (output)
    model             : FastAryColBring
    num_threads       : int >= 1
    """
    if verbose:
        fprintf(stderr,
                b"[DEBUG] predict_ranks: n_users = %d num_threads = %d\n",
                test_interactions.rows, num_threads)

    cdef int i, j, user_id, item_id, predictions_size
    cdef int row_start, row_stop
    cdef flt *user_repr
    cdef flt *it_repr
    cdef int *test_item_ids_buf
    cdef flt *test_preds_buf
    cdef flt  prediction

    # Determine maximum row width (for buffer sizing)
    predictions_size = 0
    for user_id in range(test_interactions.rows):
        predictions_size = int_max(
            predictions_size,
            test_interactions.get_row_end(user_id)
            - test_interactions.get_row_start(user_id))
    if verbose:
        fprintf(stderr,
                b"[DEBUG] predict_ranks: max_row_width=%d\n", predictions_size)

    if predictions_size == 0:
        fprintf(stderr, b"[WARN] predict_ranks: no test interactions found\n")
        return

    with nogil, parallel(num_threads = num_threads):
        user_repr       = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        it_repr         = <flt*>malloc(sizeof(flt) * (model.no_components + 1))
        test_item_ids_buf = <int*>malloc(sizeof(int) * predictions_size)
        test_preds_buf  = <flt*>malloc(sizeof(flt) * predictions_size)

        if (user_repr == NULL or it_repr == NULL
                or test_item_ids_buf == NULL or test_preds_buf == NULL):
            fprintf(stderr, b"[ERROR] predict_ranks: malloc failed\n")
        else:
            for user_id in prange(test_interactions.rows, schedule='dynamic'):
                row_start = test_interactions.get_row_start(user_id)
                row_stop  = test_interactions.get_row_end(user_id)

                if row_stop == row_start:
                    continue  # no test interactions for this user

                # Step 1 – compute user representation
                compute_representation(user_features,
                                       model.user_features, model.user_biases,
                                       model, user_id,
                                       model.user_scale, user_repr)

                # Step 2 – score every test-positive item for this user
                for i in range(row_stop - row_start):
                    item_id = test_interactions.indices[row_start + i]
                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, item_id, model.item_scale, it_repr)
                    test_item_ids_buf[i] = item_id
                    test_preds_buf[i]    = compute_prediction_from_repr(
                        user_repr, it_repr, model.no_components)

                # Step 3 – count catalogue items that beat each test positive
                for item_id in range(test_interactions.cols):
                    if in_positives(item_id, user_id, train_interactions):
                        continue  # skip known train positives

                    compute_representation(item_features,
                                           model.item_features, model.item_biases,
                                           model, item_id, model.item_scale, it_repr)
                    prediction = compute_prediction_from_repr(
                        user_repr, it_repr, model.no_components)

                    for i in range(row_stop - row_start):
                        if item_id != test_item_ids_buf[i] and prediction >= test_preds_buf[i]:
                            ranks[row_start + i] += 1.0

        free(user_repr)
        free(it_repr)
        free(test_item_ids_buf)
        free(test_preds_buf)
    if verbose:
        fprintf(stderr, b"[DEBUG] predict_ranks: ranking complete\n")
