# _cy_regularize.pxd
# Inline L2 regularisation helpers for arycolbring.
# regularize()        – flush accumulated lazy scale factor across all weights.
# locked_regularize() – thread-safe version using an OMP lock pointer.

from _cy_types cimport FastAryColBring, flt

cdef extern from "omp.h" nogil:
    ctypedef struct omp_lock_t:
        pass
    void omp_init_lock(omp_lock_t *) nogil
    void omp_destroy_lock(omp_lock_t *) nogil
    void omp_set_lock(omp_lock_t *) nogil
    void omp_unset_lock(omp_lock_t *) nogil


cdef inline void regularize(FastAryColBring model,
                             double item_alpha,
                             double user_alpha) nogil:
    """
    Flush the accumulated lazy-regularisation scale factors so that
    item_scale and user_scale are reset to 1.0.
    Every weight is divided by its current scale value.
    """
    cdef int i, j
    cdef int no_item_features = model.item_features.shape[0]
    cdef int no_user_features = model.user_features.shape[0]

    for i in range(no_item_features):
        for j in range(model.no_components):
            model.item_features[i, j] /= model.item_scale
        model.item_biases[i] /= model.item_scale

    for i in range(no_user_features):
        for j in range(model.no_components):
            model.user_features[i, j] /= model.user_scale
        model.user_biases[i] /= model.user_scale

    model.item_scale = 1.0
    model.user_scale = 1.0


cdef inline void w(FastAryColBring model,
                                   double item_alpha,
                                   double user_alpha,
                                   omp_lock_t *lock,
                                   double MAX_REG_SCALE) nogil:
    """
    Thread-safe regularisation flush.
    Acquires *lock*, re-checks the threshold (another thread may have flushed
    already) and flushes if needed, then releases *lock*.
    """
    omp_set_lock(lock)
    if model.item_scale > MAX_REG_SCALE or model.user_scale > MAX_REG_SCALE:
        regularize(model, item_alpha, user_alpha)
    omp_unset_lock(lock)
