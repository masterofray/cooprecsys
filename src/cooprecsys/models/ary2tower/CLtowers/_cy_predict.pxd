# cython: language_level=3
# _cy_predict.pxd

from ._cy_types cimport TwoTowerModel, flt

cdef void c_predict_pairs(
    int[::1]      user_ids,
    int[::1]      item_ids,
    flt[::1]      scores,
    TwoTowerModel model,
    int           num_threads,
    flt*          scratch_pool,
    bint          verbose
) noexcept nogil

cdef void c_predict_user_items(
    int[::1]      user_ids,
    flt[:, ::1]   item_outputs,
    flt[:, ::1]   scores_out,
    TwoTowerModel model,
    int           num_threads,
    flt*          scratch_pool,
    bint          verbose
) noexcept nogil