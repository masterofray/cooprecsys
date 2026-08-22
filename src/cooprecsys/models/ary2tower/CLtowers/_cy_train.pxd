# cython: language_level=3
# _cy_train.pxd
# C-level declarations for BPR pairwise two-tower training step.

from cooprecsys.models.ary2tower.CLtowers._cy_types cimport TwoTowerModel, flt

cdef flt c_sigmoid(flt x) noexcept nogil
cdef unsigned int c_rand_r(unsigned int *seed) noexcept nogil
cdef flt c_dot(flt* a, flt* b, int dim) noexcept nogil

cdef void tower_forward_single(
    flt[:, ::1] embeddings,
    flt[:, ::1] w1,
    flt[::1]    b1,
    flt[:, ::1] w2,
    flt[::1]    b2,
    int         entity_id,
    int         embedding_dim,
    int         hidden_dim,
    int         output_dim,
    flt*        hidden_scratch,
    flt*        out_scratch
) noexcept nogil

cdef void tower_backward_update(
    flt[:, ::1] embeddings,
    flt[:, ::1] w1, flt[::1] b1,
    flt[:, ::1] w2, flt[::1] b2,
    flt[:, ::1] embeddings_momentum,
    flt[:, ::1] w1_momentum, flt[::1] b1_momentum,
    flt[:, ::1] w2_momentum, flt[::1] b2_momentum,
    int         entity_id,
    flt*        hidden,
    flt*        d_out,
    int         embedding_dim,
    int         hidden_dim,
    int         output_dim,
    double      learning_rate,
    double      momentum_coef,
    flt*        d_hidden_scratch
) noexcept nogil

cdef void c_fit_two_tower(
    int[::1]          user_ids,
    int[::1]          positive_item_ids,
    int[::1]          shuffle_indices,
    TwoTowerModel     model,
    double            learning_rate,
    double            momentum_coef,
    int               num_threads,
    unsigned int[::1] random_states,
    flt*              scratch_pool,
    bint              verbose
) noexcept nogil