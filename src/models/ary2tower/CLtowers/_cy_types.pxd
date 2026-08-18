# _cy_types.pxd
# Shared type declarations for ary2tower Cython modules.
# cimport this file in any .pyx that needs TwoTowerModel.
#
# Mirrors the style of arycolbring/CLproximity/_cy_types.pxd (same `flt`
# alias, same cdef-class-of-memoryviews pattern) so the two Cython
# modules in this repo read consistently.

ctypedef float flt


cdef class TwoTowerModel:
    """
    Holds all trainable state for both towers of a two-tower model:

        tower_out = W2 @ ReLU(W1 @ embedding[id] + b1) + b2

    One instance holds BOTH the user tower and the item tower's state
    side by side (rather than two separate objects) so a single
    `TwoTowerModel` can be passed into the forward/update kernels and
    the caller picks user_* vs item_* fields as needed.

    All fields are typed memoryviews for direct, nogil-safe C-level
    access -- same convention as FastAryColBring in
    arycolbring/CLproximity/_cy_types.pxd.
    """
    # --- User tower ---
    cdef flt[:, ::1] user_embeddings      # (n_users,        embedding_dim)
    cdef flt[:, ::1] user_w1              # (embedding_dim,  hidden_dim)
    cdef flt[::1]    user_b1              # (hidden_dim,)
    cdef flt[:, ::1] user_w2              # (hidden_dim,     output_dim)
    cdef flt[::1]    user_b2              # (output_dim,)

    cdef flt[:, ::1] user_w1_momentum
    cdef flt[::1]    user_b1_momentum
    cdef flt[:, ::1] user_w2_momentum
    cdef flt[::1]    user_b2_momentum
    cdef flt[:, ::1] user_embeddings_momentum

    # --- Item tower ---
    cdef flt[:, ::1] item_embeddings      # (n_items,        embedding_dim)
    cdef flt[:, ::1] item_w1              # (embedding_dim,  hidden_dim)
    cdef flt[::1]    item_b1              # (hidden_dim,)
    cdef flt[:, ::1] item_w2              # (hidden_dim,     output_dim)
    cdef flt[::1]    item_b2              # (output_dim,)

    cdef flt[:, ::1] item_w1_momentum
    cdef flt[::1]    item_b1_momentum
    cdef flt[:, ::1] item_w2_momentum
    cdef flt[::1]    item_b2_momentum
    cdef flt[:, ::1] item_embeddings_momentum

    cdef int embedding_dim
    cdef int hidden_dim
    cdef int output_dim
