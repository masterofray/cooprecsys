# cython: language_level=3
# _cy_types.pxd
# Shared C-level type declarations for the ary2tower Cython module.

ctypedef float flt


cdef class TwoTowerModel:
    """
    Container C-class yang menampung seluruh state parameter dan momentum
    untuk User Tower dan Item Tower dalam bentuk Typed Memoryviews.
    
    Seluruh field dideklarasikan dengan 'public' agar bisa diinspeksi dari
    level Python (misal: unit test / logging) tanpa mengorbankan kecepatan C.
    """
    # --- User Tower Parameters ---
    cdef public flt[:, ::1] user_embeddings          # (n_users, embedding_dim)
    cdef public flt[:, ::1] user_w1                  # (embedding_dim, hidden_dim)
    cdef public flt[::1]    user_b1                  # (hidden_dim,)
    cdef public flt[:, ::1] user_w2                  # (hidden_dim, output_dim)
    cdef public flt[::1]    user_b2                  # (output_dim,)

    # --- User Tower Momentum (SGD / Adam) ---
    cdef public flt[:, ::1] user_embeddings_momentum
    cdef public flt[:, ::1] user_w1_momentum
    cdef public flt[::1]    user_b1_momentum
    cdef public flt[:, ::1] user_w2_momentum
    cdef public flt[::1]    user_b2_momentum

    # --- Item Tower Parameters ---
    cdef public flt[:, ::1] item_embeddings          # (n_items, embedding_dim)
    cdef public flt[:, ::1] item_w1                  # (embedding_dim, hidden_dim)
    cdef public flt[::1]    item_b1                  # (hidden_dim,)
    cdef public flt[:, ::1] item_w2                  # (hidden_dim, output_dim)
    cdef public flt[::1]    item_b2                  # (output_dim,)

    # --- Item Tower Momentum (SGD / Adam) ---
    cdef public flt[:, ::1] item_embeddings_momentum
    cdef public flt[:, ::1] item_w1_momentum
    cdef public flt[::1]    item_b1_momentum
    cdef public flt[:, ::1] item_w2_momentum
    cdef public flt[::1]    item_b2_momentum

    # --- Model Dimensions ---
    cdef public int embedding_dim
    cdef public int hidden_dim
    cdef public int output_dim