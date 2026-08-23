#!python
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_types.pyx
# Implementasi Inisialisasi dan Guardrails Validation untuk TwoTowerModel.

from libc.stdio cimport fprintf, stderr


cdef class TwoTowerModel:
    """
    Inisialisasi TwoTowerModel dengan validasi dimensi ketat di level Python
    sebelum pointer memoryview dikirim ke kernel C (nogil/prange).
    """

    def __init__(self,
                 flt[:, ::1] user_embeddings,
                 flt[:, ::1] user_w1,
                 flt[::1]    user_b1,
                 flt[:, ::1] user_w2,
                 flt[::1]    user_b2,
                 flt[:, ::1] item_embeddings,
                 flt[:, ::1] item_w1,
                 flt[::1]    item_b1,
                 flt[:, ::1] item_w2,
                 flt[::1]    item_b2,
                 flt[:, ::1] user_embeddings_momentum,
                 flt[:, ::1] user_w1_momentum,
                 flt[::1]    user_b1_momentum,
                 flt[:, ::1] user_w2_momentum,
                 flt[::1]    user_b2_momentum,
                 flt[:, ::1] item_embeddings_momentum,
                 flt[:, ::1] item_w1_momentum,
                 flt[::1]    item_b1_momentum,
                 flt[:, ::1] item_w2_momentum,
                 flt[::1]    item_b2_momentum,
                 bint verbose = False,
                ):

        # ------------------------------------------------------------------
        # 1. Extraction Dimensi Dasar
        # ------------------------------------------------------------------
        cdef int emb_dim = user_embeddings.shape[1]
        cdef int hid_dim = user_w1.shape[1]
        cdef int out_dim = user_w2.shape[1]

        # ------------------------------------------------------------------
        # 2. Shape Validation Guardrails
        #    Mencegah Segfault / Memory Corruption saat operasi nogil C
        # ------------------------------------------------------------------
        # User Tower MatMul Checks: (Emb @ W1 + b1) -> (Hidden @ W2 + b2)
        if user_w1.shape[0] != emb_dim:
            raise ValueError(f"user_w1.shape[0] ({user_w1.shape[0]}) != embedding_dim ({emb_dim})")
        if user_b1.shape[0] != hid_dim:
            raise ValueError(f"user_b1.shape[0] ({user_b1.shape[0]}) != hidden_dim ({hid_dim})")
        if user_w2.shape[0] != hid_dim:
            raise ValueError(f"user_w2.shape[0] ({user_w2.shape[0]}) != hidden_dim ({hid_dim})")
        if user_b2.shape[0] != out_dim:
            raise ValueError(f"user_b2.shape[0] ({user_b2.shape[0]}) != output_dim ({out_dim})")

        # Item Tower MatMul Checks
        if item_embeddings.shape[1] != emb_dim:
            raise ValueError(f"item_embeddings dim ({item_embeddings.shape[1]}) != user embedding_dim ({emb_dim})")
        if item_w1.shape[0] != emb_dim or item_w1.shape[1] != hid_dim:
            raise ValueError("item_w1 shape mismatch with model architecture dimensions")
        if item_b1.shape[0] != hid_dim:
            raise ValueError(f"item_b1.shape[0] ({item_b1.shape[0]}) != hidden_dim ({hid_dim})")
        if item_w2.shape[0] != hid_dim or item_w2.shape[1] != out_dim:
            raise ValueError("item_w2 shape mismatch with model architecture dimensions")
        if item_b2.shape[0] != out_dim:
            raise ValueError(f"item_b2.shape[0] ({item_b2.shape[0]}) != output_dim ({out_dim})")

        # Momentum Matrix Match Checks
        if user_embeddings_momentum.shape[0] != user_embeddings.shape[0] or user_embeddings_momentum.shape[1] != emb_dim:
            raise ValueError("user_embeddings_momentum shape does not match user_embeddings")
        if item_embeddings_momentum.shape[0] != item_embeddings.shape[0] or item_embeddings_momentum.shape[1] != emb_dim:
            raise ValueError("item_embeddings_momentum shape does not match item_embeddings")
        if user_w1_momentum.shape[0] != user_w1.shape[0] or user_w1_momentum.shape[1] != user_w1.shape[1]:
            raise ValueError("user_w1_momentum shape does not match user_w1")
        if user_b1_momentum.shape[0] != user_b1.shape[0]:
            raise ValueError("user_b1_momentum shape does not match user_b1")
        if user_w2_momentum.shape[0] != user_w2.shape[0] or user_w2_momentum.shape[1] != user_w2.shape[1]:
            raise ValueError("user_w2_momentum shape does not match user_w2")
        if user_b2_momentum.shape[0] != user_b2.shape[0]:
            raise ValueError("user_b2_momentum shape does not match user_b2")
        if item_w1_momentum.shape[0] != item_w1.shape[0] or item_w1_momentum.shape[1] != item_w1.shape[1]:
            raise ValueError("item_w1_momentum shape does not match item_w1")
        if item_b1_momentum.shape[0] != item_b1.shape[0]:
            raise ValueError("item_b1_momentum shape does not match item_b1")
        if item_w2_momentum.shape[0] != item_w2.shape[0] or item_w2_momentum.shape[1] != item_w2.shape[1]:
            raise ValueError("item_w2_momentum shape does not match item_w2")
        if item_b2_momentum.shape[0] != item_b2.shape[0]:
            raise ValueError("item_b2_momentum shape does not match item_b2")

        # ------------------------------------------------------------------
        # 3. Assignment
        # ------------------------------------------------------------------
        self.user_embeddings = user_embeddings
        self.user_w1         = user_w1
        self.user_b1         = user_b1
        self.user_w2         = user_w2
        self.user_b2         = user_b2

        self.item_embeddings = item_embeddings
        self.item_w1         = item_w1
        self.item_b1         = item_b1
        self.item_w2         = item_w2
        self.item_b2         = item_b2

        self.user_embeddings_momentum = user_embeddings_momentum
        self.user_w1_momentum         = user_w1_momentum
        self.user_b1_momentum         = user_b1_momentum
        self.user_w2_momentum         = user_w2_momentum
        self.user_b2_momentum         = user_b2_momentum

        self.item_embeddings_momentum = item_embeddings_momentum
        self.item_w1_momentum         = item_w1_momentum
        self.item_b1_momentum         = item_b1_momentum
        self.item_w2_momentum         = item_w2_momentum
        self.item_b2_momentum         = item_b2_momentum

        self.embedding_dim = emb_dim
        self.hidden_dim    = hid_dim
        self.output_dim    = out_dim

        if verbose:
            fprintf(stderr,
                    b"[DEBUG] TwoTowerModel initialized: emb_dim=%d hid_dim=%d out_dim=%d | n_users=%d n_items=%d\n",
                    self.embedding_dim, self.hidden_dim, self.output_dim,
                    <int>user_embeddings.shape[0], <int>item_embeddings.shape[0])