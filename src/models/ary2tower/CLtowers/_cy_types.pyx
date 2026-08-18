#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# _cy_types.pyx
# Implementation of the TwoTowerModel cdef class.

from libc.stdio cimport fprintf, stderr


cdef class TwoTowerModel:
    """
    Central model-state container for the two-tower model.
    All embedding/weight/bias matrices and their momentum accumulators
    live here. Passed by reference into every Cython kernel (forward
    pass, similarity, SGD update) so they can read/mutate the arrays
    in-place without Python-object overhead -- same design as
    FastAryColBring in arycolbring/CLproximity/_cy_types.pyx.
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

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim    = user_w1.shape[1]
        self.output_dim    = user_w2.shape[1]

        if verbose:
            fprintf(stderr,
                    b"[DEBUG] TwoTowerModel.__init__: embedding_dim=%d hidden_dim=%d "
                    b"output_dim=%d n_users=%d n_items=%d\n",
                    self.embedding_dim, self.hidden_dim, self.output_dim,
                    <int>user_embeddings.shape[0], <int>item_embeddings.shape[0])
