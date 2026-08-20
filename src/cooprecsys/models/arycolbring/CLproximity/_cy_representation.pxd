# _cy_representation.pxd
# Inline functions for computing latent representations and dot-product
# predictions for arycolbring collaborative filtering.

from cooprecsys.models.arycolbring.CLproximity._cy_types cimport CSRMatrix, FastAryColBring, flt


cdef inline void compute_representation(CSRMatrix features,
                                        flt[:, ::1] feature_embeddings,
                                        flt[::1]    feature_biases,
                                        FastAryColBring model,
                                        int  row_id,
                                        double scale,
                                        flt  *representation) noexcept nogil:
    """
    Accumulate the weighted embedding and bias for a given row (user or item).

    The output `representation` is a C array of length (no_components + 1):
      - indices 0 .. no_components-1 : latent factor values
      - index   no_components         : bias term
    """
    cdef int i, j, start_index, stop_index, feature
    cdef flt feature_weight

    start_index = features.get_row_start(row_id)
    stop_index  = features.get_row_end(row_id)

    # Zero-initialise output buffer
    for i in range(model.no_components + 1):
        representation[i] = 0.0

    for i in range(start_index, stop_index):
        feature        = features.indices[i]
        feature_weight = <flt>(features.data[i] * scale)

        for j in range(model.no_components):
            representation[j] += feature_weight * feature_embeddings[feature, j]

        # Bias sits at position no_components
        representation[model.no_components] += feature_weight * feature_biases[feature]


cdef inline flt compute_prediction_from_repr(flt *user_repr,
                                             flt *item_repr,
                                             int  no_components) noexcept nogil:
    """
    Compute the score for (user, item) as:
        user_bias + item_bias + dot(user_latent, item_latent)
    """
    cdef int i
    cdef flt result

    # Bias terms
    result = user_repr[no_components] + item_repr[no_components]

    # Dot product of latent factors
    for i in range(no_components):
        result += user_repr[i] * item_repr[i]

    return result