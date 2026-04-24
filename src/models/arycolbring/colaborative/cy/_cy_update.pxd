# _cy_update.pxd
# Inline SGD update kernels for arycolbring.
# Supports both Adagrad and Adadelta learning-rate schedules.

from _cy_types cimport CSRMatrix, FastAryColBring, flt

cdef extern from "math.h" nogil:
    double sqrt(double)


# ── Bias update ──────────────────────────────────────────────────────────────

cdef inline double update_biases(CSRMatrix feature_indices,
                                 int start,
                                 int stop,
                                 flt[::1] biases,
                                 flt[::1] gradients,
                                 flt[::1] momentum,
                                 double gradient,
                                 int    adadelta,
                                 double learning_rate,
                                 double alpha,
                                 flt    rho,
                                 flt    eps) nogil:
    """
    Apply one SGD step on bias terms using Adagrad or Adadelta.
    Returns sum of per-feature local learning rates.
    """
    cdef int i, feature
    cdef double feature_weight, local_lr, update, sum_lr = 0.0

    if adadelta:
        for i in range(start, stop):
            feature        = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature]  = (rho * gradients[feature]
                                   + (1.0 - rho) * (feature_weight * gradient) ** 2)
            local_lr            = (sqrt(momentum[feature] + eps)
                                   / sqrt(gradients[feature] + eps))
            update              = local_lr * gradient * feature_weight
            momentum[feature]   = (rho * momentum[feature]
                                   + (1.0 - rho) * update ** 2)
            biases[feature]    -= <flt>update
            # Lazy L2 regularisation: inflate scale, not every weight
            biases[feature]    *= <flt>(1.0 + alpha * local_lr)
            sum_lr             += local_lr
    else:
        for i in range(start, stop):
            feature        = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_lr           = learning_rate / sqrt(gradients[feature])
            biases[feature]   -= <flt>(local_lr * feature_weight * gradient)
            gradients[feature] += <flt>((gradient * feature_weight) ** 2)
            biases[feature]   *= <flt>(1.0 + alpha * local_lr)
            sum_lr            += local_lr

    return sum_lr


# ── Embedding update ─────────────────────────────────────────────────────────

cdef inline double update_features(CSRMatrix  feature_indices,
                                   flt[:, ::1] features,
                                   flt[:, ::1] gradients,
                                   flt[:, ::1] momentum,
                                   int    component,
                                   int    start,
                                   int    stop,
                                   double gradient,
                                   int    adadelta,
                                   double learning_rate,
                                   double alpha,
                                   flt    rho,
                                   flt    eps) nogil:
    """
    Apply one SGD step for a single latent component across all active features.
    Returns sum of per-feature local learning rates.
    """
    cdef int i, feature
    cdef double feature_weight, local_lr, update, sum_lr = 0.0

    if adadelta:
        for i in range(start, stop):
            feature        = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            gradients[feature, component]  = (
                rho * gradients[feature, component]
                + (1.0 - rho) * (feature_weight * gradient) ** 2)
            local_lr = (sqrt(momentum[feature, component] + eps)
                        / sqrt(gradients[feature, component] + eps))
            update   = local_lr * gradient * feature_weight
            momentum[feature, component] = (
                rho * momentum[feature, component] + (1.0 - rho) * update ** 2)
            features[feature, component] -= <flt>update
            features[feature, component] *= <flt>(1.0 + alpha * local_lr)
            sum_lr += local_lr
    else:
        for i in range(start, stop):
            feature        = feature_indices.indices[i]
            feature_weight = feature_indices.data[i]

            local_lr = learning_rate / sqrt(gradients[feature, component])
            features[feature, component]  -= <flt>(local_lr * feature_weight * gradient)
            gradients[feature, component] += <flt>((gradient * feature_weight) ** 2)
            features[feature, component]  *= <flt>(1.0 + alpha * local_lr)
            sum_lr += local_lr

    return sum_lr


# ── Logistic gradient step ────────────────────────────────────────────────────

cdef inline void update(double loss,
                        CSRMatrix item_features,
                        CSRMatrix user_features,
                        int user_id,
                        int item_id,
                        flt *user_repr,
                        flt *it_repr,
                        FastAryColBring model,
                        double item_alpha,
                        double user_alpha) nogil:
    """
    Apply the logistic gradient update for a single (user, item) pair.
    """
    cdef int i
    cdef int item_start = item_features.get_row_start(item_id)
    cdef int item_stop  = item_features.get_row_end(item_id)
    cdef int user_start = user_features.get_row_start(user_id)
    cdef int user_stop  = user_features.get_row_end(user_id)
    cdef double avg_lr  = 0.0
    cdef flt item_component, user_component

    avg_lr += update_biases(item_features, item_start, item_stop,
                            model.item_biases, model.item_bias_gradients,
                            model.item_bias_momentum,
                            loss, model.adadelta, model.learning_rate,
                            item_alpha, model.rho, model.eps)

    avg_lr += update_biases(user_features, user_start, user_stop,
                            model.user_biases, model.user_bias_gradients,
                            model.user_bias_momentum,
                            loss, model.adadelta, model.learning_rate,
                            user_alpha, model.rho, model.eps)

    for i in range(model.no_components):
        item_component = it_repr[i]
        user_component = user_repr[i]

        avg_lr += update_features(item_features, model.item_features,
                                  model.item_feature_gradients,
                                  model.item_feature_momentum,
                                  i, item_start, item_stop,
                                  loss * user_component,
                                  model.adadelta, model.learning_rate,
                                  item_alpha, model.rho, model.eps)

        avg_lr += update_features(user_features, model.user_features,
                                  model.user_feature_gradients,
                                  model.user_feature_momentum,
                                  i, user_start, user_stop,
                                  loss * item_component,
                                  model.adadelta, model.learning_rate,
                                  user_alpha, model.rho, model.eps)

    avg_lr /= ((model.no_components + 1) * (item_stop - item_start)
               + (model.no_components + 1) * (user_stop - user_start))

    model.item_scale *= 1.0 + item_alpha * avg_lr
    model.user_scale *= 1.0 + user_alpha * avg_lr


# ── WARP gradient step ────────────────────────────────────────────────────────

cdef inline void warp_update(double loss,
                             CSRMatrix item_features,
                             CSRMatrix user_features,
                             int user_id,
                             int positive_item_id,
                             int negative_item_id,
                             flt *user_repr,
                             flt *pos_it_repr,
                             flt *neg_it_repr,
                             FastAryColBring model,
                             double item_alpha,
                             double user_alpha) nogil:
    """
    Apply the WARP pairwise gradient update.
    Positive item receives a push-up; negative item receives a push-down.
    """
    cdef int i
    cdef int pos_start  = item_features.get_row_start(positive_item_id)
    cdef int pos_stop   = item_features.get_row_end(positive_item_id)
    cdef int neg_start  = item_features.get_row_start(negative_item_id)
    cdef int neg_stop   = item_features.get_row_end(negative_item_id)
    cdef int user_start = user_features.get_row_start(user_id)
    cdef int user_stop  = user_features.get_row_end(user_id)
    cdef double avg_lr  = 0.0
    cdef flt pos_comp, neg_comp, user_comp

    # Bias updates
    avg_lr += update_biases(item_features, pos_start, pos_stop,
                            model.item_biases, model.item_bias_gradients,
                            model.item_bias_momentum,
                            -loss, model.adadelta, model.learning_rate,
                            item_alpha, model.rho, model.eps)

    avg_lr += update_biases(item_features, neg_start, neg_stop,
                            model.item_biases, model.item_bias_gradients,
                            model.item_bias_momentum,
                            loss, model.adadelta, model.learning_rate,
                            item_alpha, model.rho, model.eps)

    avg_lr += update_biases(user_features, user_start, user_stop,
                            model.user_biases, model.user_bias_gradients,
                            model.user_bias_momentum,
                            loss, model.adadelta, model.learning_rate,
                            user_alpha, model.rho, model.eps)

    # Embedding updates
    for i in range(model.no_components):
        user_comp = user_repr[i]
        pos_comp  = pos_it_repr[i]
        neg_comp  = neg_it_repr[i]

        avg_lr += update_features(item_features, model.item_features,
                                  model.item_feature_gradients,
                                  model.item_feature_momentum,
                                  i, pos_start, pos_stop,
                                  -loss * user_comp,
                                  model.adadelta, model.learning_rate,
                                  item_alpha, model.rho, model.eps)

        avg_lr += update_features(item_features, model.item_features,
                                  model.item_feature_gradients,
                                  model.item_feature_momentum,
                                  i, neg_start, neg_stop,
                                  loss * user_comp,
                                  model.adadelta, model.learning_rate,
                                  item_alpha, model.rho, model.eps)

        avg_lr += update_features(user_features, model.user_features,
                                  model.user_feature_gradients,
                                  model.user_feature_momentum,
                                  i, user_start, user_stop,
                                  loss * (neg_comp - pos_comp),
                                  model.adadelta, model.learning_rate,
                                  user_alpha, model.rho, model.eps)

    avg_lr /= ((model.no_components + 1) * (user_stop - user_start)
               + (model.no_components + 1) * (pos_stop - pos_start)
               + (model.no_components + 1) * (neg_stop - neg_start))

    model.item_scale *= 1.0 + item_alpha * avg_lr
    model.user_scale *= 1.0 + user_alpha * avg_lr
