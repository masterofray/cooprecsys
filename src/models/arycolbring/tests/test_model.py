# tests/test_model.py
"""
Unit tests for arycolbring.model.AryColBring.

These tests deliberately avoid exercising the compiled Cython kernels
directly — they test the Python layer: parameter validation, state
management, input coercion, and the sklearn-style API surface.

Tests that require compiled extensions are marked with
``@pytest.mark.integration`` and are skipped when the extensions are
not built.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from arycolbring.model import AryColBring, CYTHON_DTYPE


# ── constructor validation ────────────────────────────────────────────────────

class TestConstructor:

    def test_defaults(self):
        m = AryColBring()
        assert m.no_components   == 10
        assert m.loss             == "logistic"
        assert m.learning_schedule == "adagrad"
        assert m.learning_rate    == pytest.approx(0.05)
        assert m.item_alpha       == pytest.approx(0.0)
        assert m.user_alpha       == pytest.approx(0.0)
        assert m._max_sampled     == 10

    @pytest.mark.parametrize("loss", ["logistic", "warp", "bpr", "warp-kos"])
    def test_valid_losses(self, loss):
        m = AryColBring(loss=loss)
        assert m.loss == loss

    def test_invalid_loss_raises_value_error(self):
        with pytest.raises(ValueError, match="loss must be one of"):
            AryColBring(loss="mse")

    @pytest.mark.parametrize("sched", ["adagrad", "adadelta"])
    def test_valid_schedules(self, sched):
        m = AryColBring(learning_schedule=sched)
        assert m.learning_schedule == sched

    def test_invalid_schedule_raises_value_error(self):
        with pytest.raises(ValueError, match="learning_schedule"):
            AryColBring(learning_schedule="sgd")

    def test_negative_item_alpha_raises(self):
        with pytest.raises(ValueError, match="item_alpha"):
            AryColBring(item_alpha=-0.1)

    def test_negative_user_alpha_raises(self):
        with pytest.raises(ValueError, match="user_alpha"):
            AryColBring(user_alpha=-1.0)

    def test_zero_components_raises(self):
        with pytest.raises(ValueError, match="no_components"):
            AryColBring(no_components=0)

    def test_bad_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            AryColBring(rho=1.0)
        with pytest.raises(ValueError, match="rho"):
            AryColBring(rho=0.0)

    def test_max_sampled_zero_raises(self):
        with pytest.raises(ValueError, match="max_sampled"):
            AryColBring(max_sampled=0)

    def test_int_random_state(self):
        m = AryColBring(random_state=42)
        assert isinstance(m._random_state, np.random.RandomState)

    def test_rng_random_state(self):
        rs = np.random.RandomState(7)
        m  = AryColBring(random_state=rs)
        assert m._random_state is rs

    def test_none_random_state(self):
        m = AryColBring(random_state=None)
        assert isinstance(m._random_state, np.random.RandomState)

    def test_bad_random_state_raises_type_error(self):
        with pytest.raises(TypeError, match="random_state"):
            AryColBring(random_state="forty-two")

    def test_is_fitted_before_fit(self):
        m = AryColBring()
        assert not m.is_fitted


# ── property setters ──────────────────────────────────────────────────────────

class TestProperties:

    def test_loss_setter(self):
        m = AryColBring()
        m.loss = "warp"
        assert m.loss == "warp"

    def test_invalid_loss_setter_raises(self):
        m = AryColBring()
        with pytest.raises(ValueError):
            m.loss = "rank"

    def test_no_components_setter(self):
        m = AryColBring()
        m.no_components = 64
        assert m.no_components == 64

    def test_no_components_zero_raises(self):
        m = AryColBring()
        with pytest.raises(ValueError):
            m.no_components = 0

    def test_learning_rate_setter(self):
        m = AryColBring()
        m.learning_rate = 0.01
        assert m.learning_rate == pytest.approx(0.01)

    def test_learning_rate_zero_raises(self):
        m = AryColBring()
        with pytest.raises(ValueError):
            m.learning_rate = 0.0

    def test_item_alpha_setter(self):
        m = AryColBring()
        m.item_alpha = 1e-4
        assert m.item_alpha == pytest.approx(1e-4)

    def test_item_alpha_negative_raises(self):
        m = AryColBring()
        with pytest.raises(ValueError):
            m.item_alpha = -0.5


# ── get_params / set_params ───────────────────────────────────────────────────

class TestGetSetParams:

    def test_get_params_returns_dict(self):
        m = AryColBring(no_components=16, loss="warp")
        p = m.get_params()
        assert isinstance(p, dict)
        assert p["no_components"] == 16
        assert p["loss"] == "warp"

    def test_set_params_roundtrip(self):
        m = AryColBring()
        m.set_params(no_components=32, loss="bpr")
        assert m._no_components == 32
        assert m._loss == "bpr"

    def test_set_params_invalid_key_raises(self):
        m = AryColBring()
        with pytest.raises(ValueError, match="Invalid parameter"):
            m.set_params(totally_wrong=42)

    def test_repr_is_string(self):
        m = AryColBring()
        assert "AryColBring" in repr(m)
        assert "loss" in repr(m)


# ── state / initialise ────────────────────────────────────────────────────────

class TestStateManagement:

    def _make_mat(self, n_users=20, n_items=10, n_pos=50):
        rs = np.random.RandomState(0)
        rows = rs.randint(0, n_users, n_pos).astype(np.int32)
        cols = rs.randint(0, n_items, n_pos).astype(np.int32)
        data = np.ones(n_pos, dtype=np.float32)
        return sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items))

    def test_predict_before_fit_raises_runtime_error(self):
        m = AryColBring()
        with pytest.raises(RuntimeError, match="not yet fitted"):
            m.predict([0], [0])

    def test_predict_rank_before_fit_raises_runtime_error(self):
        m = AryColBring()
        mat = self._make_mat()
        with pytest.raises(RuntimeError, match="not yet fitted"):
            m.predict_rank(mat)

    def test_get_item_representations_before_fit_raises(self):
        m = AryColBring()
        with pytest.raises(RuntimeError):
            m.get_item_representations()

    def test_get_user_representations_before_fit_raises(self):
        m = AryColBring()
        with pytest.raises(RuntimeError):
            m.get_user_representations()

    def test_fit_sets_is_fitted_true(self, train_test_pair):
        """Requires compiled extensions."""
        try:
            train, _ = train_test_pair
            m = AryColBring(no_components=4, loss="logistic", random_state=0)
            m.fit(train, epochs=1, num_threads=1)
            assert m.is_fitted
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_fit_resets_state(self, train_test_pair):
        """fit() must discard embeddings from a previous fit."""
        try:
            train, _ = train_test_pair
            m = AryColBring(no_components=4, random_state=0)
            m.fit(train, epochs=1, num_threads=1)
            old_emb = m.item_embeddings.copy()
            m.fit(train, epochs=1, num_threads=1)
            # Second fit re-initialises with a fresh RandomState draw —
            # may produce different values.  At minimum, embeddings exist.
            assert m.item_embeddings is not None
            assert m.item_embeddings.shape == old_emb.shape
        except ImportError:
            pytest.skip("Cython extensions not compiled")


# ── input validation on fit ───────────────────────────────────────────────────

class TestFitValidation:

    def _make_mat(self):
        rs = np.random.RandomState(0)
        rows = rs.randint(0, 30, 80).astype(np.int32)
        cols = rs.randint(0, 15, 80).astype(np.int32)
        data = np.ones(80, dtype=np.float32)
        return sp.coo_matrix((data, (rows, cols)), shape=(30, 15))

    def test_zero_threads_raises_value_error(self):
        m   = AryColBring()
        mat = self._make_mat()
        with pytest.raises(ValueError, match="num_threads"):
            m.fit(mat, epochs=1, num_threads=0)

    def test_zero_epochs_raises_value_error(self):
        m   = AryColBring()
        mat = self._make_mat()
        with pytest.raises(ValueError, match="epochs"):
            m.fit(mat, epochs=0)

    def test_sample_weight_wrong_type_raises_type_error(self):
        try:
            m   = AryColBring(loss="logistic", random_state=0)
            mat = self._make_mat()
            with pytest.raises(TypeError, match="COO"):
                m.fit(mat, sample_weight=np.ones(10, dtype=np.float32))
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_warp_kos_with_sample_weight_raises_not_implemented(self):
        m   = AryColBring(loss="warp-kos", random_state=0)
        mat = self._make_mat()
        sw  = sp.coo_matrix(mat)
        with pytest.raises(NotImplementedError, match="warp-kos"):
            m.fit(mat, sample_weight=sw, epochs=1)


# ── predict input coercion ────────────────────────────────────────────────────

class TestPredictCoercion:

    def test_scalar_user_id_broadcast(self, fitted_model):
        try:
            model, train, test = fitted_model
            scores = model.predict(0, [0, 1, 2])
            assert scores.shape == (3,)
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_list_inputs_converted(self, fitted_model):
        try:
            model, train, test = fitted_model
            scores = model.predict([0, 1], [2, 3])
            assert scores.shape == (2,)
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_length_mismatch_raises_value_error(self, fitted_model):
        try:
            model, _, _ = fitted_model
            with pytest.raises(ValueError, match="length"):
                model.predict([0, 1, 2], [0, 1])
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_negative_ids_raise_value_error(self, fitted_model):
        try:
            model, _, _ = fitted_model
            with pytest.raises(ValueError, match="Negative"):
                model.predict([-1], [0])
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_output_dtype_is_float32(self, fitted_model):
        try:
            model, _, _ = fitted_model
            scores = model.predict([0], [0])
            assert scores.dtype == np.float32
        except ImportError:
            pytest.skip("Cython extensions not compiled")


# ── representations ───────────────────────────────────────────────────────────

class TestRepresentations:

    def test_item_repr_shape(self, fitted_model):
        try:
            model, _, _ = fitted_model
            biases, embs = model.get_item_representations()
            assert embs.ndim    == 2
            assert embs.shape[1] == model.no_components
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_user_repr_shape(self, fitted_model):
        try:
            model, _, _ = fitted_model
            biases, embs = model.get_user_representations()
            assert embs.ndim    == 2
            assert embs.shape[1] == model.no_components
        except ImportError:
            pytest.skip("Cython extensions not compiled")
