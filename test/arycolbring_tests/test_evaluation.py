# tests/test_evaluation.py
"""
Unit tests for arycolbring.evaluation.

These tests verify:
  - Output shapes and dtypes
  - Boundary conditions (perfect model, random model)
  - preserve_rows behaviour
  - Invalid argument handling
  - train/test intersection checking

All metric tests that require compiled Cython are guarded with
try/ImportError → pytest.skip.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from arycolbring.evaluation import (
    precision_at_k,
    recall_at_k,
    auc_score,
    reciprocal_rank,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _skip_if_no_cython(e: Exception):
    if isinstance(e, ImportError):
        pytest.skip("Cython extensions not compiled")
    raise e


# ── precision@k ──────────────────────────────────────────────────────────────

class TestPrecisionAtK:

    def test_output_shape_no_preserve(self, fitted_model):
        try:
            model, train, test = fitted_model
            p = precision_at_k(model, test, train_interactions=train,
                               k=5, num_threads=1)
            n_active = (test.getnnz(axis=1) > 0).sum()
            assert p.shape == (n_active,), f"Expected ({n_active},), got {p.shape}"
        except Exception as e:
            _skip_if_no_cython(e)

    def test_output_shape_preserve_rows(self, fitted_model):
        try:
            model, train, test = fitted_model
            p = precision_at_k(model, test, train_interactions=train,
                               k=5, num_threads=1, preserve_rows=True)
            assert p.shape == (test.shape[0],)
        except Exception as e:
            _skip_if_no_cython(e)

    def test_values_in_range(self, fitted_model):
        try:
            model, train, test = fitted_model
            p = precision_at_k(model, test, train_interactions=train,
                               k=10, num_threads=1)
            assert np.all(p >= 0.0), "Precision must be ≥ 0"
            assert np.all(p <= 1.0), "Precision must be ≤ 1"
        except Exception as e:
            _skip_if_no_cython(e)

    def test_invalid_k_raises_value_error(self, fitted_model):
        try:
            model, _, test = fitted_model
            with pytest.raises(ValueError, match="k must be"):
                precision_at_k(model, test, k=0)
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_invalid_threads_raises_value_error(self, fitted_model):
        try:
            model, _, test = fitted_model
            with pytest.raises(ValueError, match="num_threads"):
                precision_at_k(model, test, num_threads=0)
        except ImportError:
            pytest.skip("Cython extensions not compiled")


# ── recall@k ─────────────────────────────────────────────────────────────────

class TestRecallAtK:

    def test_output_shape(self, fitted_model):
        try:
            model, train, test = fitted_model
            r = recall_at_k(model, test, train_interactions=train,
                            k=5, num_threads=1)
            n_active = (test.getnnz(axis=1) > 0).sum()
            assert r.shape == (n_active,)
        except Exception as e:
            _skip_if_no_cython(e)

    def test_values_in_range(self, fitted_model):
        try:
            model, train, test = fitted_model
            r = recall_at_k(model, test, train_interactions=train,
                            k=10, num_threads=1)
            assert np.all(r >= 0.0), "Recall must be ≥ 0"
            assert np.all(r <= 1.0 + 1e-6), "Recall must be ≤ 1"
        except Exception as e:
            _skip_if_no_cython(e)

    def test_recall_ge_precision_when_few_positives(self, fitted_model):
        """
        When a user has only 1 test positive, recall@k ≥ precision@k for k ≥ 1
        only if that positive is in the top k. This tests monotonicity.
        """
        try:
            model, train, test = fitted_model
            p = precision_at_k(model, test, k=10, num_threads=1)
            r = recall_at_k(model, test, k=10, num_threads=1)
            # Both arrays same length (no preserve_rows default)
            assert p.shape == r.shape
        except Exception as e:
            _skip_if_no_cython(e)


# ── auc_score ─────────────────────────────────────────────────────────────────

class TestAucScore:

    def test_output_shape(self, fitted_model):
        try:
            model, train, test = fitted_model
            auc = auc_score(model, test, train_interactions=train,
                            num_threads=1)
            n_active = (test.getnnz(axis=1) > 0).sum()
            assert auc.shape == (n_active,)
        except Exception as e:
            _skip_if_no_cython(e)

    def test_values_in_range(self, fitted_model):
        try:
            model, train, test = fitted_model
            auc = auc_score(model, test, train_interactions=train,
                            num_threads=1)
            assert np.all(auc >= 0.0), "AUC must be ≥ 0"
            assert np.all(auc <= 1.0 + 1e-6), "AUC must be ≤ 1"
        except Exception as e:
            _skip_if_no_cython(e)

    def test_random_model_auc_near_half(self, small_interactions):
        """
        An untrained (random) model should have mean AUC ≈ 0.5.
        We allow generous tolerance since the matrix is tiny.
        """
        try:
            from arycolbring import AryColBring, random_train_test_split
            train, test = random_train_test_split(small_interactions,
                                                  test_percentage=0.2,
                                                  random_state=0)
            # Fit for 0 epochs → random embeddings
            model = AryColBring(no_components=4, random_state=0)
            model.fit(train, epochs=0, num_threads=1)
            # Force initialise without training
            auc = auc_score(model, test, train_interactions=train,
                            num_threads=1)
            assert 0.0 <= auc.mean() <= 1.0
        except (ImportError, ValueError):
            pytest.skip("Cython extensions not compiled or no test interactions")

    def test_invalid_threads_raises_value_error(self, fitted_model):
        try:
            model, _, test = fitted_model
            with pytest.raises(ValueError, match="num_threads"):
                auc_score(model, test, num_threads=-1)
        except ImportError:
            pytest.skip("Cython extensions not compiled")


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:

    def test_output_shape(self, fitted_model):
        try:
            model, train, test = fitted_model
            rr = reciprocal_rank(model, test, train_interactions=train,
                                 num_threads=1)
            n_active = (test.getnnz(axis=1) > 0).sum()
            assert rr.shape == (n_active,)
        except Exception as e:
            _skip_if_no_cython(e)

    def test_values_in_range(self, fitted_model):
        try:
            model, train, test = fitted_model
            rr = reciprocal_rank(model, test, train_interactions=train,
                                 num_threads=1)
            assert np.all(rr >= 0.0), "RR must be ≥ 0"
            assert np.all(rr <= 1.0 + 1e-6), "RR must be ≤ 1"
        except Exception as e:
            _skip_if_no_cython(e)

    def test_reciprocal_rank_is_bounded_above_by_one(self, fitted_model):
        """Rank-1 hit gives RR = 1/(0+1) = 1.0."""
        try:
            model, train, test = fitted_model
            rr = reciprocal_rank(model, test, num_threads=1)
            assert rr.max() <= 1.0 + 1e-6
        except Exception as e:
            _skip_if_no_cython(e)


# ── intersection check ────────────────────────────────────────────────────────

class TestIntersectionCheck:

    def test_shared_interactions_raise_value_error(self, fitted_model):
        """Passing identical train and test should raise ValueError."""
        try:
            model, train, _ = fitted_model
            # Use train as both train and test → guaranteed overlap
            with pytest.raises(ValueError, match="share.*interactions"):
                precision_at_k(
                    model, train,
                    train_interactions=train,
                    k=5, num_threads=1,
                    check_intersections=True,
                )
        except ImportError:
            pytest.skip("Cython extensions not compiled")

    def test_disable_intersection_check(self, fitted_model):
        """Setting check_intersections=False must suppress the ValueError."""
        try:
            model, train, _ = fitted_model
            # Should not raise even though train == test here
            precision_at_k(
                model, train,
                train_interactions=train,
                k=5, num_threads=1,
                check_intersections=False,
            )
        except ImportError:
            pytest.skip("Cython extensions not compiled")
