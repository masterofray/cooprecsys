# tests/test_cross_validation.py
"""Unit tests for arycolbring.cross_validation."""

import numpy as np
import pytest
import scipy.sparse as sp

from arycolbring.cross_validation import (
    random_train_test_split,
    user_based_train_test_split,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dense_nnz(mat):
    """Return total count of non-zero entries after COO → CSR sum-dup."""
    return mat.tocsr().nnz


# ── random_train_test_split ───────────────────────────────────────────────────

class TestRandomTrainTestSplit:

    def test_shapes_preserved(self, small_interactions):
        train, test = random_train_test_split(small_interactions,
                                              test_percentage=0.2,
                                              random_state=0)
        assert train.shape == small_interactions.shape
        assert test.shape  == small_interactions.shape

    def test_nnz_sums_to_original(self, small_interactions):
        train, test = random_train_test_split(small_interactions,
                                              test_percentage=0.2,
                                              random_state=0)
        # The split is on raw COO entries (may contain duplicates)
        assert train.nnz + test.nnz == small_interactions.nnz

    def test_approximate_split_ratio(self, medium_interactions):
        test_pct = 0.25
        train, test = random_train_test_split(medium_interactions,
                                              test_percentage=test_pct,
                                              random_state=1)
        actual_pct = test.nnz / medium_interactions.nnz
        assert abs(actual_pct - test_pct) < 0.02, (
            f"Expected ~{test_pct:.2f}, got {actual_pct:.4f}"
        )

    def test_reproducible_with_same_seed(self, small_interactions):
        t1, e1 = random_train_test_split(small_interactions, random_state=7)
        t2, e2 = random_train_test_split(small_interactions, random_state=7)
        np.testing.assert_array_equal(t1.data, t2.data)
        np.testing.assert_array_equal(e1.data, e2.data)

    def test_different_seeds_differ(self, small_interactions):
        _, e1 = random_train_test_split(small_interactions, random_state=1)
        _, e2 = random_train_test_split(small_interactions, random_state=999)
        assert not np.array_equal(e1.row, e2.row), (
            "Different seeds should produce different splits"
        )

    def test_non_sparse_raises_type_error(self):
        with pytest.raises(TypeError):
            random_train_test_split(np.eye(5))

    def test_invalid_percentage_raises_value_error(self, small_interactions):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            random_train_test_split(small_interactions, test_percentage=0.0)
        with pytest.raises(ValueError):
            random_train_test_split(small_interactions, test_percentage=1.0)
        with pytest.raises(ValueError):
            random_train_test_split(small_interactions, test_percentage=1.5)

    def test_empty_matrix_raises_runtime_error(self):
        empty = sp.coo_matrix((10, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="no non-zero entries"):
            random_train_test_split(empty)

    def test_output_dtype_preserved(self, small_interactions):
        train, test = random_train_test_split(small_interactions,
                                              random_state=0)
        assert train.dtype == small_interactions.dtype
        assert test.dtype  == small_interactions.dtype

    def test_random_state_object_accepted(self, small_interactions):
        rs = np.random.RandomState(42)
        train, test = random_train_test_split(small_interactions,
                                              random_state=rs)
        assert train.nnz + test.nnz == small_interactions.nnz


# ── user_based_train_test_split ───────────────────────────────────────────────

class TestUserBasedTrainTestSplit:

    def test_shapes_preserved(self, small_interactions):
        train, test = user_based_train_test_split(small_interactions,
                                                  test_percentage=0.2,
                                                  random_state=0)
        assert train.shape == small_interactions.shape
        assert test.shape  == small_interactions.shape

    def test_nnz_sums_to_original(self, small_interactions):
        train, test = user_based_train_test_split(small_interactions,
                                                  test_percentage=0.2,
                                                  random_state=0)
        assert train.nnz + test.nnz == small_interactions.nnz

    def test_users_with_one_interaction_stay_in_train(self):
        """A user with a single interaction should never end up in test."""
        rows = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        cols = np.array([0, 0, 1, 0, 1, 2], dtype=np.int32)
        data = np.ones(6, dtype=np.float32)
        mat  = sp.coo_matrix((data, (rows, cols)), shape=(3, 5))

        train, test = user_based_train_test_split(mat,
                                                  test_percentage=0.5,
                                                  random_state=0)
        # User 0 has exactly 1 interaction → must appear only in train
        test_csr = test.tocsr()
        assert test_csr[0, :].nnz == 0, (
            "User with 1 interaction must not appear in test set"
        )

    def test_non_sparse_raises_type_error(self):
        with pytest.raises(TypeError):
            user_based_train_test_split(np.eye(5))

    def test_invalid_percentage_raises_value_error(self, small_interactions):
        with pytest.raises(ValueError):
            user_based_train_test_split(small_interactions, test_percentage=0.0)
        with pytest.raises(ValueError):
            user_based_train_test_split(small_interactions, test_percentage=1.5)
