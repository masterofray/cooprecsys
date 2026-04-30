# tests/test_data_utils.py
"""Unit tests for arycolbring.data_utils."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from arycolbring.data_utils import (
    load_interactions_from_df,
    describe_interactions,
    validate_sparse_matrix,
)


# ── load_interactions_from_df ─────────────────────────────────────────────────

class TestLoadInteractionsFromDf:

    def _make_df(self):
        return pd.DataFrame({
            "user": ["u1", "u2", "u1", "u3", "u2"],
            "item": ["i1", "i2", "i3", "i1", "i3"],
            "rating": [1.0, 2.0, 3.0, 1.0, 0.5],
        })

    def test_basic_shape(self):
        df = self._make_df()
        mat, users, items = load_interactions_from_df(
            df, user_col="user", item_col="item"
        )
        assert mat.shape == (3, 3), f"Expected (3,3), got {mat.shape}"
        assert len(users) == 3
        assert len(items) == 3

    def test_with_rating_col(self):
        df = self._make_df()
        mat, _, _ = load_interactions_from_df(
            df, user_col="user", item_col="item", rating_col="rating"
        )
        assert mat.dtype == np.float32
        # data should reflect actual ratings, not just ones
        assert not np.all(mat.data == 1.0)

    def test_default_rating_is_one(self):
        df = self._make_df()
        mat, _, _ = load_interactions_from_df(
            df, user_col="user", item_col="item", rating_col=None
        )
        assert np.all(mat.data == 1.0)

    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame({"user": ["u1"], "item": ["i1"]})
        with pytest.raises(ValueError, match="missing columns"):
            load_interactions_from_df(df, user_col="user", item_col="nonexistent")

    def test_output_is_coo_matrix(self):
        df = self._make_df()
        mat, _, _ = load_interactions_from_df(
            df, user_col="user", item_col="item"
        )
        assert sp.issparse(mat)

    def test_nnz_equals_unique_pairs(self):
        """Duplicate (user, item) pairs are summed in COO (scipy default)."""
        df = pd.DataFrame({
            "user": ["u1", "u1"],
            "item": ["i1", "i1"],
        })
        mat, _, _ = load_interactions_from_df(
            df, user_col="user", item_col="item"
        )
        # COO allows duplicates; converting to CSR sums them
        mat_csr = mat.tocsr()
        assert mat_csr[0, 0] == pytest.approx(2.0)


# ── describe_interactions ─────────────────────────────────────────────────────

class TestDescribeInteractions:

    def test_columns_present(self, small_interactions):
        summary = describe_interactions(small_interactions)
        expected = {
            "n_users", "n_items", "nnz", "density",
            "avg_interactions_per_user",
            "min_interactions_per_user",
            "max_interactions_per_user",
        }
        assert expected.issubset(set(summary.columns))

    def test_n_users_matches(self, small_interactions):
        summary = describe_interactions(small_interactions)
        assert summary["n_users"].iloc[0] == small_interactions.shape[0]

    def test_n_items_matches(self, small_interactions):
        summary = describe_interactions(small_interactions)
        assert summary["n_items"].iloc[0] == small_interactions.shape[1]

    def test_density_in_range(self, small_interactions):
        summary = describe_interactions(small_interactions)
        d = summary["density"].iloc[0]
        assert 0.0 <= d <= 1.0


# ── validate_sparse_matrix ────────────────────────────────────────────────────

class TestValidateSparseMatrix:

    def test_valid_matrix_passes(self, small_interactions):
        validate_sparse_matrix(small_interactions, "test")  # should not raise

    def test_non_sparse_raises_type_error(self):
        with pytest.raises(TypeError, match="scipy sparse"):
            validate_sparse_matrix(np.eye(3), "bad")

    def test_nan_raises_value_error(self):
        rows = np.array([0, 1], dtype=np.int32)
        cols = np.array([0, 1], dtype=np.int32)
        data = np.array([np.nan, 1.0], dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(3, 3))
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_sparse_matrix(mat, "nan_mat")

    def test_inf_raises_value_error(self):
        rows = np.array([0], dtype=np.int32)
        cols = np.array([0], dtype=np.int32)
        data = np.array([np.inf], dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_sparse_matrix(mat, "inf_mat")

    def test_degenerate_shape_raises_runtime_error(self):
        mat = sp.coo_matrix((0, 5), dtype=np.float32)
        with pytest.raises(RuntimeError, match="degenerate shape"):
            validate_sparse_matrix(mat, "zero_rows")

    def test_empty_nnz_warns_but_passes(self, caplog):
        import logging
        mat = sp.coo_matrix((10, 10), dtype=np.float32)
        with caplog.at_level(logging.WARNING):
            validate_sparse_matrix(mat, "empty")
        assert any("zero non-zero" in r.message for r in caplog.records)
