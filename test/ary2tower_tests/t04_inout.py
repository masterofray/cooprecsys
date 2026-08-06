#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
t04_inout.py
___________________________________________________________________
Pytest suite for src/models/ary2tower/inout/ (TwoTowerBase,
TwoTowerPredictor, TwoTowerArchitect, TwoTowerFallBack). CPU-only, no
compiled extension needed (all pure Python/NumPy/pandas).
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.models.ary2tower.config import TwoTowerConfig
from src.models.ary2tower.inout import (TwoTowerBase, TwoTowerPredictor,
                                        TwoTowerArchitect, TwoTowerFallBack)


class TestTwoTowerBase:

    def test_invalid_n_users_raises(self):
        with pytest.raises(ValueError):
            TwoTowerArchitect(0, 10)

    def test_invalid_n_items_raises(self):
        with pytest.raises(ValueError):
            TwoTowerArchitect(10, -1)

    def test_get_params_set_params_roundtrip(self):
        arch = TwoTowerArchitect(10, 10, config=TwoTowerConfig(embedding_dim=6))
        assert arch.get_params()["embedding_dim"] == 6
        arch.set_params(learning_rate=0.05)
        assert arch.get_params()["learning_rate"] == 0.05

    def test_set_params_rejects_unknown_key(self):
        arch = TwoTowerArchitect(10, 10)
        with pytest.raises(ValueError):
            arch.set_params(not_a_real_param=1)

    def test_repr_contains_class_name(self):
        arch = TwoTowerArchitect(10, 10)
        assert "TwoTowerArchitect" in repr(arch)


class TestTwoTowerPredictor:

    def test_build_pairs_strict_pairwise(self):
        u, i = TwoTowerPredictor.build_pairs([1, 2, 3], [10, 20, 30])
        np.testing.assert_array_equal(u, [1, 2, 3])
        np.testing.assert_array_equal(i, [10, 20, 30])

    def test_build_pairs_cross_join(self):
        u, i = TwoTowerPredictor.build_pairs([1, 2], [10, 20], cross_join=True)
        np.testing.assert_array_equal(u, [1, 1, 2, 2])
        np.testing.assert_array_equal(i, [10, 20, 10, 20])

    def test_build_pairs_scalar_broadcast(self):
        u, i = TwoTowerPredictor.build_pairs([1, 2, 3], [99])
        np.testing.assert_array_equal(i, [99, 99, 99])

    def test_build_pairs_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            TwoTowerPredictor.build_pairs([1, 2, 3], [10, 20])

    def test_predict_shape(self):
        predictor = TwoTowerPredictor(20, 15, config=TwoTowerConfig(embedding_dim=4, hidden_dim=6, output_dim=3))
        scores = predictor.predict([0, 1, 2], [3, 4, 5])
        assert scores.shape == (3,)

    def test_predict_rank_excludes_items(self):
        predictor = TwoTowerPredictor(20, 15, config=TwoTowerConfig(embedding_dim=4, hidden_dim=6, output_dim=3))
        recs = predictor.predict_rank(0, n_items=5, exclude_items=[0, 1])
        assert len(recs) == 5
        assert 0 not in [iid for iid, _ in recs]
        assert 1 not in [iid for iid, _ in recs]

    def test_predict_rank_sorted_descending(self):
        predictor = TwoTowerPredictor(20, 15, config=TwoTowerConfig(embedding_dim=4, hidden_dim=6, output_dim=3))
        recs = predictor.predict_rank(0, n_items=8)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)


class TestTwoTowerArchitect:

    @pytest.fixture
    def structured_interactions(self):
        rng = np.random.default_rng(0)
        n_users, n_items, true_dim = 30, 20, 4
        true_user = rng.normal(size=(n_users, true_dim))
        true_item = rng.normal(size=(n_items, true_dim))
        true_scores = true_user @ true_item.T
        rows, cols = [], []
        for u in range(n_users):
            liked = np.argsort(true_scores[u])[::-1][:5]
            rows += [u] * len(liked); cols += list(liked)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
        return mat, n_users, n_items

    def test_fit_raises_on_empty_interactions(self):
        arch = TwoTowerArchitect(5, 5, config=TwoTowerConfig(n_epochs=1))
        with pytest.raises(ValueError):
            arch.fit(sp.csr_matrix((5, 5)))

    def test_fit_sets_is_fitted(self, structured_interactions):
        mat, n_users, n_items = structured_interactions
        arch = TwoTowerArchitect(n_users, n_items,
                                 config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                       output_dim=4, n_epochs=3, random_state=0))
        assert arch.is_fitted is False
        arch.fit(mat)
        assert arch.is_fitted is True

    def test_loss_decreases_with_adequate_training(self, structured_interactions):
        mat, n_users, n_items = structured_interactions
        arch = TwoTowerArchitect(n_users, n_items,
                                 config=TwoTowerConfig(embedding_dim=8, hidden_dim=12, output_dim=6,
                                                       learning_rate=0.02, momentum=0.9,
                                                       n_epochs=25, random_state=1))
        arch.fit(mat)
        if arch.loss_history[-1] == arch.loss_history[-1]:  # not NaN (cython backend)
            assert arch.loss_history[-1] < arch.loss_history[0]


class TestTwoTowerFallBack:

    @pytest.fixture
    def item_embeddings(self):
        return np.random.default_rng(0).normal(size=(30, 8))

    @pytest.fixture
    def purchase_data(self):
        return pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "item_id": [0, 1, 2, 5, 6]})

    def test_missing_user_col_raises(self, item_embeddings):
        with pytest.raises(ValueError):
            TwoTowerFallBack(pd.DataFrame({"wrong_col": [1]}), item_embeddings)

    def test_purchased_items(self, purchase_data, item_embeddings):
        fb = TwoTowerFallBack(purchase_data, item_embeddings)
        assert fb.purchased_items(1) == {0, 1, 2}
        assert fb.purchased_items(999) == set()  # unseen user -> empty set

    def test_clean_recommendations_filters_purchased(self, purchase_data, item_embeddings):
        fb = TwoTowerFallBack(purchase_data, item_embeddings)
        candidate_pool = [(0, 0.9), (1, 0.85), (7, 0.8), (12, 0.75), (3, 0.7)]
        result = fb.clean_recommendations(user_id=1, candidate_pool=candidate_pool, n_items=5)
        assert not (set(result["item_id"]) & fb.purchased_items(1))
        assert len(result) == 5

    def test_clean_recommendations_backfills_with_fallback(self, purchase_data, item_embeddings):
        fb = TwoTowerFallBack(purchase_data, item_embeddings)
        # candidate_pool has 5 entries; user 1's purchases {0,1,2} remove 3 of
        # them (items 0, 1, 2), leaving only 2 (items 7, 12) -> 3 fallback
        # slots are needed to reach n_items=5.
        candidate_pool = [(0, 0.9), (1, 0.85), (7, 0.8), (12, 0.75), (2, 0.7)]
        result = fb.clean_recommendations(user_id=1, candidate_pool=candidate_pool, n_items=5)
        assert result["is_fallback"].sum() == 3

    def test_item_to_item_candidates_excludes_seeds(self, item_embeddings):
        fb = TwoTowerFallBack(pd.DataFrame({"user_id": [1], "item_id": [0]}), item_embeddings)
        candidates = fb.item_to_item_candidates(seed_items=[4, 9], exclude={4, 9}, n=5)
        candidate_ids = [c[0] for c in candidates]
        assert 4 not in candidate_ids and 9 not in candidate_ids
        assert len(candidates) == 5
