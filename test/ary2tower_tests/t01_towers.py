#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-30"

"""
t01_towers.py
___________________________________________________________________
Pytest suite for src/models/ary2tower/.

Runs against whichever backend is active (the compiled CLtowers
extension if it's been built via cysetup.py, otherwise the NumPy
fallback in towers.py) -- ary2tower.towers._HAS_CYTHON tells you which
one is live. In this repo's own sandbox during development, only the
NumPy path could be exercised (no C/Cython toolchain available); these
tests will exercise the compiled path automatically in any environment
where `python cysetup.py build_ext --inplace` has been run first.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.cooprecsys.models.ary2tower        import (TwoTowerConfig, TwoTowerTrainer,
                                                    TwoTowerInference, UserTower, ItemTower,
                                                    TwoTowerWeights)
from src.cooprecsys.models.ary2tower.towers import dot_product_similarity, cosine_similarity


# ================================================================
# TwoTowerConfig
# ================================================================

class TestTwoTowerConfig:

    def test_valid_config(self):
        cfg = TwoTowerConfig(embedding_dim=8, hidden_dim=16, output_dim=4)
        assert cfg.embedding_dim == 8
        assert cfg.get_params()["hidden_dim"] == 16

    @pytest.mark.parametrize("kwargs", [
        {"embedding_dim": 0}, {"embedding_dim": -1},
        {"hidden_dim": 0}, {"output_dim": 0},
        {"learning_rate": 0.0}, {"learning_rate": -0.1},
        {"momentum": 1.0}, {"momentum": -0.1},
        {"n_epochs": 0}, {"num_threads": 0},
    ])
    def test_invalid_config_raises_value_error(self, kwargs):
        with pytest.raises(ValueError):
            TwoTowerConfig(**kwargs)

    def test_invalid_random_state_type_raises(self):
        with pytest.raises(TypeError):
            TwoTowerConfig(random_state="not-a-seed")


# ================================================================
# TwoTowerWeights / UserTower / ItemTower
# ================================================================

class TestTowers:

    @pytest.fixture
    def weights(self):
        return TwoTowerWeights(n_users=20, n_items=15, embedding_dim=6,
                               hidden_dim=10, output_dim=4, random_state=0)

    def test_weight_shapes(self, weights):
        assert weights.user_embeddings.shape == (20, 6)
        assert weights.user_w1.shape == (6, 10)
        assert weights.user_w2.shape == (10, 4)
        assert weights.item_embeddings.shape == (15, 6)

    def test_user_tower_forward_shape(self, weights):
        tower = UserTower(weights)
        out = tower.forward([0, 1, 2])
        assert out.shape == (3, 4)

    def test_item_tower_forward_shape(self, weights):
        tower = ItemTower(weights)
        out = tower.forward(np.array([0, 5, 10]))
        assert out.shape == (3, 4)

    def test_forward_is_deterministic_for_fixed_weights(self, weights):
        tower = UserTower(weights)
        out1 = tower.forward([3])
        out2 = tower.forward([3])
        np.testing.assert_array_almost_equal(out1, out2)

    def test_dot_product_similarity_shape_and_value(self):
        a = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        sims = dot_product_similarity(a, b)
        np.testing.assert_array_almost_equal(sims, [1.0, 2.0])

    def test_cosine_similarity_identical_vectors_is_one(self):
        a = np.array([[3.0, 4.0]], dtype=np.float32)
        sims = cosine_similarity(a, a)
        assert sims[0] == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal_is_zero(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sims = cosine_similarity(a, b)
        assert sims[0] == pytest.approx(0.0, abs=1e-5)


# ================================================================
# TwoTowerTrainer
# ================================================================

class TestTwoTowerTrainer:

    @pytest.fixture
    def structured_interactions(self):
        """Interactions with real latent structure (not pure noise), so
        a trained model can be checked against a random baseline."""
        rng = np.random.default_rng(0)
        n_users, n_items, true_dim = 40, 20, 4
        true_user = rng.normal(size=(n_users, true_dim))
        true_item = rng.normal(size=(n_items, true_dim))
        true_scores = true_user @ true_item.T
        rows, cols = [], []
        for u in range(n_users):
            liked = np.argsort(true_scores[u])[::-1][:5]
            rows += [u] * len(liked)
            cols += list(liked)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
        return mat, true_scores, n_users, n_items

    def test_fit_raises_on_empty_interactions(self):
        trainer = TwoTowerTrainer(5, 5, config=TwoTowerConfig(n_epochs=1))
        empty = sp.csr_matrix((5, 5))
        with pytest.raises(ValueError):
            trainer.fit(empty)

    def test_fit_sets_is_fitted(self, structured_interactions):
        mat, _, n_users, n_items = structured_interactions
        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                        output_dim=4, n_epochs=3,
                                                        random_state=0))
        assert trainer.is_fitted is False
        trainer.fit(mat)
        assert trainer.is_fitted is True

    def test_loss_decreases_over_epochs(self, structured_interactions):
        mat, _, n_users, n_items = structured_interactions
        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=8, hidden_dim=12,
                                                        output_dim=6, learning_rate=0.03,
                                                        n_epochs=15, random_state=1))
        trainer.fit(mat)
        # Only meaningful on the NumPy backend (the Cython kernel doesn't
        # return a per-epoch loss -- see trainer.py's fit()).
        if trainer.loss_history[-1] == trainer.loss_history[-1]:  # not NaN
            assert trainer.loss_history[-1] < trainer.loss_history[0]

    def test_save_and_load_round_trip(self, structured_interactions, tmp_path):
        mat, _, n_users, n_items = structured_interactions
        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                        output_dim=4, n_epochs=2,
                                                        random_state=0))
        trainer.fit(mat)
        path = tmp_path / "model.npz"
        trainer.save_model(path)
        assert path.exists()

        loaded = TwoTowerTrainer.load_model(path)
        assert loaded.is_fitted is True
        np.testing.assert_array_almost_equal(loaded.weights.user_embeddings,
                                             trainer.weights.user_embeddings)


# ================================================================
# TwoTowerInference
# ================================================================

class TestExcludePurchased:
    """recommend(exclude_purchased=True): guarantees n_items whenever
    that many non-purchased catalogue items exist at all (fixes the
    'some customers get fewer than n_items' bug), and degrades
    gracefully (fewer results, no crash) only when a user has
    genuinely purchased almost the whole catalogue."""

    @pytest.fixture
    def trained_model_path(self, tmp_path):
        rng = np.random.default_rng(0)
        n_users, n_items, true_dim = 20, 50, 4
        true_user = rng.normal(size=(n_users, true_dim))
        true_item = rng.normal(size=(n_items, true_dim))
        true_scores = true_user @ true_item.T
        rows, cols = [], []
        for u in range(n_users):
            liked = np.argsort(true_scores[u])[::-1][:5]
            rows += [u] * len(liked)
            cols += list(liked)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                        output_dim=4, n_epochs=8, random_state=0))
        trainer.fit(mat)
        path = tmp_path / "model.npz"
        trainer.save_model(path)
        return path, n_users, n_items, true_scores

    def test_exclude_purchased_without_purchase_data_raises(self, trained_model_path):
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        with pytest.raises(ValueError):
            infer.recommend(0, n_items=10, exclude_purchased=True)

    def test_exclude_purchased_fills_shortfall_when_catalogue_allows(self, trained_model_path):
        """The core bug being fixed: even when a user has purchased
        exactly the items the model would naively rank highest, they
        must still get the full n_items -- pulled from further down
        the ranking, not artificially truncated first."""
        path, n_users, n_items, true_scores = trained_model_path
        # Purchase the top 15 items for user 0 -- deliberately the ones
        # a naive top-N-then-filter approach would have returned.
        top_15_for_user0 = np.argsort(true_scores[0])[::-1][:15].tolist()
        purchase_data = pd.DataFrame({"user_id": [0] * 15, "item_id": top_15_for_user0})

        infer = TwoTowerInference(path, purchase_data=purchase_data)
        recs = infer.recommend(0, n_items=10, exclude_purchased=True)
        assert len(recs) == 10
        rec_ids = {iid for iid, _ in recs}
        assert not (rec_ids & set(top_15_for_user0)), "no purchased item should be recommended"

    def test_exclude_purchased_degrades_gracefully_when_catalogue_exhausted(self, trained_model_path):
        path, n_users, n_items, true_scores = trained_model_path
        # Purchase 45 of the 50 catalogue items -- only 5 non-purchased
        # items can possibly exist; asking for 10 must not crash.
        purchased = list(range(45))
        purchase_data = pd.DataFrame({"user_id": [1] * len(purchased), "item_id": purchased})

        infer = TwoTowerInference(path, purchase_data=purchase_data)
        recs = infer.recommend(1, n_items=10, exclude_purchased=True)
        assert len(recs) == 5
        rec_ids = {iid for iid, _ in recs}
        assert not (rec_ids & set(purchased))

    def test_set_purchase_data_after_construction(self, trained_model_path):
        path, n_users, n_items, true_scores = trained_model_path
        infer = TwoTowerInference(path)
        purchase_data = pd.DataFrame({"user_id": [0, 0], "item_id": [1, 2]})
        infer.set_purchase_data(purchase_data)
        recs = infer.recommend(0, n_items=5, exclude_purchased=True)
        assert len(recs) == 5
        assert not ({iid for iid, _ in recs} & {1, 2})

    def test_exclude_purchased_false_is_unaffected(self, trained_model_path):
        """Baseline behavior (no purchase filtering) must be identical
        to before this feature existed."""
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        recs = infer.recommend(0, n_items=10)
        assert len(recs) == 10

    def test_batch_recommend_exclude_purchased(self, trained_model_path):
        path, n_users, n_items, true_scores = trained_model_path
        purchase_data = pd.DataFrame({
            "user_id": [0] * 10 + [1] * 3,
            "item_id": np.argsort(true_scores[0])[::-1][:10].tolist() + [5, 6, 7],
        })
        infer = TwoTowerInference(path, purchase_data=purchase_data)
        results = infer.batch_recommend([0, 1], n_items=8, exclude_purchased=True)
        assert len(results[0]) == 8
        assert len(results[1]) == 8
        assert not ({iid for iid, _ in results[1]} & {5, 6, 7})


class TestTwoTowerInference:

    @pytest.fixture
    def trained_model_path(self, tmp_path):
        rng = np.random.default_rng(0)
        n_users, n_items, true_dim = 30, 15, 4
        true_user = rng.normal(size=(n_users, true_dim))
        true_item = rng.normal(size=(n_items, true_dim))
        true_scores = true_user @ true_item.T
        rows, cols = [], []
        for u in range(n_users):
            liked = np.argsort(true_scores[u])[::-1][:5]
            rows += [u] * len(liked)
            cols += list(liked)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

        trainer = TwoTowerTrainer(n_users, n_items,
                                  config=TwoTowerConfig(embedding_dim=6, hidden_dim=8,
                                                        output_dim=4, n_epochs=10,
                                                        random_state=0))
        trainer.fit(mat)
        path = tmp_path / "model.npz"
        trainer.save_model(path)
        return path, true_scores, n_users, n_items

    def test_predict_paired_shape(self, trained_model_path):
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        scores = infer.predict(user_ids=[0, 1, 2], item_ids=[3, 4, 5])
        assert scores.shape == (3,)

    def test_predict_mismatched_lengths_raises(self, trained_model_path):
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        with pytest.raises(ValueError):
            infer.predict(user_ids=[0, 1], item_ids=[3])

    def test_recommend_respects_n_items_and_excludes(self, trained_model_path):
        path, _, n_users, n_items = trained_model_path
        infer = TwoTowerInference(path)
        recs = infer.recommend(user_id=0, n_items=5, exclude_items=[0, 1])
        assert len(recs) == 5
        rec_ids = [iid for iid, _ in recs]
        assert 0 not in rec_ids
        assert 1 not in rec_ids

    def test_recommend_sorted_by_score_descending(self, trained_model_path):
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        recs = infer.recommend(user_id=0, n_items=8)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_batch_recommend_covers_all_requested_users(self, trained_model_path):
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        result = infer.batch_recommend([0, 1, 2], n_items=3)
        assert set(result.keys()) == {0, 1, 2}
        assert all(len(v) == 3 for v in result.values())

    def test_get_metrics_has_no_ranking_quality_keys(self, trained_model_path):
        """Mirrors the Task-1 dashboard fix: an inference report must
        only ever contain real production metrics, never fabricated or
        misplaced ranking-quality numbers."""
        path, *_ = trained_model_path
        infer = TwoTowerInference(path)
        infer.predict(user_ids=[0], item_ids=[1])
        metrics = infer.get_metrics()
        for forbidden in ("precision_at_k", "recall_at_k", "ndcg", "auc", "coverage"):
            assert forbidden not in metrics

    def test_trained_model_beats_random_baseline(self, trained_model_path):
        path, true_scores, n_users, n_items = trained_model_path
        infer = TwoTowerInference(path)
        rng = np.random.default_rng(42)

        def precision_at_5(recs, u):
            true_top5 = set(np.argsort(true_scores[u])[::-1][:5].tolist())
            return len(set(r[0] for r in recs) & true_top5) / 5

        model_precision = np.mean([precision_at_5(infer.recommend(u, n_items=5), u)
                                   for u in range(n_users)])
        random_precision = np.mean([
            precision_at_5([(i, 0.0) for i in rng.choice(n_items, 5, replace=False)], u)
            for u in range(n_users)])

        assert model_precision > random_precision
