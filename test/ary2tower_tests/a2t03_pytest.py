#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Cooperative Recommendation Engine"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-23"

"""Direct pytest suite for the ary2tower model package.

This file tests modules under src/cooprecsys/models/ary2tower (or the
repository's equivalent cooprecsys/models/ary2tower layout). It does NOT
import either ary2tower smoke-test script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

from src.cooprecsys.models.ary2tower.config import TwoTowerConfig
from src.cooprecsys.models.ary2tower.towers import (
    ItemTower,
    TwoTowerWeights,
    UserTower,
    backend_info,
    cosine_similarity,
    dot_product_similarity,
)
from src.cooprecsys.models.ary2tower.inout.approximator import TwoTowerPredictor
from src.cooprecsys.models.ary2tower.trainer import TwoTowerTrainer
from src.cooprecsys.models.ary2tower.inference import TwoTowerInference

DATA_PATH = ROOT / "data" / "sampledata.parquet"


def interaction_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2, 3, 3],
        "item_id": [0, 1, 1, 2, 2, 3, 3, 4],
    })


@pytest.fixture
def tiny_trainer() -> TwoTowerTrainer:
    return TwoTowerTrainer(
        4, 5,
        config=TwoTowerConfig(
            embedding_dim=4, hidden_dim=6, output_dim=3,
            learning_rate=0.02, momentum=0.0, n_epochs=1,
            num_threads=1, random_state=42, verbose=False,
        ),
    )


class TestConfig:
    def test_defaults_and_params(self):
        cfg = TwoTowerConfig()
        p = cfg.get_params()
        assert p["embedding_dim"] == 32
        assert p["hidden_dim"] == 64
        assert p["output_dim"] == 16
        assert p["n_epochs"] == 10
        assert "verbose" not in p

    @pytest.mark.parametrize("field,value", [
        ("embedding_dim", 0), ("hidden_dim", 0), ("output_dim", 0),
        ("learning_rate", 0), ("n_epochs", 0), ("num_threads", 0),
    ])
    def test_rejects_non_positive(self, field, value):
        with pytest.raises(ValueError):
            TwoTowerConfig(**{field: value})

    def test_rejects_bad_momentum_and_seed(self):
        with pytest.raises(ValueError):
            TwoTowerConfig(momentum=1.0)
        with pytest.raises(TypeError):
            TwoTowerConfig(random_state="42")


class TestTowers:
    def test_weights_shapes_and_backend(self):
        w = TwoTowerWeights(4, 5, 4, 6, 3, random_state=42)
        assert w.user_embeddings.shape == (4, 4)
        assert w.item_embeddings.shape == (5, 4)
        assert w.user_w1.shape == (4, 6)
        assert w.item_w2.shape == (6, 3)
        assert set(backend_info()) == {"backend", "compiled"}

    def test_forward_and_similarity(self):
        w = TwoTowerWeights(4, 5, 4, 6, 3, random_state=42)
        u = UserTower(w).forward(np.array([0, 1, 2], dtype=np.int32))
        i = ItemTower(w).forward(np.array([0, 1, 2], dtype=np.int32))
        assert u.shape == i.shape == (3, 3)
        assert np.isfinite(u).all() and np.isfinite(i).all()

        uu = np.array([[1., 0.], [1., 1.]], dtype=np.float32)
        ii = np.array([[2., 0.], [1., 0.]], dtype=np.float32)
        np.testing.assert_allclose(dot_product_similarity(uu, ii), [2., 1.])
        np.testing.assert_allclose(cosine_similarity(uu, ii), [1., 1./np.sqrt(2)], atol=1e-6)


class TestPredictor:
    def test_build_pairs(self):
        u, i = TwoTowerPredictor.build_pairs([0, 1], [2, 3])
        np.testing.assert_array_equal(u, [0, 1])
        np.testing.assert_array_equal(i, [2, 3])

        u, i = TwoTowerPredictor.build_pairs(0, [1, 2, 3])
        np.testing.assert_array_equal(u, [0, 0, 0])
        np.testing.assert_array_equal(i, [1, 2, 3])

        u, i = TwoTowerPredictor.build_pairs([0, 1], [2, 3], cross_join=True)
        np.testing.assert_array_equal(u, [0, 0, 1, 1])
        np.testing.assert_array_equal(i, [2, 3, 2, 3])

        with pytest.raises(ValueError):
            TwoTowerPredictor.build_pairs([0, 1], [2, 3, 4])

    def test_predict_and_rank(self):
        model = TwoTowerPredictor(
            4, 5,
            config=TwoTowerConfig(embedding_dim=4, hidden_dim=5, output_dim=3, random_state=42),
        )
        scores = model.predict([0, 1], [1, 2])
        assert scores.shape == (2,) and np.isfinite(scores).all()
        cosine = model.predict([0, 1], [1, 2], similarity="cosine")
        assert cosine.shape == (2,) and np.isfinite(cosine).all()
        with pytest.raises(ValueError):
            model.predict([0], [1], similarity="bad")

        ranked = model.predict_rank(0, n_items=3, exclude_items=[1])
        assert len(ranked) == 3
        assert 1 not in {x[0] for x in ranked}
        assert len({x[0] for x in ranked}) == len(ranked)
        assert all(np.isfinite(x[1]) for x in ranked)


class TestTrainer:
    def test_fit_dataframe_and_params(self, tiny_trainer):
        assert not tiny_trainer.is_fitted
        tiny_trainer.set_params(learning_rate=0.01)
        assert tiny_trainer.config.learning_rate == 0.01
        with pytest.raises(ValueError):
            tiny_trainer.set_params(unknown_param=1)

        tiny_trainer.fit_dataframe(interaction_frame())
        assert tiny_trainer.is_fitted
        assert len(tiny_trainer.loss_history) == 1

    def test_fit_dataframe_validation(self, tiny_trainer):
        with pytest.raises(TypeError):
            tiny_trainer.fit_dataframe([{"user_id": 0, "item_id": 0}])
        with pytest.raises(ValueError):
            tiny_trainer.fit_dataframe(pd.DataFrame({"user_id": [0]}))
        with pytest.raises(ValueError):
            tiny_trainer.fit_dataframe(pd.DataFrame({"user_id": [99], "item_id": [0]}))

    def test_save_load_roundtrip(self, tiny_trainer, tmp_path):
        tiny_trainer.fit_dataframe(interaction_frame())
        path = tmp_path / "model.npz"
        tiny_trainer.save_model(path)
        assert path.exists()
        restored = TwoTowerTrainer.load_model(path)
        assert restored.is_fitted
        assert (restored.n_users, restored.n_items) == (4, 5)
        for name in (
            "user_embeddings", "user_w1", "user_b1", "user_w2", "user_b2",
            "item_embeddings", "item_w1", "item_b1", "item_w2", "item_b2",
        ):
            np.testing.assert_array_equal(getattr(restored.weights, name), getattr(tiny_trainer.weights, name))


class TestInference:
    def test_predict_recommend_batch_metrics(self, tiny_trainer, tmp_path):
        tiny_trainer.fit_dataframe(interaction_frame())
        path = tmp_path / "model.npz"
        tiny_trainer.save_model(path)
        inf = TwoTowerInference(path, num_threads=1, cache_enabled=True)

        scores = inf.predict([0, 1], [1, 2])
        assert scores.shape == (2,) and np.isfinite(scores).all()
        ranked = inf.recommend(0, n_items=3)
        assert len(ranked) == 3
        assert len({x[0] for x in ranked}) == len(ranked)
        batch = inf.batch_recommend([0, 1], n_items=2)
        assert set(batch) == {0, 1}
        assert all(len(v) == 2 for v in batch.values())
        metrics = inf.get_metrics()
        assert metrics["n_predictions"] > 0
        assert metrics["avg_latency_ms"] >= 0

    def test_validation_and_purchase_exclusion(self, tiny_trainer, tmp_path):
        tiny_trainer.fit_dataframe(interaction_frame())
        path = tmp_path / "model.npz"
        tiny_trainer.save_model(path)
        inf = TwoTowerInference(path, num_threads=1)
        with pytest.raises(ValueError):
            inf.predict([0, 1], [1])
        with pytest.raises(ValueError):
            inf.recommend(0, n_items=0)
        with pytest.raises(ValueError):
            inf.recommend(0, n_items=2, exclude_purchased=True)

        inf = TwoTowerInference(
            path, num_threads=1,
            purchase_data=interaction_frame(), user_col="user_id", item_col="item_id",
        )
        ranked = inf.recommend(0, n_items=2, exclude_purchased=True)
        purchased = set(interaction_frame().loc[interaction_frame()["user_id"] == 0, "item_id"])
        assert purchased.isdisjoint({x[0] for x in ranked})


@pytest.mark.integration
@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/sampledata.parquet is missing")
def test_production_data_path_to_ary2tower(tmp_path):
    """Verify the real parquet/preparation path can feed ary2tower."""
    from cooprecsys.features.load import load_data
    from cooprecsys.prepare import DetectReco_Identifier
    from cooprecsys.noisemaker import exnorex

    frame = load_data(DATA_PATH, memory_limit="2GB")
    assert isinstance(frame, pd.DataFrame) and not frame.empty

    ids = DetectReco_Identifier(frame)
    assert ids["user_col"] in frame.columns
    assert ids["item_col"] in frame.columns

    user_features = [c for c in ["EmployeeAge", "EmployeeGender", "CityName", "CountryName"] if c in frame.columns]
    item_features = [c for c in ["ProductPrice", "Quantity", "Discount", "TotalPrice", "Class", "Resistant", "IsAllergic", "VitalityDays"] if c in frame.columns]
    weight_col = ids.get("total_col") if ids.get("total_col") in frame.columns else None

    exchange = exnorex(
        frame,
        user_col=ids["user_col"],
        item_col=ids["item_col"],
        rating_col=None,
        weight_col=weight_col,
        user_feature_cols=user_features,
        item_feature_cols=item_features,
    )
    assert exchange.interactions.nnz > 0
    assert exchange.interactions.shape == (len(exchange.user_ids), len(exchange.item_ids))

    trainer = TwoTowerTrainer(
        exchange.interactions.shape[0], exchange.interactions.shape[1],
        config=TwoTowerConfig(
            embedding_dim=4, hidden_dim=6, output_dim=3,
            learning_rate=0.01, momentum=0.0, n_epochs=1,
            num_threads=1, random_state=42,
        ),
    )
    trainer.fit(exchange.interactions)
    path = tmp_path / "production_model.npz"
    trainer.save_model(path)
    assert path.exists()
