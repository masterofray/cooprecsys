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


"""Production-data smoke test for ary2tower training.

Execution path intentionally mirrors the repository's production pipeline:

    data/sampledata.parquet
        -> cooprecsys.features.load_data
        -> cooprecsys.prepare.DetectReco_Identifier
        -> cooprecsys.noisemaker.exnorex
        -> cooprecsys.models.ary2tower.TwoTowerTrainer

No synthetic interaction matrix is created here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" if (REPO_ROOT / "src" / "cooprecsys").is_dir() else REPO_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cooprecsys.features import load_data
from cooprecsys.models.ary2tower import TwoTowerConfig, TwoTowerTrainer
from cooprecsys.models.ary2tower.towers import backend_info
from cooprecsys.noisemaker import exnorex
from cooprecsys.prepare import DetectReco_Identifier


DEFAULT_DATA = REPO_ROOT / "data" / "sampledata.parquet"
DEFAULT_MODEL = REPO_ROOT / "test" / "ary2tower_tests" / ".artifacts" / "ary2tower_smoke_model.npz"


def build_production_payload(data_path: Path):
    """Load real sampledata and run the same ID/tensor preparation path."""
    frame = load_data(data_path, sample_n=None)
    if frame.empty:
        raise AssertionError("sampledata.parquet loaded an empty DataFrame")

    identifiers = DetectReco_Identifier(frame)
    user_col = identifiers["user_col"]
    item_col = identifiers["item_col"]
    if not user_col or not item_col:
        raise AssertionError(
            f"Could not resolve user/item columns from sampledata: {identifiers}"
        )

    # These are the feature columns used by the repository's own noisemaker
    # smoke test. Only pass columns which are actually present in the dataset.
    user_features = [c for c in ("CityName", "CountryName") if c in frame.columns]
    item_features = [
        c for c in ("ProductPrice", "Quantity", "Discount", "TotalPrice", "Class", "VitalityDays")
        if c in frame.columns
    ]
    if not user_features:
        raise AssertionError("No production user feature columns found in sampledata")
    if not item_features:
        raise AssertionError("No production item feature columns found in sampledata")

    weight_col = "TotalPrice" if "TotalPrice" in frame.columns else None
    payload = exnorex(
        data=frame,
        user_col=user_col,
        item_col=item_col,
        rating_col=None,
        weight_col=weight_col,
        user_feature_cols=user_features,
        item_feature_cols=item_features,
    )

    if payload.interactions.nnz == 0:
        raise AssertionError("Production preprocessing produced zero interactions")
    if payload.interactions.shape != (
        len(payload.user_ids),
        len(payload.item_ids),
    ):
        raise AssertionError("Interaction shape does not match encoded ID maps")

    return frame, identifiers, payload


def train_smoke(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL) -> Path:
    print(f"[ary2tower-train] data={data_path}")
    print(f"[ary2tower-train] backend={backend_info()}")

    frame, identifiers, payload = build_production_payload(data_path)
    print(
        "[ary2tower-train] loaded="
        f"rows={len(frame):,} users={len(payload.user_ids):,} "
        f"items={len(payload.item_ids):,} nnz={payload.interactions.nnz:,} "
        f"user_col={identifiers['user_col']} item_col={identifiers['item_col']}"
    )

    config = TwoTowerConfig(
        embedding_dim=8,
        hidden_dim=16,
        output_dim=8,
        learning_rate=0.02,
        momentum=0.9,
        n_epochs=2,
        num_threads=2,
        verbose=False,
        random_state=42,
    )
    trainer = TwoTowerTrainer(
        n_users=len(payload.user_ids),
        n_items=len(payload.item_ids),
        config=config,
    )
    trainer.fit(payload.interactions)

    assert trainer.is_fitted, "trainer did not mark itself fitted"
    assert len(trainer.loss_history) == config.n_epochs
    assert np.isfinite(trainer.weights.user_embeddings).all()
    assert np.isfinite(trainer.weights.item_embeddings).all()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_path)
    assert model_path.exists() and model_path.stat().st_size > 0

    loaded = TwoTowerTrainer.load_model(model_path)
    np.testing.assert_allclose(
        trainer.weights.user_embeddings,
        loaded.weights.user_embeddings,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        trainer.weights.item_embeddings,
        loaded.weights.item_embeddings,
        rtol=0,
        atol=0,
    )

    print("[ary2tower-train] PASS")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        train_smoke(args.data, args.model)
        return 0
    except Exception as exc:
        print(f"[ary2tower-train] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
