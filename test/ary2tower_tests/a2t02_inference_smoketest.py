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


"""Production-data smoke test for ary2tower inference.

The model is trained from data/sampledata.parquet through the same production
preparation path when the expected smoke-test artifact is not present. This
keeps the script directly runnable by itself while keeping the actual checks
focused on inference.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from src.cooprecsys.configs import logger
from src.cooprecsys.features import load_data
from src.cooprecsys.models.ary2tower import TwoTowerInference
from src.cooprecsys.models.ary2tower.towers import backend_info
from src.cooprecsys.prepare import DetectReco_Identifier
from src.cooprecsys.noisemaker import exnorex
from test.ary2tower_tests.a2t01_train_smoketest import DEFAULT_DATA, DEFAULT_MODEL, train_smoke


def inference_smoke(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL) -> None:
    print(f"[ary2tower-inference] data={data_path}")
    print(f"[ary2tower-inference] backend={backend_info()}")

    frame = load_data(data_path, sample_n=None)
    identifiers = DetectReco_Identifier(frame)
    user_col = identifiers["user_col"]
    item_col = identifiers["item_col"]
    if not user_col or not item_col:
        raise AssertionError(f"Could not resolve user/item columns: {identifiers}")

    # Re-run the production tensor preparation to establish the actual encoded
    # user/item index spaces used by the trained model.
    user_features = [c for c in ("CityName", "CountryName") if c in frame.columns]
    item_features = [
        c for c in ("ProductPrice", "Quantity", "Discount", "TotalPrice", "Class", "VitalityDays")
        if c in frame.columns
    ]
    payload = exnorex(
        data=frame,
        user_col=user_col,
        item_col=item_col,
        rating_col=None,
        weight_col="TotalPrice" if "TotalPrice" in frame.columns else None,
        user_feature_cols=user_features,
        item_feature_cols=item_features,
    )

    if not model_path.exists():
        print("[ary2tower-inference] model artifact missing; creating it via train smoke")
        train_smoke(data_path, model_path)

    inference = TwoTowerInference(
        model_path=model_path,
        num_threads=2,
        cache_enabled=True,
    )

    # The model consumes encoded integer indices, which are exactly the row/
    # column positions produced by exnorex's DENSE_RANK mapping.
    user_ids = np.arange(min(3, len(payload.user_ids)), dtype=np.int32)
    item_ids = np.arange(min(3, len(payload.item_ids)), dtype=np.int32)
    if len(user_ids) == 0 or len(item_ids) == 0:
        raise AssertionError("Production payload does not contain users/items")

    pair_count = min(len(user_ids), len(item_ids))
    scores = inference.predict(user_ids[:pair_count], item_ids[:pair_count])
    assert scores.shape == (pair_count,)
    assert np.isfinite(scores).all()

    target_user = int(user_ids[0])
    recs = inference.recommend(target_user, n_items=min(5, len(payload.item_ids)))
    assert recs, "recommend() returned no candidates"
    assert len({int(item) for item, _ in recs}) == len(recs)
    assert all(np.isfinite(float(score)) for _, score in recs)
    assert all(0 <= int(item) < len(payload.item_ids) for item, _ in recs)

    batch_users = user_ids.tolist()
    batch = inference.batch_recommend(
        batch_users,
        n_items=min(3, len(payload.item_ids)),
    )
    assert set(batch.keys()) == set(batch_users)
    for uid, user_recs in batch.items():
        assert isinstance(uid, int)
        assert len({int(item) for item, _ in user_recs}) == len(user_recs)

    metrics = inference.get_metrics()
    for key in ("n_predictions", "n_users_served", "avg_latency_ms", "throughput_preds_per_sec", "qps"):
        assert key in metrics
        assert np.isfinite(float(metrics[key]))

    print(
        f"[ary2tower-inference] predictions={metrics['n_predictions']} "
        f"recommendations={len(recs)} qps={metrics['qps']:.3f}"
    )
    print("[ary2tower-inference] PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        inference_smoke(args.data, args.model)
        return 0
    except Exception as exc:
        print(f"[ary2tower-inference] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
