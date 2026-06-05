#!/usr/bin/env python3

"""
example.py
~~~~~~~~~~
End-to-end demonstration of the arycolbring collaborative-filtering pipeline.

Runs without any real dataset — generates a synthetic sparse interaction
matrix, splits it, trains with every supported loss function, evaluates
with all four metrics, and prints a summary table.

Usage
-----
    python example.py

Expected output (values will vary)
------------------------------------
    ┌──────────────────────────────────────────────────────────────┐
    │  arycolbring  —  Collaborative Filtering Demo                │
    ├──────────────┬──────────┬──────────┬──────────┬─────────────┤
    │ Loss         │ AUC      │ P@10     │ R@10     │ MRR         │
    ├──────────────┼──────────┼──────────┼──────────┼─────────────┤
    │ logistic     │ 0.xxxx   │ 0.xxxx   │ 0.xxxx   │ 0.xxxx      │
    │ warp         │ 0.xxxx   │ 0.xxxx   │ 0.xxxx   │ 0.xxxx      │
    │ bpr          │ 0.xxxx   │ 0.xxxx   │ 0.xxxx   │ 0.xxxx      │
    │ warp-kos     │ 0.xxxx   │ 0.xxxx   │ 0.xxxx   │ 0.xxxx      │
    └──────────────┴──────────┴──────────┴──────────┴─────────────┘
"""

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"

from __future__ import annotations

import logging
import sys
import time
from typing import Dict

import numpy as np
import scipy.sparse as sp

# ── configure logging before importing arycolbring ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("example")

# ── import arycolbring ────────────────────────────────────────────────────────
try:
    import arycolbring as ac
except ImportError as exc:
    sys.exit(
        f"[ERROR] Could not import arycolbring: {exc}\n"
        "Did you compile the Cython extensions?\n"
        "    python setup.py build_ext --inplace"
    )


# ── constants ─────────────────────────────────────────────────────────────────
N_USERS       = 800
N_ITEMS       = 300
N_INTERACTIONS = 12_000
TEST_PCT      = 0.20
NO_COMPONENTS = 24
EPOCHS        = 15
NUM_THREADS   = 4
K             = 10
SEED          = 2024

LOSSES = ["logistic", "warp", "bpr", "warp-kos"]


# ── synthetic data ────────────────────────────────────────────────────────────

def build_interactions(
    n_users:       int,
    n_items:       int,
    n_interactions: int,
    seed:          int,
) -> sp.coo_matrix:
    """
    Build a synthetic implicit-feedback sparse matrix.

    Interaction probabilities are drawn from a power-law distribution so
    that popular items and active users are more likely — more realistic
    than uniform random.
    """
    logger.info(
        "Building synthetic interactions: %d users × %d items, %d interactions",
        n_users, n_items, n_interactions,
    )
    rs = np.random.RandomState(seed)

    # Power-law weights for users and items
    user_weights = rs.exponential(scale=1.0, size=n_users) ** 2
    item_weights = rs.exponential(scale=1.0, size=n_items) ** 2
    user_weights /= user_weights.sum()
    item_weights /= item_weights.sum()

    rows = rs.choice(n_users, size=n_interactions, replace=True, p=user_weights)
    cols = rs.choice(n_items, size=n_interactions, replace=True, p=item_weights)
    data = np.ones(n_interactions, dtype=np.float32)

    mat = sp.coo_matrix(
        (data, (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    desc = ac.describe_interactions(mat)
    logger.info(
        "Matrix summary: density=%.4f  avg_interactions_per_user=%.1f",
        desc["density"].iloc[0],
        desc["avg_interactions_per_user"].iloc[0],
    )
    return mat


# ── evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, train, test) -> Dict[str, float]:
    """Run all four metrics and return their means."""
    return {
        "AUC":  float(ac.auc_score(model, test,
                                   train_interactions=train,
                                   num_threads=NUM_THREADS).mean()),
        f"P@{K}": float(ac.precision_at_k(model, test,
                                           train_interactions=train,
                                           k=K,
                                           num_threads=NUM_THREADS).mean()),
        f"R@{K}": float(ac.recall_at_k(model, test,
                                        train_interactions=train,
                                        k=K,
                                        num_threads=NUM_THREADS).mean()),
        "MRR":  float(ac.reciprocal_rank(model, test,
                                          train_interactions=train,
                                          num_threads=NUM_THREADS).mean()),
    }


# ── pretty-print table ────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    loss_w = 12
    col_w  = 10
    cols   = list(next(iter(results.values())).keys())

    # Header
    header_cells = [f"{'Loss':<{loss_w}}"] + [f"{c:>{col_w}}" for c in cols]
    header = "│ " + " │ ".join(header_cells) + " │"
    sep    = "├─" + "─┼─".join("─" * loss_w, *("─" * col_w for _ in cols)) + "─┤"
    top    = "┌─" + "─┬─".join("─" * loss_w, *("─" * col_w for _ in cols)) + "─┐"
    bot    = "└─" + "─┴─".join("─" * loss_w, *("─" * col_w for _ in cols)) + "─┘"

    title  = f"  arycolbring  —  Collaborative Filtering Demo"
    width  = len(header)

    print()
    print("┌" + "─" * (width - 2) + "┐")
    print("│" + title.center(width - 2) + "│")
    print(top)
    print(header)
    print(sep)

    for loss, metrics in results.items():
        row_cells = [f"{loss:<{loss_w}}"] + [
            f"{v:>{col_w}.4f}" for v in metrics.values()
        ]
        print("│ " + " │ ".join(row_cells) + " │")

    print(bot)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== arycolbring example ===")

    # 1 – Build data
    interactions = build_interactions(N_USERS, N_ITEMS, N_INTERACTIONS, SEED)

    # 2 – Split
    logger.info("Splitting interactions (test_pct=%.0f%%)", TEST_PCT * 100)
    train, test = ac.random_train_test_split(
        interactions, test_percentage=TEST_PCT, random_state=SEED
    )
    logger.info("train.nnz=%d  test.nnz=%d", train.nnz, test.nnz)

    # 3 – Train + evaluate every loss
    results: Dict[str, Dict[str, float]] = {}

    for loss in LOSSES:
        logger.info("─── Training  loss='%s' ───", loss)

        model = ac.AryColBring(
            no_components    = NO_COMPONENTS,
            loss             = loss,
            learning_schedule = "adagrad",
            learning_rate    = 0.05,
            item_alpha       = 1e-6,
            user_alpha       = 1e-6,
            max_sampled      = 10,
            random_state     = SEED,
        )

        t0 = time.perf_counter()
        model.fit(train, epochs=EPOCHS, num_threads=NUM_THREADS, verbose=True)
        elapsed = time.perf_counter() - t0

        logger.info("Training complete in %.2fs", elapsed)

        metrics = evaluate(model, train, test)
        results[loss] = metrics

        logger.info(
            "loss=%-10s  AUC=%.4f  P@%d=%.4f  R@%d=%.4f  MRR=%.4f",
            loss,
            metrics["AUC"],
            K, metrics[f"P@{K}"],
            K, metrics[f"R@{K}"],
            metrics["MRR"],
        )

    # 4 – Print summary table
    print_results_table(results)

    # 5 – Show representation shapes
    best_loss = max(results, key=lambda l: results[l]["AUC"])
    logger.info("Best loss by AUC: '%s'", best_loss)

    # Re-train the best model to expose representation API
    best_model = ac.AryColBring(
        no_components=NO_COMPONENTS, loss=best_loss, random_state=SEED
    )
    best_model.fit(train, epochs=EPOCHS, num_threads=NUM_THREADS, verbose=False)

    item_biases, item_embs = best_model.get_item_representations()
    user_biases, user_embs = best_model.get_user_representations()

    print(f"Item embeddings shape : {item_embs.shape}")
    print(f"User embeddings shape : {user_embs.shape}")
    print(f"Item biases shape     : {item_biases.shape}")

    # 6 – Demo: top-5 items for user 0 (brute-force scoring)
    logger.info("Scoring all items for user_id=0 …")
    user_id   = 0
    all_items = np.arange(N_ITEMS, dtype=np.int32)
    user_arr  = np.repeat(np.int32(user_id), N_ITEMS)

    scores    = best_model.predict(user_arr, all_items, num_threads=NUM_THREADS)
    top5_idx  = np.argsort(-scores)[:5]

    print(f"\nTop-5 item recommendations for user {user_id}:")
    for rank, item_id in enumerate(top5_idx, start=1):
        print(f"  Rank {rank}: item_id={item_id:4d}  score={scores[item_id]:.4f}")

    logger.info("=== Example complete ===")


if __name__ == "__main__":
    main()
