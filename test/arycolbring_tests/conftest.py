# tests/conftest.py
"""
Shared fixtures for all arycolbring test modules.

Each fixture builds a tiny but structurally realistic sparse interaction
matrix so tests are fast even without compiled Cython extensions.
"""

import numpy as np
import pytest
import scipy.sparse as sp


# ── reproducible RNG ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(seed=42)


# ── tiny interaction matrices ─────────────────────────────────────────────────

def _make_coo(n_users: int, n_items: int, n_interactions: int, seed: int = 42):
    """Helper: random COO interaction matrix."""
    rs = np.random.RandomState(seed)
    rows = rs.randint(0, n_users, n_interactions).astype(np.int32)
    cols = rs.randint(0, n_items, n_interactions).astype(np.int32)
    data = np.ones(n_interactions, dtype=np.float32)
    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


@pytest.fixture(scope="session")
def small_interactions():
    """100 users × 50 items, 500 positive interactions."""
    return _make_coo(100, 50, 500)


@pytest.fixture(scope="session")
def medium_interactions():
    """500 users × 200 items, 5000 positive interactions."""
    return _make_coo(500, 200, 5000)


@pytest.fixture(scope="session")
def train_test_pair(small_interactions):
    """Pre-split (train, test) from the small matrix."""
    from arycolbring import random_train_test_split
    return random_train_test_split(small_interactions,
                                   test_percentage=0.2,
                                   random_state=0)


@pytest.fixture(scope="session")
def fitted_model(train_test_pair):
    """A warp-loss model fitted for 3 epochs (fast, deterministic)."""
    from arycolbring import AryColBring
    train, _ = train_test_pair
    model = AryColBring(
        no_components=8,
        loss="warp",
        learning_rate=0.05,
        max_sampled=5,
        random_state=99,
    )
    model.fit(train, epochs=3, num_threads=1, verbose=False)
    return model, train, _  # model, train, test
