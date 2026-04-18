# arycolbring

**Ultra-optimised user-to-item collaborative filtering** powered by Cython + OpenMP.

Rebuilt from the LightFM architecture with:
- Each Cython kernel in its own `.pyx` file for easy debugging
- `prange` (OpenMP) parallelism in every hot loop
- `malloc` / `free` per-thread buffers — no Python GIL in hot paths
- DuckDB-backed data ingestion
- `tqdm` progress bars throughout
- Structured `logging` at `DEBUG` level
- Sparse matrix acceleration everywhere

---

## Table of Contents

1. [Requirements](#requirements)
2. [How to Build & Run](#how-to-build--run)
3. [Object Inputs Needed](#object-inputs-needed)
4. [Flow Process in This System](#flow-process-in-this-system)
5. [Quick Example](#quick-example)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Contribution](#contribution)

---

## Requirements

| Package | Min version | Purpose |
|---------|-------------|---------|
| Python  | 3.8+        | interpreter |
| Cython  | 0.29+       | transpilation |
| NumPy   | 1.21+       | array operations |
| SciPy   | 1.7+        | sparse matrices |
| Pandas  | 1.3+        | DataFrame helpers |
| DuckDB  | 0.8+        | SQL on DataFrames |
| tqdm    | 4.60+       | progress bars |
| joblib  | 1.1+        | parallel Python utilities |
| seaborn | 0.12+       | evaluation visualisation |
| GCC / Clang with **OpenMP** support | any | parallel Cython kernels |

---

## How to Build & Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/arycolbring.git
cd arycolbring
pip install -r requirements.txt
```

### 2. Compile Cython extensions (in-place, for development)

```bash
python setup.py build_ext --inplace
```

This produces `_cy_*.so` (Linux / macOS) or `_cy_*.pyd` (Windows) files
inside `arycolbring/cy/`.

> **macOS note** — Apple's default `clang` does not ship OpenMP.
> Install LLVM via Homebrew and point the compiler at it:
> ```bash
> brew install libomp llvm
> export CC=$(brew --prefix llvm)/bin/clang
> python setup.py build_ext --inplace
> ```

### 3. Install as a package (production)

```bash
pip install .
```

### 4. Run a quick smoke-test

```bash
python -c "from arycolbring import AryColBring; print(AryColBring())"
```

### 5. Debug Cython output

Each `_cy_*.pyx` emits `fprintf(stderr, ...)` messages.
Redirect stderr to a file to capture them:

```bash
python train.py 2>debug.log
tail -f debug.log
```

---

## Object Inputs Needed

### Interaction matrix

| Property | Requirement |
|----------|-------------|
| Type | `scipy.sparse` matrix (any format — internally converted to COO/CSR) |
| Shape | `(n_users, n_items)` |
| Values | Positive interaction weights (e.g. 1.0 for implicit, rating for explicit) |
| dtype | float32 preferred; float64 is automatically cast |

### Optional feature matrices

| Argument | Shape | Description |
|----------|-------|-------------|
| `user_features` | `(n_users, n_user_features)` CSR float32 | Side-information for users |
| `item_features` | `(n_items, n_item_features)` CSR float32 | Side-information for items |

If either is `None`, an identity matrix is used (pure ID-based embedding).

### Sample weights

| Argument | Shape | Description |
|----------|-------|-------------|
| `sample_weight` | same shape as `interactions`, COO format | Per-interaction training weight |

Not supported with `loss="warp-kos"`.

---

## Flow Process in This System

```
┌─────────────────────────────────────────────────────────────────┐
│                         PYTHON LAYER                            │
│                                                                 │
│  Raw data (CSV / DataFrame)                                     │
│        │                                                        │
│        ▼  load_interactions_from_csv / load_interactions_from_df│
│  DuckDB encodes user/item IDs → scipy COO sparse matrix         │
│        │                                                        │
│        ▼  random_train_test_split / user_based_train_test_split │
│  (train COO, test COO)                                          │
│        │                                                        │
│        ▼  AryColBring.fit(train, epochs=N, num_threads=T)       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Per-epoch loop (tqdm)                       │   │
│  │  • shuffle interaction indices (numpy)                   │   │
│  │  • construct CSRMatrix wrappers (Cython cdef class)      │   │
│  │  • dispatch to Cython kernel ──────────────────────────► │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│        ▼  AryColBring.predict / predict_rank                    │
│  Scores / ranks returned as numpy arrays                        │
│        │                                                        │
│        ▼  precision_at_k / recall_at_k / auc_score / mrr       │
│  Evaluation metrics (float arrays per user)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        CYTHON LAYER  (nogil + OpenMP)           │
│                                                                 │
│  _cy_types.pyx  ─── CSRMatrix, FastAryColBring cdef classes     │
│  _cy_math.pxd   ─── inline PRNG, sigmoid, in_positives, sorts  │
│  _cy_representation.pxd ─ compute_representation,               │
│                            compute_prediction_from_repr         │
│  _cy_update.pxd ─── update_biases, update_features,            │
│                      update (logistic), warp_update             │
│  _cy_regularize.pxd ─ regularize (lazy L2 flush),              │
│                        locked_regularize (OMP lock)             │
│                                                                 │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐    │
│  │fit_logistic  │  │ fit_warp        │  │ fit_bpr         │    │
│  │fit_warp_kos  │  │ predict_arycolbring│ predict_ranks  │    │
│  │calculate_auc_from_rank                                  │    │
│  └──────────────┘  └────────────────┘  └─────────────────┘    │
│                                                                 │
│  Each function:                                                 │
│   • converts Python args to C scalars / typed memviews          │
│   • enters `with nogil, parallel(num_threads=T):`               │
│   • malloc() per-thread buffers                                 │
│   • iterates via prange(N, schedule='dynamic')                  │
│   • free() buffers after prange                                 │
│   • flushes lazy L2 regularisation with OMP lock               │
└─────────────────────────────────────────────────────────────────┘
```

### Loss functions

| Loss | Description |
|------|-------------|
| `logistic` | Point-wise binary cross-entropy. Good starting point. |
| `warp` | Weighted Approximate-Rank Pairwise. Best for implicit feedback. |
| `bpr` | Bayesian Personalised Ranking. Efficient pairwise alternative to WARP. |
| `warp-kos` | WARP with k-th Order Statistic positive selection. Most robust for noisy data. |

---

## Quick Example

```python
import scipy.sparse as sp
import numpy as np
from arycolbring import (
    AryColBring,
    random_train_test_split,
    auc_score,
    precision_at_k,
)

# Build a toy interaction matrix (500 users × 200 items)
np.random.seed(42)
rows = np.random.randint(0, 500, 5000)
cols = np.random.randint(0, 200, 5000)
interactions = sp.coo_matrix(
    (np.ones(5000, dtype=np.float32), (rows, cols)),
    shape=(500, 200),
)

# Split
train, test = random_train_test_split(interactions, test_percentage=0.2)

# Train
model = AryColBring(
    no_components=32,
    loss="warp",
    learning_schedule="adagrad",
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42,
)
model.fit(train, epochs=10, num_threads=4, verbose=True)

# Evaluate
auc   = auc_score(model, test, train_interactions=train, num_threads=4)
prec  = precision_at_k(model, test, train_interactions=train, k=10, num_threads=4)

print(f"AUC:          {auc.mean():.4f}")
print(f"Precision@10: {prec.mean():.4f}")

# Score specific pairs
user_ids = np.array([0, 1, 2], dtype=np.int32)
item_ids = np.array([5, 10, 15], dtype=np.int32)
scores = model.predict(user_ids, item_ids, num_threads=4)
print(f"Scores: {scores}")
```

---

## Evaluation Metrics

| Function | Returns |
|----------|---------|
| `precision_at_k(model, test, k=10)` | Fraction of top-k that are true positives |
| `recall_at_k(model, test, k=10)` | Fraction of true positives in top-k |
| `auc_score(model, test)` | ROC AUC per user (mean ≈ 0.5 random, 1.0 perfect) |
| `reciprocal_rank(model, test)` | 1/rank of best true positive per user |

All metrics accept `train_interactions` to exclude known training positives.

---

## Configuration

Edit `arycolbring/config.ini` to change defaults:

```ini
[tqdm]
colour = #05ad46
ncols  = 88

[model]
loss              = warp
no_components     = 10
learning_schedule = adagrad
epochs            = 10
num_threads       = 4

[duckdb]
threads = 4

[logging]
level = DEBUG
```

---

## Project Structure

```
arycolbring/
├── setup.py                       # Cython build script (OpenMP enabled)
├── requirements.txt
├── README.md
└── arycolbring/
    ├── __init__.py                # Public API surface
    ├── config.ini                 # Default settings
    ├── model.py                   # AryColBring Python class
    ├── cross_validation.py        # Train/test splitting
    ├── evaluation.py              # Ranking metrics
    ├── data_utils.py              # DuckDB data loading helpers
    └── cy/
        ├── __init__.py
        ├── _cy_types.pxd          # cdef class declarations (shared header)
        ├── _cy_types.pyx          # CSRMatrix, FastAryColBring implementation
        ├── _cy_math.pxd           # Inline math utilities (PRNG, sigmoid, …)
        ├── _cy_math.pyx           # (stub — all logic is inline in .pxd)
        ├── _cy_representation.pxd # Inline latent-repr computation
        ├── _cy_representation.pyx # (stub)
        ├── _cy_update.pxd         # Inline SGD update kernels
        ├── _cy_update.pyx         # (stub)
        ├── _cy_regularize.pxd     # Inline lazy L2 flush + OMP lock version
        ├── _cy_regularize.pyx     # (stub)
        ├── _cy_fit_logistic.pyx   # Logistic-loss training epoch
        ├── _cy_fit_warp.pyx       # WARP-loss training epoch
        ├── _cy_fit_bpr.pyx        # BPR-loss training epoch
        ├── _cy_fit_warp_kos.pyx   # WARP-kOS training epoch
        ├── _cy_predict.pyx        # Pointwise scoring + rank computation
        └── _cy_evaluate.pyx       # ROC-AUC from rank arrays
```

---

## Contribution

| Role | Name |
|------|------|
| **Author / maintainer** | **aryanto** |

Contributions are welcome via pull request.  Please:

1. Open an issue describing the bug or feature first.
2. Write a test that fails before your fix and passes after.
3. Follow the existing code style (type annotations, DEBUG logging, tqdm bars).
4. Do not remove `fprintf(stderr, …)` debug instrumentation from `.pyx` files —
   it is essential for diagnosing issues in compiled code.
