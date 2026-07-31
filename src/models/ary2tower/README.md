# ary2tower — Two-Tower Neural Recommender

A two-tower (user tower / item tower) recommender: each tower maps an
id to a dense representation via `Embedding -> Dense -> ReLU -> Dense`,
and recommendation scores are the dot product (or cosine similarity)
between a user's and an item's tower output. Trained with a BPR-style
pairwise loss (SGD + momentum) over implicit-feedback interactions.

Performance-critical code (the forward pass and the training update)
is implemented in Cython with OpenMP parallelism, following the same
build conventions already established by `src/models/arycolbring/`
(`cysetup.py`'s platform-conditional OpenMP flags, `fprintf(stderr,
...)`-based verbose diagnostics instead of a Python logger callback
inside `nogil` code, and the same Hogwild!-style unlocked parallel
embedding update arycolbring's own `fit_bpr()`/`fit_warp()` kernels
already use).

## Quick start

```python
from src.models.ary2tower import TwoTowerConfig, TwoTowerTrainer, TwoTowerInference

config = TwoTowerConfig(embedding_dim=32, hidden_dim=64, output_dim=16,
                        learning_rate=0.01, momentum=0.9, n_epochs=20)
trainer = TwoTowerTrainer(n_users, n_items, config=config)
trainer.fit(interactions)                       # interactions: scipy.sparse matrix
trainer.save_model("artifacts/ary2tower/model.npz")

infer = TwoTowerInference("artifacts/ary2tower/model.npz", cache_enabled=True)
recommendations = infer.recommend(user_id=0, n_items=10, exclude_items=[3, 7])
```

## Building the Cython extension

```bash
cd src/models/ary2tower
bash cysetup.sh          # Linux/macOS
# or: pwsh cysetup.ps1    # Windows
```

This compiles `CLtowers/*.pyx` into a loadable extension. **If the
extension isn't built, everything still works** — `towers.py` detects
this (`ImportError`) and transparently falls back to an
algorithmically-identical pure-NumPy implementation (slower, no OpenMP
parallelism, but numerically the same). `towers._HAS_CYTHON` tells you
which backend is active; a warning is logged on import when falling
back.

## Module layout

```
src/models/ary2tower/
├── __init__.py          # public API: TwoTowerConfig, TwoTowerTrainer, TwoTowerInference, ...
├── config.py             # TwoTowerConfig -- validated at construction (see config.py docstring)
├── towers.py              # TwoTowerWeights / UserTower / ItemTower + Cython-or-NumPy dispatch
├── trainer.py            # TwoTowerTrainer.fit() / save_model() / load_model()
├── inference.py           # TwoTowerInference.predict() / recommend() / batch_recommend()
├── cysetup.py / .sh / .ps1  # build the Cython extension (mirrors arycolbring/cysetup.*)
└── CLtowers/               # Cython + OpenMP kernels
    ├── _cy_types.pxd/.pyx     # TwoTowerModel cdef class (all weight/momentum state)
    ├── _cy_forward.pyx        # batched tower_forward() -- used by inference
    ├── _cy_similarity.pyx     # dot_product() / cosine_similarity()
    └── _cy_train.pyx          # fit_two_tower() -- one epoch of BPR-style training
```

## A note on verification

This module's Cython/OpenMP code (`CLtowers/*.pyx`) was written and
reviewed carefully, but **has not actually been compiled or executed**
in the environment it was developed in (no `cython` package, no C
compiler, no network access to install either). What *was* verified,
for real, in that environment:

- The exact same forward/backward-pass math was ported line-for-line
  into pure Python/NumPy and checked against **numerical (finite-
  difference) gradients** for every parameter tensor in both towers —
  all matched to within `1e-12` relative error. This confirms the
  *algorithm* the Cython kernel implements is mathematically correct.
- A full BPR training loop (same loss, same sign conventions as
  `fit_two_tower`) was run on synthetic structured data and shown to
  reduce loss and beat a random-recommendation baseline on Precision@5.
- The entire Python layer (`config.py`, `towers.py`, `trainer.py`,
  `inference.py`, `__init__.py`) runs for real via its automatic
  NumPy fallback — including the full `t01_towers.py` pytest suite,
  driven end to end (config validation, tower shapes, training
  convergence, save/load round-trip, predict/recommend/batch_recommend,
  and a trained-model-beats-random-baseline check).
- Every `.pxd`/`.pyx` field declaration was cross-checked for
  consistency (no field declared but unassigned, or vice versa), and
  the file structure follows `arycolbring/CLproximity/_cy_predict.pyx`'s
  exact conventions for thread-local scratch-buffer declarations.

**Before relying on this in production**: build the extension
(`cysetup.sh`/`cysetup.ps1`) and run `test/ary2tower_tests/t01_towers.py`
against the compiled backend (`towers._HAS_CYTHON` should read `True`)
to confirm the Cython path itself compiles and matches the NumPy
path's behavior — that step could not be completed here.
