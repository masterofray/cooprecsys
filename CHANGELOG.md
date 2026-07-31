# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **`src/models/ary2tower/`**: new two-tower neural recommender
  (`Embedding -> Dense -> ReLU -> Dense` per tower, dot-product/cosine
  similarity, BPR-style pairwise training with SGD+momentum).
  Cython + OpenMP kernels in `CLtowers/`, with an automatic pure-NumPy
  fallback (`towers._HAS_CYTHON`) when the extension isn't built.
  Build scripts (`cysetup.py/.sh/.ps1`) adapted directly from
  `arycolbring/cysetup.*`. See `src/models/ary2tower/README.md` for
  the module's own notes on what has and hasn't been compiled/run.
- `test/ary2tower_tests/t01_towers.py` -- 22 tests covering config
  validation, tower shapes, training convergence, save/load round-trip,
  and inference (predict/recommend/batch_recommend/get_metrics).
- New **Insights** tab on the AryColBring inference dashboard
  (`narative/infrc_insights.html.j2` + `insights.css`/`insights.js`):
  prediction score histogram, 2D PCA embedding projection, item-item
  similarity heatmap (Plotly, reusing the `.js-heatmap-render`
  data-attribute convention already established in
  `ltr_lgbm/report/static/js/heatmaps.js`).
- `src/assets/vizdata.py` -- shared, dependency-light (NumPy/pandas
  only) visualization-data builders: `score_distribution`,
  `embedding_projection_2d`, `similarity_heatmap`, `top_k_similar_items`.
- `src/assets/dashboard_utils.py` -- `generate_scorecards`,
  `generate_gauges`, `normalize_charts`, `bealabel`, `safe_float`,
  `detect_gauge_metric`, `overall_score_percent`, deduplicated out of
  `advirender.py` and `rearender.py` (previously near-identical copies
  in both files).
- "Similar Items (Top 3)" column + click-to-sort Score header on the
  inference dashboard's Rankings table.
- 6 new notebooks in `notebook/`: `01_Quick_Start`,
  `02_Data_Preparation`, `03_Training_AryColBring`,
  `04_Interactive_Dashboard`, `05_LTR_LGBM_Comparison`,
  `06_Cold_Start_Handling` (alongside the pre-existing
  `AryColBring_Training_Pipeline.ipynb`).
- `test/arycolbring_tests/t05_dashboard_refactor.py` -- regression
  suite for the dashboard/metrics fixes below.
- `lint` and `pytest-suite` jobs in
  `.github/workflows/arycolbring_pipeline.yml`, plus a step chaining
  `t02_reasoner.py` onto the model `t01_advisor.py` trains in the same
  CI job (exercises the full inference + dashboard pipeline in CI).
- 26 new tests in `test/arycolbring_tests/t03_pytest.py`
  (`TestReasonerHyperparameterValidation`, `TestReasonerParams`,
  `TestReasonerPairUtilities`) covering `TheReasoner`/`TheAdvisor`'s
  shared validation logic and `build_pairs()`/`_is_string_type()`.

### Fixed
- **`get_env()` in `narative/rensupport.py` silently swallowed all Jinja
  initialization failures.** `finally: return env` unconditionally
  discarded any exception raised in the `try`/`except` above it (a
  `return`/`break`/`continue` inside `finally` always wins in Python) --
  meaning `raise RuntimeError() from exc` was dead code, and a broken
  template setup would silently return a loader-less `Environment`
  instead, which would then fail later with a confusing, unrelated
  error on the first `get_template()` call. Fixed: the success path now
  returns directly from `try`, and a genuine failure correctly
  propagates as `RuntimeError`. Verified both paths (happy path
  unaffected; simulated failure now raises as intended) plus a full
  re-run of the Task 1 dashboard pipeline end to end.
- Mutable default argument (`dirlist: List = ['css', 'js']`) in
  `copymaps()` (`narative/rensupport.py`) -- benign in practice (the
  list was only read, never mutated) but replaced with a `None`
  sentinel per standard practice. Verified no state leaks across calls.

- **Fabricated "coverage" metric removed.** The inference dashboard
  (`inference.py` / `narative/rearender.py`) previously always showed
  a hardcoded `0.75` "Coverage" stat as if it were measured; nothing in
  the codebase ever computed a real value. Removed rather than shipped.
- **Eval metrics no longer leak into the inference dashboard.**
  Precision/Recall/NDCG/AUC/MRR-style ranking-quality metrics were
  being categorized and gauge-displayed on the *inference* report;
  those belong on the *training* dashboard only. `generate_gauges()`
  now correctly produces zero gauges for a clean inference-metrics
  dict (latency/qps/throughput).
- **`bealabel()` formatting bug**: `ndcg_at_10` rendered as
  `"Ndcg @ 10"` instead of `"NDCG@10"` (substring-replace-after-
  underscore-expansion bug). Fixed via token-based formatting;
  caught by a unit test during the `dashboard_utils.py` extraction.
- **Two broken tests in `t03_pytest.py`** (`test_invalid_loss_function`,
  `test_invalid_learning_schedule`) asserted construction *succeeds*
  with an invalid value, with a comment claiming validation happens "at
  fit time." Confirmed by reading `AryColBringBase.__init__`
  (`inout/scaffold.py`): validation actually happens immediately at
  construction. Tests corrected to expect `ValueError` at construction.
- **`TestDataHandling` in `t03_pytest.py`** read `stats["n_interactions"]`
  / `stats["sparsity"]` from `describe_interactions()`'s return value --
  neither column exists (real columns: `n_users`, `n_items`, `nnz`,
  `density`, `avg/min/max_interactions_per_user`, confirmed by
  cross-referencing `trainer.py`'s own internal usage). Fixed to use
  the real column names and `.iloc[0]` scalar access.
- **Coverage measurement was silently broken project-wide**:
  `pyproject.toml`'s `--cov=cooprecsys` / `source = ["cooprecsys"]`
  pointed at a module name that isn't importable anywhere in this
  repo (the real root package is `src`). Every coverage number ever
  reported under the old config was measuring nothing. Fixed to
  `--cov=src` / `source = ["src"]`.
- **`pytest`'s bare discovery found zero arycolbring tests**:
  `python_files = ["ltrlgbm_example.py", "*_test.py"]` matched neither
  `t03_pytest.py` nor `t05_dashboard_refactor.py`. Both filenames added.
- Missing dev dependencies (`pytest-cov`, `black`, `flake8`, `isort`)
  were used/implied (via `--cov` in `addopts`, and the CI spec's lint
  requirement) but never declared in `pyproject.toml`'s `dev` extras.
  Added.

### Changed
- Retheme: `arycolbring/narative/{stinferc,sttrain}/css/base.css` and
  related CSS/JS moved from a light-green palette to light + orange
  (`#FF6B35`/`#F7931E` primary, `#4ECDC4` secondary, `#F8F9FA`
  background), per the requested design spec.
  Note: the original task brief's premise that the *current* dashboard
  was dark-themed (needing to become light, "matching ltr_lgbm/report")
  didn't match the repo as found -- `narative` was already light green,
  and `ltr_lgbm/report` is actually dark navy. Followed the explicit
  hex palette given rather than either of those two as-found states.
- `arycolbring_pipeline.yml` `pull_request` trigger widened to include
  `main` (previously only `dev*`).

### Known limitations / not done in this session
- The `ary2tower` Cython/OpenMP kernels (`CLtowers/*.pyx`) have been
  written and structurally reviewed, but **not compiled or executed**
  -- no `cython` package, C compiler, or network access were available
  in the development sandbox. The underlying algorithm *was* verified
  via numerical gradient-checking against a pure-Python port and a
  full training-convergence run; see `src/models/ary2tower/README.md`.
  Build and test the compiled path (`cysetup.sh`/`.ps1`, then
  `t01_towers.py` with `towers._HAS_CYTHON == True`) before relying on
  it in production.
- `--cov-fail-under` thresholds were deliberately **not** added to the
  new CI `pytest-suite` job. No real coverage baseline exists yet (the
  old config never measured anything real -- see Fixed, above);
  picking a round number now would be a second fabricated threshold on
  top of the first one. Set this once a real run establishes a baseline.
- `black`/`flake8`/`isort`/`pytest` could not be run locally to verify
  the new `lint`/`pytest-suite` CI jobs before this change -- same
  network/dependency constraints as above.
- A full-repo bug scan (AST-based, all 92 `.py` files) found the same
  `finally: return <value>` exception-swallowing pattern (see Fixed,
  above) in 6 more places **outside this session's file scope**, not
  fixed here: `src/assets/statsrender.py:119`,
  `src/models/ltr_lgbm/report/renderutils.py:331`,
  `src/features/feat_utils.py:93`, `src/prepare/unzips.py:59`,
  `src/prepare/columns_identifier.py:213`,
  `src/qrates/multi_scann.py:111`. Same root cause each time (a
  `return` inside `finally` discards any pending exception); each was
  checked and confirmed not to additionally risk an `UnboundLocalError`
  (the returned variables are all pre-initialized before their `try`
  block). A near-duplicate of the mutable-default-argument `copymaps()`
  pattern also exists in `src/models/ltr_lgbm/report/renderpot.py:54`.

  `test/arycolbring_tests/t01_advisor.py` references a module-level
  `LocDir` that is only ever defined in a commented-out line;
  `AryColBring_Train_Test.__init__(output_dir=None)` and the
  `datapath` default would raise `NameError` if ever hit directly
  (currently masked because the CLI always supplies both `-o` and `-d`
  explicitly).
