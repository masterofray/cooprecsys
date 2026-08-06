# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

* **`ary2tower` Architecture & Structure**:
* Restructured package layout to mirror `arycolbring` with dedicated `inout/` and `narative/` subpackages:
* `inout/scaffold.py` (`TwoTowerBase`) and `inout/approximator.py` (`TwoTowerPredictor` containing `build_pairs()`, `predict()`, and `predict_rank()`).
* `inout/model_architect.py` (`TwoTowerArchitect` containing the Bayesian Personalized Ranking (BPR) training loop).
* `inout/fallback_reasoner.py` (`TwoTowerFallBack` providing purchase-aware filtering and item-to-item cosine-similarity cold-start fallback).


* Refactored `trainer.py` and `inference.py` into thin orchestration wrappers around new `inout/` classes while preserving existing public APIs.
* Implemented native single-page dashboard renderer under `narative/` (`rensupport.py`, `a2trearender.py`, `a2tadvirender.py`, class wrappers, templates, and static CSS/JS) styled with a light + orange theme.
* Updated `report.py` to delegate directly to the new native `narative/` renderer.
* Added end-to-end test suites `test/ary2tower_tests/t04_inout.py` and `t05_narative.py`.


* **Visualization & Reporting Framework**:
* Added static CPU plotting module (`src/models/ary2tower/viztower/`) supporting loss curves, metric bars, 2D embedding projections, similarity heatmaps, score distributions, precision/recall curves, and HTML gallery conversion.
* Added interactive dashboard reporting via `src/models/ary2tower/report.py` and `TwoTowerInference.generate_inference_report()`.
* Added test suites `t02_viztower.py` (17 tests) and `t03_report.py` (6 tests).



---

### Changed

* **`ary2tower` Core Neural Engine**:
* Added two-tower neural recommender model (`Embedding -> Dense -> ReLU -> Dense` per tower, dot-product/cosine similarity, pairwise BPR loss with SGD + momentum).
* Added Cython + OpenMP accelerated kernels in `CLtowers/` with automatic pure-NumPy fallback (`towers._HAS_CYTHON`) when extensions are unbuilt.
* Ported build setup scripts (`cysetup.py`, `cysetup.sh`, `cysetup.ps1`) from `arycolbring`.
* Added unit test suite `test/ary2tower_tests/t01_towers.py` (22 tests covering validation, training convergence, serialization, and inference).


* **Dashboard & User Interface**:
* Added **Insights** tab to the AryColBring inference dashboard (`infrc_insights.html.j2`, `insights.css`, `insights.js`) featuring prediction score histograms, 2D PCA embedding projections, and interactive Plotly similarity heatmaps.
* Updated `arycolbring` dashboard theme palette (`base.css` and static assets) to light background with orange accents (`#FF6B35` / `#F7931E` primary, `#4ECDC4` secondary).
* Added "Similar Items (Top 3)" column and interactive sorting to the Rankings table header.


* **Shared Assets & Infrastructure**:
* Extracted shared visualization builders into `src/assets/vizdata.py` (`score_distribution`, `embedding_projection_2d`, `similarity_heatmap`, `top_k_similar_items`).
* Refactored `src/assets/dashboard_utils.py` to consolidate shared scorecards, gauge generators, chart normalizers, and formatting utilities previously duplicated across renderers.
* Added 6 onboarding and comparison notebooks under `notebook/` (`01_Quick_Start` through `06_Cold_Start_Handling`).
* Added regression test suite `test/arycolbring_tests/t05_dashboard_refactor.py`.
* Extended CI workflow (`.github/workflows/arycolbring_pipeline.yml`) with `lint` and `pytest-suite` jobs, and expanded `pull_request` triggers to include `main`.
* Added 26 unit tests in `test/arycolbring_tests/t03_pytest.py` for shared parameter validation and utility functions.


* **Design Choices**:
* Retained CPU-first Cython + OpenMP architecture for `ary2tower`, explicitly declining PyTorch/GPU/DDP hardware dependencies to maintain repository-wide consistency and zero-GPU operational capability.

---

### Fixed

* **Recommendation Shortfall & Candidate Filtering**:
* Fixed an issue where `recommend()` and `batch_recommend()` returned fewer than `n_items` results for users with heavy purchase history due to post-truncation filtering.
* Added `exclude_purchased: bool` parameter to score full candidate pools prior to filtering, backed by `TwoTowerFallBack` cosine similarity fallback for edge-case catalogue exhaustion.
* Fixed a code edit error in `t01_towers.py` where a dropped `class TestTwoTowerInference:` declaration merged test scopes.


* **Imports & Execution Reliability**:
* Fixed incorrect relative import depth in `ary2tower/narative/a2trearender.py`.
* Fixed exception swallowing in `get_env()` (`narative/rensupport.py`) caused by an unconditional `return` statement inside a `finally` block.
* Replaced mutable default argument (`dirlist: List = ['css', 'js']`) in `copymaps()` with a `None` sentinel.


* **Metrics & Formatting**:
* Removed hardcoded `0.75` coverage metric baseline from inference reports.
* Prevented offline evaluation metrics (Precision, Recall, NDCG) from rendering on inference dashboards.
* Fixed label parsing bug in `bealabel()` where `ndcg_at_10` rendered as `"Ndcg @ 10"` instead of `"NDCG@10"`.


* **Test Suite & CI Alignment**:
* Corrected `t03_pytest.py` assertions to expect validation errors at instance initialization rather than fit time.
* Updated dataframe column lookups in `TestDataHandling` to match actual output keys (`n_users`, `n_items`, `nnz`, `density`).
* Corrected `pyproject.toml` coverage target from `cooprecsys` to `src`.
* Updated `pytest` discovery rules in `pyproject.toml` to include `t03_pytest.py` and `t05_dashboard_refactor.py`.
* Added missing development dependencies (`pytest-cov`, `black`, `flake8`, `isort`) to `pyproject.toml`.

---

### Known Limitations

* The Cython/OpenMP kernels (`CLtowers/*.pyx`) require compilation via `cysetup.sh` / `cysetup.ps1` before enabling the hardware-accelerated execution path (`towers._HAS_CYTHON == True`).
* Coverage failure thresholds (`--cov-fail-under`) remain unconfigured pending an established baseline run.
* Legacy `finally: return` exception-swallowing patterns remain unpatched in 6 external files outside current scope (`statsrender.py`, `renderutils.py`, `feat_utils.py`, `unzips.py`, `columns_identifier.py`, `multi_scann.py`).
* Missing default value for `LocDir` in `test/arycolbring_tests/t01_advisor.py` requires explicit CLI parameters (`-o` and `-d`) during execution.