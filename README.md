[![CoopRecSys CI CD Pipeline](https://github.com/masterofray/cooprecsys/actions/workflows/pipeline.yml/badge.svg?branch=dev)](https://github.com/masterofray/cooprecsys/actions/workflows/arycolbring_flow.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Python](https://img.shields.io/badge/code%20style-python-blue)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masterofray/cooprecsys/blob/dev/notebook/AryColBring_Training_Pipeline.ipynb)

# CoopRecSys v0.1.1
<p align="center">
  <img src="https://ik.imagekit.io/arydatalabs/cooprecsys/cooprecsys_banner.jpg" alt="CoopRecSys v0.1.1 Banner" width="100%">
</p>

**Cooperative Recommender System ML/AI Module Release 0.1.1**
A production-grade machine learning and AI module for building intelligent recommendation systems tailored for cooperative (koperasi) product recommendations. This system combines collaborative filtering, learning-to-rank techniques, and explainable AI dashboards.

> **Current release is v0.1.1**: a packaging, native-extension compatibility, build-system, and PyPI distribution hardening release. The release preserves the core Cython implementations while improving reproducible Linux wheel builds, cross-platform packaging, artifact validation, and PyPI metadata compliance.

### Release Highlights

- Linux native extensions are built in a controlled manylinux-compatible environment.
- AryColBring and Ary2Tower Cython implementations are retained; the hardening is concentrated in the build pipeline.
- Python wheel coverage targets CPython 3.10–3.13.
- Windows wheels continue to use the native MSVC toolchain.
- Release artifacts are validated before PyPI publication.
- Package metadata is aligned with the official PyPI classifier taxonomy.

---

## Release 0.1.1

CoopRecSys v0.1.1 is the current released package version. This release focuses on packaging reliability, native-extension compatibility, reproducible wheel generation, cross-platform distribution, and PyPI metadata compliance while preserving the core recommendation-model implementations.

For installation, use the published release directly from PyPI:

```bash
pip install cooprecsys==0.1.1
```


## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Notebooks](#notebooks)
- [Architecture](#architecture)
- [Dashboard & Explainability](#dashboard--explainability)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)
- [Changelog](CHANGELOG.md)

---

## Features

### **Advanced Recommendation Models**

- **AryColBring**: Ultra-optimized collaborative filtering powered by Cython + OpenMP
  - Multi-loss support: Logistic, WARP, BPR, WARP-kOS
  - Sparse matrix acceleration with DuckDB integration
  - Per-thread buffers with no Python GIL in hot paths
  
- **LTR-LightGBM**: Learning-to-Rank using LightGBM
  - Group-aware train/test splitting
  - Ranking metrics: NDCG, MAP, AUC
  - MLflow experiment tracking

- **ary2tower**: Two-tower neural recommender (Cython + OpenMP)
  - `Embedding -> Dense -> ReLU -> Dense` per tower, dot-product/cosine similarity
  - BPR-style pairwise training with SGD + momentum
  - Automatic pure-NumPy fallback when the compiled extension isn't built
    (see `src/cooprecsys/models/ary2tower/README.md`)

### **Explainable AI Dashboard**

- Interactive HTML-based dashboard powered by JavaScript
- Feature importance visualization
- Model prediction explanations
- Real-time ranking visualization
- SHAP-style local interpretability

### **Performance Optimizations**

- **69.7% Python** for core logic and data processing
- **9.7% Cython** for high-performance numerical kernels
- **14.9% CSS** for responsive UI styling
- **2.9% JavaScript** for interactive dashboards
- Multi-threading support with joblib parallelization
- Sparse matrix operations with SciPy

### **Production Ready**

- CI/CD Pipeline with GitHub Actions
- Comprehensive error handling and logging
- Model persistence with cloudpickle
- DuckDB-backed data ingestion
- TQDM progress bars for user feedback

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10+ |
| **Performance** | Cython | 3.0.1+ |
| **ML Framework** | LightGBM | Latest |
| **Matrix Ops** | NumPy, SciPy | 1.21+, 1.7+ |
| **Data Processing** | Pandas, DuckDB | 1.3+, 0.8+ |
| **Experiment Tracking** | MLflow | 1.20+ |
| **Parallelization** | joblib | 1.1+ |
| **Visualization** | Matplotlib, Seaborn | 3.4+, 0.11+ |
| **Frontend** | HTML5, CSS3, JavaScript | Modern |

---

## Installation

### Recommended: Install from PyPI

```bash
pip install cooprecsys==0.1.1
```

Or upgrade an existing installation:

```bash
pip install --upgrade cooprecsys
```

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install build-essential python3-dev gcc g++ make

# macOS
brew install gcc llvm libomp
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/masterofray/cooprecsys.git
   cd cooprecsys
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Compile native Cython extensions (repository development only)**
   The published v0.1.1 wheels contain the compiled native extensions for supported platforms. Manual Cython compilation is primarily required when developing directly from source.

---

## Quick Start

### Using the Test Suite

The test suite demonstrates how to use both recommendation models:

#### LTR LightGBM Example

```bash
python -m test.ltrlgbm_test.ltrlgbm_example
```

**Test location**: `test/ltrlgbm_test/ltrlgbm_example.py`

This script:
- Loads sample data from `data/sampledata.parquet`
- Performs group-aware train/test splitting
- Trains the LightGBM LTR model with hyperparameter tuning
- Generates rankings and MLflow tracking
- Produces HTML reports and artifact visualizations

#### LTR Inference Test

```bash
python -m test.ltrlgbm_test.ltrlgbm_inferencing
```

**Test location**: `test/ltrlgbm_test/ltrlgbm_inferencing.py`

This script demonstrates inference using trained models with fallback strategies.

#### Collaborative Filtering Tests

```bash
pytest test/arycolbring_tests/test_model.py -v
pytest test/arycolbring_tests/test_evaluation.py -v
pytest test/arycolbring_tests/test_data_utils.py -v
```

#### Ary2Tower Smoke and Unit Tests

```bash
python test/ary2tower_tests/ary2tower_train_smoketest.py
python test/ary2tower_tests/ary2tower_inference_smoketest.py
pytest test/ary2tower_tests/t03_pytest.py -v --tb=short
```

The Ary2Tower test suite uses the production `data/sampledata.parquet` path and directly exercises the model modules under `src/cooprecsys/models/ary2tower/`.


**Test locations**:
- `test/arycolbring_tests/test_model.py` - Model initialization and fitting
- `test/arycolbring_tests/test_evaluation.py` - Evaluation metrics (Precision@k, Recall@k, AUC)
- `test/arycolbring_tests/test_data_utils.py` - Data loading and preprocessing
- `test/arycolbring_tests/test_cross_validation.py` - Train/test splitting strategies

---

## Usage Examples

### Example 1: LTR LightGBM Pipeline

```python
from cooprecsys.configs import LTRConfig
from cooprecsys.models.ltr_lgbm import lgbm_fit_transform
import pandas as pd

# Load your data
data = pd.read_parquet('data/sampledata.parquet')

# Split data by customer ID (group-aware)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
train_idx, test_idx = next(gss.split(data, groups=data['CustomerID']))
train_df = data.iloc[train_idx]
test_df = data.iloc[test_idx]

# Configure model
config = LTRConfig.from_ini('src/cooprecsys/configs/configuration.ini', 
                             features=['ProductName', 'ProductPrice', ...])
config.feature.label = 'CategoryID'
config.feature.query_id = 'CustomerID'

# Train with hyperparameter tuning
trainer = lgbm_fit_transform(
    config=config,
    train=train_df,
    test=test_df,
    run_tuning=True,
    run_name="my_recommendation_model"
)

# Access results
print(f"Best Iteration: {trainer.best_iteration}")
print(f"Runtime: {trainer.runtime_minutes} minutes")
```

### Example 2: Collaborative Filtering with AryColBring

```python
from cooprecsys.models.arycolbring import AryColBring
import scipy.sparse as sp
import numpy as np

# Create sparse interaction matrix (n_users × n_items)
interactions = sp.coo_matrix(
    (np.ones(1000), (np.random.randint(0, 100, 1000), 
                     np.random.randint(0, 50, 1000))),
    shape=(100, 50)
)

# Initialize model
model = AryColBring(
    no_components=32,
    loss='warp',  # or 'logistic', 'bpr', 'warp-kos'
    learning_rate=0.05,
    random_state=42
)

# Train
model.fit(interactions, epochs=10, num_threads=4)

# Get recommendations
user_id = 0
item_scores = model.predict([user_id] * 50, np.arange(50))
top_items = np.argsort(item_scores)[::-1][:10]
```


### Example 3: Two-Tower Recommendation with Ary2Tower

```python
from cooprecsys.models.ary2tower import TwoTowerConfig, TwoTowerTrainer, TwoTowerInference

config = TwoTowerConfig(
    embedding_dim=32,
    hidden_dims=(64, 32),
    learning_rate=0.01,
    epochs=5,
    random_state=42,
)

trainer = TwoTowerTrainer(config)
trainer.fit(...)

trainer.save_model("artifacts/ary2tower/model.npz")

inference = TwoTowerInference.load_model("artifacts/ary2tower/model.npz")
scores = inference.predict(...)
recommendations = inference.recommend(...)
```

For production data preparation, use the loaders and preparation utilities under `cooprecsys.features`, `cooprecsys.prepare`, and `cooprecsys.noisemaker` rather than constructing an ad-hoc interaction matrix.

---

## Notebooks

Step-by-step walkthroughs live in `notebook/`:

| Notebook | Covers |
|---|---|
| `01_Quick_Start.ipynb` | Train -> serve in 5 minutes |
| `02_Data_Preparation.ipynb` | Raw transactions -> validated sparse matrix |
| `03_Training_AryColBring.ipynb` | Hyperparameter sweep vs. a random baseline |
| `04_Interactive_Dashboard.ipynb` | Generates the light/orange inference dashboard inline |
| `05_LTR_LGBM_Comparison.ipynb` | AryColBring vs. LTR-LGBM evaluation methodology |
| `06_Cold_Start_Handling.ipynb` | Purchase-aware filtering + item-to-item fallback (`AryInfFallBack`) |
| `AryColBring_Training_Pipeline.ipynb` | Full annotated walkthrough of the training pipeline internals |

---

## Architecture

> The tree below reflects the current package-oriented `src/cooprecsys/` layout and highlights the major runtime, model, feature, dashboard, and data-access components. (e.g. the
> real training entry point is `trainer.py`, inference is `inference.py`,
> Cython kernels live in `CLproximity/`, and the dashboard renderers are
> `narative/advirender.py` / `narative/rearender.py` under a light-theme,
> orange-accent (`#FF6B35`) design system, not `dashboard/index.html`).
> New since the last update: `src/cooprecsys/models/ary2tower/` (two-tower neural
> recommender) and `notebook/` (usage walkthroughs) -- both shown below.

```text
cooprecsys/
├── src/
│   └── cooprecsys/
│       ├── assets/                    # Dashboard/static asset utilities
│       ├── configs/                   # Model and runtime configuration
│       ├── db/                        # DuckDB integration
│       ├── features/                  # Feature engineering and preprocessing
│       ├── models/
│       │   ├── arycolbring/           # Cython collaborative filtering
│       │   │   ├── CLproximity/       # Native numerical kernels
│       │   │   ├── inout/             # Training/inference adapters
│       │   │   ├── narative/          # Reports and explainability rendering
│       │   │   ├── eval/              # Evaluation and ranking metrics
│       │   │   ├── assist/            # Supporting utilities
│       │   │   └── inference.py / trainer.py
│       │   ├── ary2tower/              # Two-tower neural recommender
│       │   │   ├── CLtowers/           # Cython + OpenMP kernels
│       │   │   ├── inout/              # Model I/O and fallback logic
│       │   │   ├── narative/           # Training/inference reports
│       │   │   ├── viztower/           # Embedding and performance visualizations
│       │   │   └── config.py / towers.py / trainer.py / inference.py
│       │   └── ltr_lgbm/              # Learning-to-Rank with LightGBM
│       │       ├── ftcore
│       │       ├── inout
│       │       ├── report
│       │       ├── __init__.py
│       │       ├── dataprepared.py
│       │       ├── ltr_call.py
│       │       ├── ltr_predict.py
│       │       └── readme.md
│       ├── noisemaker/                # Data/noise utilities
│       ├── prepare/                   # Dataset preparation utilities
│       ├── qrates/                    # Ranking/quality-rate utilities and SQL
│       └── __init__.py
├── notebook/                          # Usage and training walkthroughs
├── test/                              # Unit and integration tests
├── .github/workflows/                 # CI/CD automation
├── docs/                              # Astro documentation site
├── pyproject.toml                     # Package/build metadata
└── README.md
```

---

## Dashboard & Explainability

**CoopRecSys Explainable AI Dashboard** is a lightweight web interface implemented with **Jinja2 templates**, **JavaScript**, and **CSS**. It is intended for production monitoring and diagnostic workflows of an LTR LightGBM ranking model, presenting concise, actionable metrics, temporal trends, and dataset context. The dashboard enables data scientists and engineers to rapidly assess model health, detect anomalies or drift, and investigate root causes of performance changes through an integrated explainability‑focused view. The dashboard provides real-time explanations for model predictions:

### Features

- **Feature Importance**: Visualization of which features drive recommendations
- **Ranking Explanations**: Why specific items are ranked higher
- **Comparison View**: Side-by-side model performance comparison
- **Interactive Filters**: Drill down by user, item, or category
- **Export Reports**: Generate PDF/HTML reports with explanations

> The AryColBring **inference** dashboard (`narative/rearender.py`) uses
> a light theme with orange accents (`#FF6B35`), and now has an
> **Insights** tab (prediction score histogram, 2D embedding PCA
> projection, item-item similarity heatmap) in place of the ranking-
> quality metrics that used to leak into it -- Precision/Recall/NDCG/
> AUC/MRR belong on the **training** dashboard
> (`narative/advirender.py`), not a production inference report.

### Running the Dashboard

```bash
# Start the web server
python -m http.server 8000 --directory ./artifacts/reports

# Open browser to http://localhost:8000/20260528_training_report.html
```

### Overviews Page
<table align       = "center" 
       bgcolor     = "#ffffff"
       cellpadding = "14" cellspacing="0"
       style       = "border-collapse:collapse;">
    <tr><td>
    <img src="https://ik.imagekit.io/arydatalabs/cooprecsys/dashboard01_overviews.jpg"
         alt="Overview of CoopRecSys Explainable AI Dashboard"
         style="display:block; max-width:100%; height:auto;"/>
    </td></tr>
    <tr><td align="center" style="padding-top:8px; color:#333333; font-size:0.95rem;">
    <strong>Figure 1.</strong> Overview of the CoopRecSys Explainable AI Dashboard.
    </td></tr>
</table>

Figure 1 displays the primary overview screen of the dashboard, combining a high‑level scorecard, temporal visualizations, navigation tabs, and dataset context to provide an immediate assessment of model status. **Key elements and purpose**:
- **Header** — identifies the dashboard and the monitored model for orientation.  
- **Navigation Tabs** — Overview, Rankings, Diagnostics, Config for structured access to analytical modules.  
- **Scorecard Metrics** — compact presentation of prediction statistics (**PRED MAX**, **PRED MEAN**, **PRED MIN**, **PRED STD**) and ranking performance (**NDCG@5**, **NDCG@10** for train and test) for rapid appraisal.  
- **Control Panel Visitor Analytics** — summary count of recent predictions and an interactive multi‑series line chart to reveal trends, spikes, or drift.  
- **Sidebar Dataset Statistics** — contextual counts such as **USERS**, **PRODUCTS**, **CATEGORIES**, and **ROWS** to indicate scale and coverage.  
- **Multi‑series Line Chart** — overlays prediction and evaluation metrics to facilitate correlation analysis and anomaly detection.

---

### Rankings Page
<table align       = "center" 
       bgcolor     = "#ffffff"
       cellpadding = "14" cellspacing="0"
       style       = "border-collapse:collapse;">
    <tr><td>
    <img src="https://ik.imagekit.io/arydatalabs/cooprecsys/dashboard02_rankings.jpg"
         alt="Rankings of CoopRecSys Explainable AI Dashboard"
         style="display:block; max-width:100%; height:auto;"/>
    </td></tr>
    <tr><td align="center" style="padding-top:8px; color:#333333; font-size:0.95rem;">
    <strong>Figure 2.</strong> Rankings CoopRecSys Dashboard.
    </td></tr>
</table>

The **Rankings** page provides a transparent, interactive view of model inference results and the feature context that produced each ranking. It is intended for analysts and engineers who require a sortable, filterable listing of top predictions together with per‑row explainability signals so that individual decisions can be inspected, validated, and traced back to input features. 

#### Key Components
- **Ranking Results Table** — Primary component showing the top N predictions produced by the LTR LightGBM model with configurable columns for identifiers, features, prediction score, and explainability metrics.  
- **Row Explainability Panel** — Per‑row detail pane that surfaces feature contributions (e.g., SHAP values), top contributing features, and short textual explanation for the predicted rank.  
- **Filters and Facets** — Controls to restrict the table by date range, user segment, product category, prediction score range, or custom tags.  
- **Sorting and Pagination** — Stable, server‑side or client‑side sorting by score and any feature column, with efficient pagination for large result sets.  
- **Export and Snapshot** — Export current view to CSV and capture a snapshot (timestamped) of the displayed ranking for audit or reporting.  
- **Contextual Metadata** — Small summary area showing dataset scope (rows, users, products), generation timestamp, and model version used for the ranking.

#### Interactions and Controls
- **Global Filters** — Date range picker; dropdowns for product category and user segment; numeric sliders for prediction score and years working.  
- **Column Filters** — Per‑column quick filters (text search, numeric range).  
- **Row Inspection** — Clicking a row opens the **Row Explainability Panel** with:  
  - Full feature vector for that row.  
  - SHAP waterfall or bar chart showing positive and negative contributions.  
  - A short natural language explanation generated from the top contributions.  
- **Compare Mode** — Select two or more rows to view a side‑by‑side comparison of features and contributions.  
- **Server Mode** — For large datasets, enable server‑side pagination and sorting; otherwise use client‑side DataTables for small to medium result sets.  
- **Audit Trail** — Each exported snapshot includes metadata: model version, timestamp, and filter state.

---

### Diagnostics Page
<table align       = "center" 
       bgcolor     = "#ffffff"
       cellpadding = "14" cellspacing="0"
       style       = "border-collapse:collapse;">
    <tr><td>
    <img src="https://ik.imagekit.io/arydatalabs/cooprecsys/dashboard03_diagnostics.jpg"
         alt="Diagnostics of CoopRecSys Explainable AI Dashboard"
         style="display:block; max-width:100%; height:auto;"/>
    </td></tr>
    <tr><td align="center" style="padding-top:8px; color:#333333; font-size:0.95rem;">
    <strong>Figure 3.</strong> diagnostics Graph for CoopRecSys model.
    </td></tr>
</table>

The **Diagnostics** page provides a consolidated environment for model introspection and validation. It combines global diagnostics (feature importance and distributional checks), prediction‑level diagnostics (relevance score histograms and drift indicators), and per‑sample explainability artifacts (SHAP summaries and downloadable SHAP files). The page is intended for data scientists, ML engineers, and auditors who require both high‑level signals and the ability to drill into individual explanations.

#### Key Components
- **Feature Importance Chart** — Horizontal bar chart showing top features by chosen importance metric (GAIN, SPLIT, or permutation importance). Interactive: sort, change metric, and toggle top‑K.
- **Histogram of Relevance Predictions** — Binned histogram of model relevance scores (test / production) with overlayed reference distribution (train) and summary statistics (mean, median, std, skewness).
- **SHAP Samples Panel** — A sample browser that lists available SHAP files (timestamped), allows download, and previews selected samples with a SHAP waterfall or bar chart and raw feature vector.
- **Drift & Distribution Alerts** — Small indicator cards that flag features with significant distributional shift (KS test, PSI) and prediction drift (population mean shift).
- **Sample Inspector** — On selecting a sample from the SHAP list or from the top predictions, show: raw features, SHAP contributions (positive/negative), cumulative contribution to score, and a short natural‑language explanation.
- **Export & Audit** — Buttons to export diagnostics snapshot (CSV/JSON) and to attach model version, feature engineering commit, and timestamp for reproducibility.

#### Operational and Implementation Notes
- **Server responsibilities**  
  - Precompute feature importance (GAIN/SPLIT) and expose as JSON.  
  - Provide binned relevance distributions for train/test/production to avoid heavy client computation.  
  - Serve SHAP files on demand and paginate sample lists for large files.

- **Performance**  
  - For large SHAP files, fetch only sample metadata for the list and request full sample SHAP vectors when the user inspects a sample.  
  - Use server‑side aggregation for histograms and KS/PSI calculations.

- **Accessibility & UX**  
  - Ensure charts have `aria-label` and textual summaries for screen readers.  
  - Allow keyboard navigation for the SHAP sample list and close preview with `Esc`.  
  - Provide clear tooltips explaining each diagnostic metric (e.g., GAIN vs SPLIT).

- **Reproducibility & Audit**  
  - Every diagnostics snapshot must include **model version**, **feature engineering commit hash**, **data window**, and **timestamp**.  
  - Exported diagnostics should embed this metadata.

---

### Configs Page
<table align       = "center" 
       bgcolor     = "#ffffff"
       cellpadding = "14" cellspacing="0"
       style       = "border-collapse:collapse;">
    <tr><td>
    <img src="https://ik.imagekit.io/arydatalabs/cooprecsys/dashboard04_configs.jpg"
         alt="Configs of CoopRecSys table Model"
         style="display:block; max-width:100%; height:auto;"/>
    </td></tr>
    <tr><td align="center" style="padding-top:8px; color:#333333; font-size:0.95rem;">
    <strong>Figure 4.</strong> Configs table as parameter LTR LGBM.
    </td></tr>
</table>

The **Config** page centralizes model configuration and hyperparameter management for the LTR LightGBM ranking model. It provides a controlled interface to **view**, **edit**, **validate**, **version**, and **apply** training and inference parameters while preserving auditability and reproducibility. The page is intended for ML engineers and platform operators who must safely tune model behavior in production or prepare reproducible training runs.

#### Key Components
- **Configuration Table** — Tabular display of current parameter names and values (e.g., `objective`, `metric`, `ndcg_eval_at`, `learning_rate`, `max_depth`, `num_leaves`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`). Each row shows **Parameter**, **Value**, **Type**, **Source** (default / experiment / production), and **Last modified** timestamp.
- **Edit Controls** — Inline editors for editable parameters with appropriate input types: numeric fields, dropdowns for enumerated options, multi-value arrays for list parameters, and toggles for booleans. Edits are staged until explicitly saved.
- **Validation Engine** — Client and server validation rules that enforce type constraints, allowed ranges, and inter‑parameter consistency (e.g., `num_leaves` consistent with `max_depth`, `feature_fraction` in (0,1]).
- **Preview and Dry Run** — A preview panel that shows the effective configuration JSON and a dry‑run button that triggers a lightweight validation job (no training) to check compatibility with current feature schema and training pipeline.


---

## Configuration

### Environment Variables

```bash
# MLflow tracking
export MLFLOW_TRACKING_URI=file:./mlruns
# or for remote server
export MLFLOW_TRACKING_URI=http://localhost:5000

# Logging
export LOG_LEVEL=DEBUG
```

### Configuration File

Edit `src/cooprecsys/configs/configuration.ini`:

```ini
[model]
loss = warp
num_components = 32
learning_rate = 0.05
epochs = 10

[data]
test_size = 0.2
random_state = 42

[mlflow]
experiment_name = cooprecsys_prod
```

---

## Testing

### Run All Tests

```bash
# Integration tests
python -m test.ltrlgbm_test.ltrlgbm_example

# Unit tests (arycolbring)
pytest test/arycolbring_tests/t03_pytest.py test/arycolbring_tests/t05_dashboard_refactor.py -v --tb=short

# Unit tests (ary2tower)
pytest test/ary2tower_tests/t03_pytest.py -v --tb=short

# Full test suite with coverage
pytest test/ --cov=cooprecsys --cov-report=html

# Lint
black --check src test && flake8 src test && isort --check-only src test
```

### CI/CD Pipeline

All tests run automatically on:
- Push to `master`/`dev` branches
- Pull requests (including into `main`)
- Manual workflow dispatch

`arycolbring_pipeline.yml` additionally runs a dedicated `lint` job
(black/flake8/isort) and a `pytest-suite` job (coverage-reported,
artifact-uploaded) alongside the existing multi-Python-version smoke
test.

**CI/CD Pipeline**: https://github.com/masterofray/cooprecsys/actions/workflows/pipeline.yml

### Test Results

Tests verify:
[x] Model initialization and parameter validation
[x] Data loading and preprocessing
[x] Training and inference
[x] Evaluation metrics computation
[x] Cross-validation splitting strategies
[x] Security checks (Bandit)

---

## Performance

### Benchmarks (Sample Dataset)

| Model | Training Time | Inference Time | Memory |
|-------|----------------|-----------------|--------|
| AryColBring (WARP) | 2.1s | 0.3s | 45MB |
| LTR-LightGBM | 5.4s | 0.8s | 120MB |
| Ensemble | 8.2s | 1.2s | 180MB |

**Dataset**: 100k interactions, 5k users, 2k items
**Hardware**: 4-core CPU, 8GB RAM

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Run tests locally (`pytest test/arycolbring_tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests for new code
- Update README for new features

---

## Author & Maintainer

**Aryanto (masterofray)** — Author and Maintainer

- Email: [aryanto.dandan@gmail.com](mailto:aryanto.dandan@gmail.com)
- GitHub: [@masterofray](https://github.com/masterofray)
- LinkedIn: [linkedin.com/in/aryanto-ray](https://www.linkedin.com/in/aryanto-ray)
- AI / Data Science Portfolio: [ai.arydatalabs.workers.dev](https://ai.arydatalabs.workers.dev/)

---

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

---

## References

- [AryColBring Model Documentation](src/cooprecsys/models/README.md)
- [Collaborative Filtering - ACM](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Learning-to-Rank Overview](https://en.wikipedia.org/wiki/Learning_to_rank)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/)
- [Cython Documentation](https://cython.readthedocs.io/)

---

## Changelog

Release history is maintained separately in [CHANGELOG.md](CHANGELOG.md).

---

## Support & Issues

- **Report Issues**: [GitHub Issues](https://github.com/masterofray/cooprecsys/issues)
- **Documentation**: See [Documentation](https://masterofray.github.io/cooprecsys/) as published documentation site
- **PyPI**: https://pypi.org/project/cooprecsys/
- **Email**: [aryanto.dandan@gmail.com](mailto:aryanto.dandan@gmail.com)

---

<div align="center">
**Built HARD for better product recommendations in cooperative systems**
**If you find this useful, please star the repository and buy me coffee!**
</div>
