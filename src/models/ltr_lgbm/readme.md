# LightGBM LTR Framework

**Production-grade Learning-to-Rank (LTR) pipeline built on LightGBM.**

Designed for maximum flexibility — dataset-agnostic, fully typed, and
observable from end to end.

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 2. Programmatic usage

```python
import pandas as pd
from ltr_framework import LTRConfig, run_pipeline

train_df = pd.read_parquet("data/train.parquet")
test_df  = pd.read_parquet("data/test.parquet")

feature_cols = [c for c in train_df.columns if c not in {"user_id", "reordered"}]

config = LTRConfig.from_ini("config.ini", features=feature_cols)

trainer = run_pipeline(
    config     = config,
    train_df   = train_df,
    test_df    = test_df,
    run_tuning = True,        # Set False to skip Bayesian tuning
    run_name   = "my_run_v1",
)

# Booster is available immediately
booster = trainer.model
```

### 3. CLI usage

```bash
python -m ltr_framework.main \
    --train  data/train.parquet \
    --test   data/test.parquet  \
    --config config.ini         \
    --label  reordered          \
    --query-id user_id          \
    --run-name baseline_v1

# Skip Bayesian tuning:
python -m ltr_framework.main \
    --train data/train.parquet \
    --test  data/test.parquet  \
    --no-tuning
```

---

## Configuration (config.ini)

| Section     | Key                    | Description                                     |
|-------------|------------------------|-------------------------------------------------|
| `FEATURES`  | `label`                | Relevance label column                          |
| `FEATURES`  | `query_id`             | Query / group ID column                         |
| `MODEL`     | `model_path`           | Path to save/load the booster `.txt`            |
| `MODEL`     | `large_data_threshold` | Rows above which parallel processing activates  |
| `TRAINING`  | `num_boost_round`      | Max boosting iterations                         |
| `TRAINING`  | `learning_rate`        | Default LR (overridden by Bayesian tuning)      |
| `TUNING`    | `n_trials`             | Optuna trial count                              |
| `TUNING`    | `timeout`              | Wall-clock budget (seconds)                     |
| `TUNING`    | `sampler`              | `tpe` / `random` / `cmaes`                     |
| `TUNING`    | `pruner`               | `median` / `hyperband` / `none`                |
| `INFERENCE` | `top_k`                | Items returned per query                        |
| `PATHS`     | `output_dir`           | Root artefact directory                         |
| `PATHS`     | `mlflow_tracking_uri`  | MLflow tracking server URI or local path        |
| `PATHS`     | `html_report_path`     | Output path for the HTML monitoring report      |

---

## Module Reference

### `DataProcessor`

- Always sorts data by `query_id` via an in-memory DuckDB connection.
- Activates `ProcessPoolExecutor` for train + test splits when row count
  exceeds `large_data_threshold`.
- Populates `X_train`, `y_train`, `group_train`, `X_test`, `y_test`,
  `group_test` on `self`.

### `BayesianTuner`

- Uses Optuna with a configurable sampler (TPE / CMA-ES / Random).
- Each trial runs `lgb.cv` with early stopping — no data leakage.
- Every trial is logged as a nested MLflow child run.
- Best params are merged back into `config.training.params` automatically.

### `LTRTrainer`

- Wraps `lgb.train` with early stopping, `lgb.record_evaluation`, and a
  `tqdm` progress callback.
- Exposes `evals_result`, `metrics`, `runtime_minutes`, `best_iteration`.

### `LTRInference`

- Lazy-loads the booster on first use if not injected at construction.
- `rank_top_k(df)` scores all rows, then keeps top-K per query group.
- `score_single_query(query_df)` — ultra-low-latency per-query scoring.
- Rankings are saved as CSV via `save_rankings()`.

### `Visualizer`

Charts produced:

| Chart                    | Filename                        |
|--------------------------|---------------------------------|
| Feature Importance       | `feature_importance.png`        |
| Prediction Distribution  | `prediction_distribution.png`   |
| Learning Curves          | `learning_curves.png`           |
| Metrics Summary          | `metrics_summary.png`           |
| Feature Correlation      | `feature_correlation.png`       |
| HTML Report              | `outputs/report.html`           |

### `MLflowMonitor`

Used as a context manager:

```python
with MLflowMonitor(config, run_name="run_v1") as monitor:
    monitor.log_params(config.training.params)
    monitor.log_metrics(trainer.metrics)
    monitor.log_model(trainer.model)
    monitor.log_artifacts(["outputs/report.html"])
```

---

## Key Design Patterns

- **`@dataclass`** for all config objects — type-safe, IDE-friendly.
- **`@property` / `@setter`** for lazy state (e.g. booster lazy-load in `LTRInference`).
- **State-on-self** — methods update `self.*` rather than returning values,
  enabling clean pipeline composition.
- **Strict type hints** throughout — compatible with `mypy --strict`.
- **`tqdm`** progress bars on every major blocking step.
- **`logging`** (not `print`) at `DEBUG` / `INFO` / `WARNING` / `ERROR` levels.

---

## Output Artefacts

```
outputs/
├── lightgbm_ltr_model.txt        ← Saved LightGBM booster
├── feature_importance.png
├── prediction_distribution.png
├── learning_curves.png
├── metrics_summary.png
├── feature_correlation.png
├── rankings_top20.csv            ← Per-query Top-K inference output
└── report.html                   ← Self-contained HTML monitoring report

mlruns/                           ← MLflow tracking directory
```

---

## Author

**Aryanto** — aryanto.dandan@gmail.com
