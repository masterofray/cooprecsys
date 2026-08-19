# LightGBM Learning-to-Rank (LTR)
## Author
**Aryanto** -- [`mail me`](mailto:aryanto.dandan@gmail.com)

A beginner-friendly machine learning module for **ranking items in the right order** using **LightGBM Learning-to-Rank (LTR)**.
Instead of answering:
- "Will user buy this product?"
- "What is the predicted price?"
LTR answers:
> **Which item should appear first, second, third, and so on?**

This is useful for:
- Product recommendation  
- Search result ordering  
- Personalized product reorder prediction  
- Candidate ranking  
- Feed ranking  

## What is Learning-to-Rank?
Normal machine learning predicts a value or class. Example:
| Problem Type | Example Question |
|------------|------------------|
| Classification | Will customer buy this? |
| Regression | How much will sales be? |
| Ranking | Which product should be shown first? |

Learning-to-Rank focuses on **sorting multiple choices** based on relevance.

## When to Use LTR
Use LTR when order matters.
* Product recommendation
* Search engine ranking
* News feed sorting
* Ads ranking
* Personalized lists
Do NOT use when only predicting yes/no.

## Real Example
Imagine user opens shopping app. Possible products:

| Product | Probability User Likes |
|--------|-------------------------|
| Milk | High |
| Bread | Medium |
| Chocolate | Low |

LTR model learns to rank:
```text
1. Milk
2. Bread
3. Chocolate
````
So the best items appear first.

## Why Use LightGBM for Ranking?
LightGBM is popular because it is:
* Fast
* Accurate
* Handles many rows
* Handles many features
* Great for tabular data

For ranking tasks, LightGBM provides:
```python
objective = "lambdarank"
```
This special algorithm trains model to improve ranking quality.

### What is LambdaRank?
LambdaRank is a ranking algorithm used inside LightGBM. Instead of learning one row alone, it compares items **inside the same group**. Example for one user:
| User | Product | Bought Again |
| ---- | ------- | ------------ |
| A    | Milk    | 1            |
| A    | Bread   | 1            |
| A    | Candy   | 0            |

Model learns:
Milk and Bread should rank above Candy.
That is how ranking intelligence is learned.

## Important Concept: Query Group
LTR needs groups. Each group means:
* one user
* one search request
* one session
* one basket

Example:
```text
user_id = 1001
```
All products under user 1001 are ranked together. Without grouping, ranking will be wrong.

## What is NDCG?
NDCG is a ranking metric. It checks:
> Did the model place the most relevant items near the top?
Higher is better. For example `NDCG@5`, means evaluate top 5 ranked items. Our module uses:
* NDCG@5
* NDCG@10

## What This Module Does
This project automates the full ranking workflow:
```text
1. Read config
2. Prepare train/test data
3. Tune parameters (optional)
4. Train LightGBM ranking model
5. Create charts and report
6. Save model
7. Predict Top-K results
```
---

## Beginner-Friendly Flow
### Input Data
Example training data:
| user_id | product_id | orders_before | days_since_last | reordered |
| ------- | ---------- | ------------- | --------------- | --------- |
| 1       | Milk       | 12            | 5               | 1         |
| 1       | Bread      | 8             | 10              | 1         |
| 1       | Candy      | 1             | 40              | 0         |

Where:
* `user_id` = group
* `reordered` = target label
* others = features

### Output Result
Model predicts ranking:
| user_id | product_id | score | rank |
| ------- | ---------- | ----- | ---- |
| 1       | Milk       | 0.92  | 1    |
| 1       | Bread      | 0.74  | 2    |
| 1       | Candy      | 0.15  | 3    |

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
from ltr_lgbm import run_pipeline
from ..configs import LTRConfig

train_df     = pd.read_parquet("data/train.parquet")
test_df      = pd.read_parquet("data/test.parquet")
feature_cols = [c for c in train_df.columns if c not in {"user_id", "reordered"}]
config       = LTRConfig.from_ini("config.ini", features=feature_cols)

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
python -m ltr_lgbm.ltr_call \
    --train  data/train.parquet \
    --test   data/test.parquet  \
    --config config.ini         \
    --label  reordered          \
    --query-id user_id          \
    --run-name baseline_v1

# Skip Bayesian tuning:
python -m ltr_lgbm.ltr_call \
    --train data/train.parquet \
    --test  data/test.parquet  \
    --no-tuning
```

## Configuration.ini

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
| `TUNING`    | `sampler`              | `tpe` / `random` / `cmaes`                      |
| `TUNING`    | `pruner`               | `median` / `hyperband` / `none`                 |
| `INFERENCE` | `top_k`                | Items returned per query                        |
| `PATHS`     | `output_dir`           | Root artefact directory                         |
| `PATHS`     | `mlflow_tracking_uri`  | MLflow tracking server URI or local path        |
| `PATHS`     | `html_report_path`     | Output path for the HTML monitoring report      |

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


