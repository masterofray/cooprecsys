#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-02"

"""
ltrlgbm_example.py
__________________________________
A demonstration script to test the complete LightGBM LTR framework.
This script loads 'sampledata.parquet', prepares a group-aware train/test split,
configures the pipeline, and triggers all downstream LTR processes.
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from configs import LTRConfig, logger
from models.ltr_lgbm import lgbm_fit_transform



def prepare_dummy_config(config_path: str = "demo_config.ini"):
    """
    Creates a temporary ini file to initialize LTRConfig if you don't 
    already have one set up for this specific demo.
    """
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""[model]
experiment_name = LTR_Demo_Parquet
seed = 42
model_path = ./output/model.txt

[training]
num_boost_round = 100
early_stopping_rounds = 20
log_evaluation = 10

[tuning]
n_trials = 5
timeout = 600
sampler = tpe
pruner = median
study_name = ltr_demo_study
direction = maximize
run_mlflow = True

[inference]
top_k = 5
score_col = predicted_score

[path]
output_dir = ./output
mlflow_tracking_uri = ./mlruns
html_report_path = ./output/monitoring_report.html

[logging]
OptunaLevel = WARNING
""")
        print(f"Created temporary config file: {config_path}")


def main():
    print("=" * 60)
    print("  Initializing LTR Framework Demo")
    print("=" * 60)

    # 1. Load Data
    data_path = "sampledata.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Please place it in the same directory.")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # 2. Define Schema & Features
    # For LTR, we need a query group (CustomerID) and a relevance label.
    # We will use 'Quantity' as our target relevance score (higher quantity = more relevant).
    query_col = "CustomerID"
    label_col = "Quantity"

    # Selecting strictly numerical columns to prevent LightGBM categorical errors during this demo
    feature_cols = [
        "ProductPrice", 
        "Discount", 
        "CategoryID", 
        "VitalityDays", 
        "EmployeeAge", 
        "YearsWorking"
    ]

    # Clean missing values for the demo to ensure smooth sailing
    df[feature_cols] = df[feature_cols].fillna(0)
    df[label_col] = df[label_col].fillna(0)

    # LTR strictly requires data to be sorted by the query/group ID
    df = df.sort_values(query_col).reset_index(drop=True)

    # 3. Group-Aware Train/Test Split
    # We MUST split by CustomerID so the same customer isn't in both Train and Test (Data Leakage)
    print("Splitting data into Train and Test sets (Grouped by CustomerID)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[query_col]))

    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    print(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")

    # 4. Initialize Configuration
    config_file = "demo_config.ini"
    prepare_dummy_config(config_file)
    
    # Initialize your config (Assuming LTRConfig has a .from_ini method as seen in ltr_call.py CLI logic)
    cfg = LTRConfig.from_ini(config_file, features=feature_cols)
    cfg.feature.label = label_col
    cfg.feature.query_id = query_col

    # 5. Execute the Pipeline
    # This single call will trigger DataProcessor, BayesianTuner, LTRTrainer, Visualizer, MLflowMonitor, and LTRInference.
    print("\nStarting the `run_pipeline` orchestrator...")
    try:
        trainer = run_pipeline(
            config=cfg,
            train_df=train_df,
            test_df=test_df,
            run_tuning=True,        # Set to True to test Optuna / enhanced_byoptimz.py
            run_name="demo_parquet" # Triggers mlflow_proc.py
        )
        
        print("\n" + "=" * 60)
        print("  Demo Completed Successfully!")
        print(f"  Best Iteration : {trainer.best_iteration}")
        print(f"  Runtime        : {trainer.runtime_minutes} minutes")
        print("  Check the './output' folder for your HTML Report and ranked Parquet files.")
        print("  Run 'mlflow ui' in your terminal to view the tracking dashboard.")
        print("=" * 60)

    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()