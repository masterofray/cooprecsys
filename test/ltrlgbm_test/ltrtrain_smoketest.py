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

'''
ltrlgbm_example.py
__________________________________
A demonstration script to test the complete LightGBM LTR framework.
This script loads 'sampledata.parquet', prepares a group-aware train/test split,
configures the pipeline, and triggers all downstream LTR processes.
'''

import mlflow
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
LocDir = Path(__file__).resolve().parents[2] / 'src' / 'cooprecsys'
#sys.path.append(str(LocDir))
from src.cooprecsys.configs         import LTRConfig, logger
from src.cooprecsys.models.ltr_lgbm import lgbm_fit_transform

def maintest():
    logger.info("=" * 60)
    logger.info("  Initializing LTR Framework Demo")
    logger.info("=" * 60)

    # 1. Calling data
    #_________________________________________
    data_path = LocDir.parents[1] / 'data' / 'sampledata.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {str(data_path)}.")
    logger.info(f"Loading data from {data_path}.")
    data = pd.read_parquet(data_path)

    # 2. Define Schema & Features
    # The crucial thing is choosen lable_col 
    # must be category less than 30 unique unit.
    # And it must type int32.
    #_________________________________________
    query_col   = "CustomerID"
    #label_col   = "Quantity"
    label_col   = "CategoryID"
    TheFeatures = ['ProductName', "ProductPrice", "Discount", "TotalPrice", 
                   "Class", "Resistant", "IsAllergic", "CityName", "CountryName", 
                   "EmployeeID", "EmployeeGender", "Employee_City", "Quantity",
                   "VitalityDays", "EmployeeAge", "YearsWorking"]
    data['TotalPrice'] = pd.to_numeric(data['TotalPrice'], errors = 'coerce').astype('float64')
    logger.debug('This is information of data schema:')
    logger.debug(data.info())
    data[TheFeatures] = data[TheFeatures].fillna(0)
    data[label_col]   = data[label_col].fillna('unkown_product')
    data              = data.sort_values(query_col).reset_index(drop=True)

    # 3. Group-Aware Train/Test Split
    #_________________________________________
    logger.info("Splitting data into Train and Test sets (Grouped by CustomerID).")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state = 4)
    train_idx, test_idx = next(gss.split(data, groups = data[query_col]))
    train_df = data.iloc[train_idx].copy()
    test_df  = data.iloc[test_idx].copy()
    logger.info(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")
    logger.debug(train_df.head(4))
    logger.debug(test_df.head(4))
    
    # 4. Initialize your config
    #_________________________________________
    cfg = LTRConfig.from_ini(ini_path = str(LocDir / 'configs' / 'configuration.ini'), 
                             features = TheFeatures)
    cfg.feature.label = label_col
    cfg.feature.query_id = query_col

    # 5. Execute the Pipeline
    #_________________________________________
    logger.info("\nStarting the `run_pipeline` orchestrator.")
    try:
        trainer = lgbm_fit_transform(config     = cfg,
                                     train      = train_df,
                                     test       = test_df,
                                     run_tuning = True,
                                     run_name   = "demo_LTR_LGBM")
        logger.info("\n" + "=" * 60)
        logger.info("  Demo Completed Successfully!")
        logger.info(f"  Best Iteration : {trainer.best_iteration}")
        logger.info(f"  Runtime        : {trainer.runtime_minutes} minutes")
        logger.info("  Check the './output' folder for your HTML Report and ranked Parquet files.")
        logger.info("  Run 'mlflow ui' in your terminal to view the tracking dashboard.")
        logger.info("=" * 60)
    except Exception as Arch:
        logger.info(f"\nPipeline failed with error: {Arch}")
        raise RuntimeError()

if __name__ == "__main__":
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    maintest()
