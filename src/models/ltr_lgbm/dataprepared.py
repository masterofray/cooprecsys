#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-05"


import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import List, Optional, Tuple

LocDir = Path(__file__).resolve()
sys.path.append(LocDir.parents[2])
from configs import logger
from features import DateProcessor, LabelEncoderManager


def data_aftermath(train_df       : pd.DataFrame,
                   test_df        : pd.DataFrame,
                   string_columns : Optional[List[str]] = None,
                   drop_columns   : Optional[List[str]] = None,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoderManager]:
    if string_columns is None:
        string_columns = train_df.select_dtypes(include = 
                         ["object", "str", "category"]).columns.tolist()
    cleanstrCL = list(filter(
        lambda x: not re.search(r'date|hour|minute|time', x, 
        flags=re.IGNORECASE), string_columns))
    logger.debug(f'Here is the String Column : {cleanstrCL}.')

    # ===== TRAIN =====
    logger.info("Run preprocessing for Train Data.")
    proc_train = DateProcessor(train_df)
    train_feat = proc_train.fit()
    columnstp = deepcopy(proc_train._result)
    if drop_columns:
        train_feat  = train_feat.drop(
            columns = [c for c in drop_columns if c in train_feat.columns])
    enc = LabelEncoderManager(data = train_feat, Column = cleanstrCL)
    enc.fit_transform()
    train_final = enc.data

    # ===== TEST =====
    logger.info("Run preprocessing for Test Data from train column types.")
    proc_test = DateProcessor(test_df)
    test_feat = proc_test.transform(columnstp)
    if drop_columns:
        test_feat   = test_feat.drop(
            columns = [c for c in drop_columns if c in test_feat.columns])
    enc_test = LabelEncoderManager(data = test_feat, Column = cleanstrCL)
    enc_test.encoders = enc.encoders
    enc_test.encoder_classes = enc.encoder_classes
    enc_test.transform()
    test_final = enc_test.data
    
    logger.debug('Success to run the datapreparation.')
    return train_final, test_final, enc


def generate_sample_data(n    : int = 50, 
                         seed : int = 4,
                        ) -> pd.DataFrame:
    np.random.seed(seed)
    base_date = pd.Timestamp("2025-01-01")
    data = {
        "id": range(1, n + 1),
        "date_transaction": [
            (base_date + pd.Timedelta(days=np.random.randint(0, 100))).strftime("%Y-%m-%d")
            for _ in range(n)
        ],
        "pickup_time": [
            (base_date + pd.Timedelta(days=np.random.randint(0, 100),
                                      hours=np.random.randint(0, 24),
                                      minutes=np.random.randint(0, 60))
            ).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(n)
        ],
        "checkin": [
            (base_date + pd.Timedelta(days=np.random.randint(0, 80))).strftime("%Y-%m-%d")
            for _ in range(n)
        ],
        "checkout": [
            (base_date + pd.Timedelta(days=np.random.randint(80, 120))).strftime("%Y-%m-%d")
            for _ in range(n)
        ],
        "timestamp_unix": [
            int((base_date + pd.Timedelta(days=np.random.randint(0, 200))).timestamp() * 1000)
            for _ in range(n)
        ],
        "nama_produk": np.random.choice(["Tabungan", "Pinjaman", "Deposito", "Asuransi"], n),
        "kategori": np.random.choice(["A", "B", "C"], n),
        "nilai": np.random.uniform(10000, 500000, n).round(-3)}

    Data = pd.DataFrame(data)
    Data.loc[0, "date_transaction"] = np.nan
    Data.loc[2, "pickup_time"] = np.nan
    Data.loc[5, "checkout"] = np.nan
    return Data


if __name__ == "__main__":
    dryrun = True
    if not dryrun:
        dtrain = generate_sample_data(n=80, seed=4)
        dtest  = generate_sample_data(n=40, seed=12)
        logger.debug(f"Data mentah (train): {dtrain.shape}.")
        logger.debug(dtrain.head())
        logger.debug(f"Data mentah (test): {dtest.shape}.")
        logger.debug(f"Kolom: {dtrain.columns.tolist()}.")
        string_columns = ["nama_produk", "kategori"]
        train_ready, test_ready, encoder_manager = data_aftermath(
            train_df = dtrain, test_df = dtest, string_columns = string_columns)

        logger.debug("\n--- Hasil ---")
        logger.debug(f"Train ready shape: {train_ready.shape}.")
        logger.debug(f"Test ready shape: {test_ready.shape}.")
        logger.debug("\nContoh kolom baru (train):")
        logger.debug(train_ready.filter(like="date_transaction").head(3))
        
        pd.set_option('display.max_columns', None)
        logger.debug("\nContoh kolom baru (test):")
        logger.debug(test_ready.head(3))
        encoder_manager.save()
    else:
        from sklearn.model_selection import train_test_split
        
        pathdata = LocDir.parents[3] / 'data' / 'sampledata.parquet'
        data = pd.read_parquet(str(pathdata))
        tr, te = train_test_split(data, test_size = 0.2)
        trx, tex, _ = data_aftermath(tr, te)
        logger.warning(trx.sample(6))
