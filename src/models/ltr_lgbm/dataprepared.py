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


import pandas as pd
from copy import deepcopy
from typing import List, Optional, Tuple
from .date_processor import DateProcessor
from .encdec import LabelEncoderManager


def data_aftermath(train_df       : pd.DataFrame,
                   test_df        : pd.DataFrame,
                   string_columns : Optional[List[str]] = None,
                   drop_columns   : Optional[List[str]] = None,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoderManager]:
    """
    Pipeline lengkap dengan DateProcessor + LabelEncoderManager.
    Enkoder di-fit HANYA pada train_df.
    Returnsnya adalah train_final, test_final, encoder_manager (bisa disimpan)
    """
    # ---------- deteksi kolom string ----------
    if string_columns is None:
        string_columns = train_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ===== TRAIN =====
    # 1. Date features
    proc_train = DateProcessor(train_df)
    train_feat = proc_train.fit_transform()

    # 2. Drop kolom (jika ada) & encode
    if drop_columns:
        train_feat = train_feat.drop(
            columns=[c for c in drop_columns if c in train_feat.columns])
    enc = LabelEncoderManager(data=train_feat, Column=string_columns)
    enc.fit_transform()
    train_final = enc.data

    # ===== TEST =====
    # 1. Date features
    proc_test = DateProcessor(test_df)
    test_feat = proc_test.fit_transform()

    # 2. Drop kolom yang sama
    if drop_columns:
        test_feat = test_feat.drop(
            columns=[c for c in drop_columns if c in test_feat.columns]
        )

    # 3. Transform saja, tanpa fit ulang
    enc_test = LabelEncoderManager(data=test_feat, Column=string_columns)
    enc_test.encoders = enc.encoders
    enc_test.encoder_classes = enc.encoder_classes
    enc_test.transform()
    test_final = enc_test.data

    return train_final, test_final, enc

if __name__ == '__main__':
    train_ready, test_ready = data_aftermath(train_df, test_df)