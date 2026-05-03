#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-03"

import os
import sys
import json
import pandas as pd
from pathlib         import Path
from copy            import deepcopy
from typing          import Dict, List, Optional
from .encdec         import LabelEncoderManager
from .date_processor import DateProcessor

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(LocDir)
from configs import _cfg, logger


def load_encoders(encoder_path: Optional[Path] = None) -> LabelEncoderManager:
    '''To make LabelEncoderManager with loaded encoders'''
    manager = LabelEncoderManager()
    manager.load(encoder_path)
    return manager


def load_feature_columns(feature_cols_path: Optional[Path] = None) -> List[str]:
    '''Load feature columns from JSON file.'''
    if feature_cols_path is None:
        opt = _cfg.get('PATHS', 'labelcoder')
        feature_cols_path = Path(opt) / "feature_columns.json"
    with open(feature_cols_path, 'r') as f:
        return json.load(f)


def load_group_sizes(groups_path: Optional[Path] = None) -> Dict[str, List[int]]:
    """Load group sizes from JSON file.
       groups_path: Path to group sizes JSON
       Returns: Dictionary with train_groups and test_groups"""
    if groups_path is None:
        opt = _cfg.get('PATHS', 'labelcoder')
        groups_path = Path(opt) / "group_sizes.json"
    with open(groups_path, 'r') as f:
        return json.load(f)


def prepare_inference_data(data             : pd.DataFrame,
                           encoder_manager  : LabelEncoderManager,
                           feature_processor: DateProcessor,
                           string_columns   : List[str],
                           drop_columns     : List[str] = None,
                          ) -> pd.DataFrame:
    """Prepare dataframe for inference.
       data             : Raw input dataframe
       encoder_manager  : Loaded label encoder manager
       feature_processor: Feature processor instance
       string_columns   : Columns to encode
       drop_columns     : Columns to drop
        The Returns is Processed dataframe ready for model inference!"""
    dataset = deepcopy(data)
    if drop_columns:
        dataset = dataset.drop(columns=[c for c in drop_columns if c in dataset.columns])
    dateproc = feature_processor(dataset)
    dateproc.fit_transform()
    datasetv1 = dateproc.data
    
    encproc = encoder_manager(data = datasetv1, Column = string_columns)
    encproc.fit_transform()
    datasetv2 = encproc.data
    return datasetv2

if __name__ == '__main__':
    pass