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

import re
import json
import time
import numpy  as np
import pandas as pd
from tqdm.auto       import tqdm
from pathlib         import Path
from copy            import deepcopy
from typing          import Dict, List, Optional, Tuple
from .encdec         import LabelEncoderManager
from .date_processor import DateProcessor

from ..configs import _cfg, logger
from ..db      import duckdb_connection


def Normalize_LargeSeries(
        Data    : pd.DataFrame,
        Feature : str = 'score',
    ) -> pd.DataFrame:
    """
    Optimized in-place outlier clipping and min-max scaling (0-100).
    Keeps the original DataFrame schema intact (userID, itemID, score).
    """
    scores        = Data[Feature].to_numpy()
    low, high     = np.percentile(scores, [1, 99])
    denom         = (high - low) if high != low else 1.0
    Data[Feature] = (np.clip(scores, low, high) - low) / denom * 100
    return Data

def Filter_TopN(
        Data      : pd.DataFrame,
        user_col  : str = 'user_id',
        score_col : str = 'score',
        top_n     : int = None,
    ) -> pd.DataFrame:
    """
    Mengambil maksimal N prediksi dengan skor tertinggi untuk setiap user.
    Mengembalikan DataFrame independen yang bersih.
    """
    DataFilter = pd.DataFrame({})
    top_n      = top_n or _cfg.getint('FEATURES', 'top_predictions')
    try:
        start_time = time.time()
        if Data.empty:
            logger.warning("Your DataFrame is empty.")
            return Data.copy()
        require = {user_col, score_col}
        if not require.issubset(Data.columns):
            missing = require - set(Data.columns)
            raise KeyError(f"Some of mandatory columns is not found: {missing}")
        uniq = Data[user_col].nunique()
        logger.debug(f'''
        Begin to do filtering Top-{top_n}. 
        Basic Data: {len(Data):,} rows,
        {uniq:,} Unique Users.''')
        DataFilter = (
            Data.sort_values(by = [user_col, score_col], ascending = [True, False])
                .groupby(user_col, sort = False)
                .head(top_n)
                .copy()
                .reset_index(drop = True))
        elapsed = time.time() - start_time
        logger.debug(f'''
        Done with filtering in {elapsed:.4f} seconds.
        New Data: {len(DataFilter):,} rows.
        Average {len(DataFilter)/uniq:.1f} item/user.''')
    except KeyError as Arc01:
        logger.error(f"You data schema wrong: {str(Arc01)}",
                     exc_info = True)
        raise Arc01
    except Exception as Arc02:
        logger.error(f"Failed to do filtering: {str(Arc02)}",
                     exc_info = True)
        raise Arc02
    finally:
        return DataFilter

def Inference_DataSplit(data     : pd.DataFrame,
                        features : List[str],
                        label    : str,
                        query_id : str,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with duckdb_connection(':memory:') as con:
        con.register_dataframe("datamentah", data)
        RAW = con.query(f'SELECT * FROM datamentah ORDER BY "{query_id}"')
    X     = RAW[features].to_numpy(dtype = np.float32)
    y     = RAW[label].to_numpy(dtype = np.int32)
    group = (RAW.groupby(query_id, sort = False)
            .size()
            .reindex(RAW[query_id].unique())
            .to_numpy(dtype = np.int32))
    return X, y, group

def load_encoders(encoder_path: Optional[Path] = './') -> LabelEncoderManager:
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


def TrueString(data: pd.DataFrame,
               id_patterns: Optional[List[str]] = None,
               exclude_patterns: Optional[List[str]] = None,
               unique_threshold: float = 0.8,
               min_unique_ratio: float = 0.0,
               max_unique_ratio: float = 1.0,
              ) -> List[str]:
    """
    Flexible version with separate include/exclude patterns and unique ratio range.
    data             : Input dataframe.
    id_patterns      : Regex patterns to INCLUDE as potential ID columns (exclude from result).
    exclude_patterns : Regex patterns to EXCLUDE from ID detection (keep in result).
    unique_threshold : Upper bound for unique ratio (columns above this are excluded).
    min_unique_ratio : Lower bound for unique ratio (columns below this are excluded).
    max_unique_ratio : Alternative upper bound if different from unique_threshold.
    """
    if id_patterns is None:
        id_patterns = [
            r'(^|_)id($|_)', r'(^|_)key($|_)', r'(^|_)code($|_)',
            r'uuid', r'guid', r'username', r'email', r'phone', 
            r'telepon', r'mobile', r'contact', r'address', 
            r'alamat', r'npwp', r'nik', r'ktp', r'hash', 
            r'signature', r'checksum', r'url', r'image', 
            r'photo', r'timestamp', r'datetime', r'date', 
            r'description', r'deskripsi', r'keterangan',
            r'time', r'note', r'comment', r'remark']
    include_regex = re.compile('|'.join(id_patterns), re.IGNORECASE)
    
    # Compile exclude patterns if provided
    exclude_regex = None
    if exclude_patterns:
        exclude_regex = re.compile('|'.join(exclude_patterns), re.IGNORECASE)
    string_cols = [col for col in data.columns
                   if pd.api.types.is_string_dtype(data[col])]
    result = list()
    n = len(data)
    for item in tqdm(string_cols, 
                 desc        = 'Check TRUE String',
                 colour      = _cfg.get('tqdm', 'colour'),
                 ncols       = _cfg.getint('tqdm', 'ncols'),
                 bar_format  = _cfg.get('tqdm', 'BarFormats'),
                 unit        = 'Column',
                 mininterval = 0.5):
        col_lower = item.lower()
        if exclude_regex and exclude_regex.search(col_lower):
            unique_ratio = data[item].nunique(dropna=True) / n
            if min_unique_ratio <= unique_ratio <= max_unique_ratio and \
            unique_ratio < unique_threshold:
                result.append(item)
            continue
        
        if include_regex.search(col_lower):
            continue
        
        unique_ratio = data[item].nunique(dropna=True) / n
        if min_unique_ratio <= unique_ratio <= max_unique_ratio and \
        unique_ratio < unique_threshold:
            result.append(item)
    return result

if __name__ == '__main__':
    pass