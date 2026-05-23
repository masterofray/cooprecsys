#!/usr/bin/env python3
from __future__ import annotations

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-30"

"""
lgbm_processor.py
_____________________________________________
DuckDB-backed, parallel-safe data preparation for LightGBM LTR datasets.
All heavy data manipulation (sorting by query ID, group-size computation,
feature/label extraction) is delegated to an in-process DuckDB connection.
For very large datasets the module forks worker processes via
`concurrent.futures.ProcessPoolExecutor` so that sorting and extraction
run in parallel per data-split.

Design notes
_____________________________________________
* State lives on ``self``; no public ``return`` from :meth:`prepare`.
* DuckDB is *always* used (even for small data) to guarantee consistent,
  SQL-based ordering semantics.
* Parallel execution is activated when ``len(df) > config.model.large_data_threshold``.
"""

import io
import re
import os
import sys
import duckdb
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

LocDir = Path(__file__).resolve().parents[1]
dates  = f'{datetime.now():%Y%m%d}'
sys.path.append(str(LocDir))
from configs import LTRConfig, _cfg, logger
from db import DuckDBManager, duckdb_connection


# ---------------------------------------------------------------------------
# Module-level worker - must be picklable (top-level function)
# ---------------------------------------------------------------------------
def _worker_prepare(df_serialised : bytes,
                    features      : List[str],
                    label         : str,
                    query_id      : str,
                    split_label   : str,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    data = pd.read_parquet(io.BytesIO(df_serialised))
    with duckdb_connection(':memory:') as con:
        con.register_dataframe("df_raw", data)
        df_sorted = con.query(f'SELECT * FROM df_raw ORDER BY "{query_id}"')
    X             = df_sorted[features].to_numpy(dtype=np.float32)
    y             = df_sorted[label].to_numpy(dtype=np.int32)
    group         = (
        df_sorted.groupby(query_id, sort=False)
        .size()
        .reindex(df_sorted[query_id].unique())
        .to_numpy(dtype=np.int32))
    return X, y, group, split_label


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class DataProcessor:
    """Prepare LightGBM-ready arrays from raw pandas DataFrames.
    The processor uses DuckDB for all sorting/aggregation and can
    optionally parallelise train/test preparation in separate worker
    processes.
    config: type class get from `~configs LTRConfig` master config.

    Attributes (populated after :meth:`prepare`)
    -------------------------------------------
    X_train, y_train, group_train : Training arrays.
    X_test, y_test, group_test    : Validation / test arrays.
    """
    def __init__(self,
                 config : LTRConfig,
                 data   : pd.DataFrame = pd.DataFrame([]),
                ) -> None:
        self._config = config
        self._datafm = data

        # Output state — populated by prepare()
        self.X_train:     np.ndarray | None = None
        self.y_train:     np.ndarray | None = None
        self.group_train: np.ndarray | None = None

        self.X_test:      np.ndarray | None = None
        self.y_test:      np.ndarray | None = None
        self.group_test:  np.ndarray | None = None

        logger.debug("DataProcessor initialised.")

    @property
    def config(self) -> LTRConfig:
        """Read-only access to the master config."""
        return self._config

    @property
    def _features(self) -> List[str]:
        return self._config.feature.features

    @property
    def _label(self) -> str:
        return self._config.feature.label

    @property
    def _query_id(self) -> str:
        return self._config.feature.query_id

    @property
    def _threshold(self) -> int:
        return self._config.model.large_data_threshold


    def _validate_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Assert all required columns are present in *df*."""
        if df.empty:
            df = deepcopy(self._datafm)
        required = set(self._features) | {self._label, self._query_id}
        missing  = required - set(df.columns)
        if missing:
            logger.warning(
                f"[{name}] Missing required columns: {sorted(missing)}.\n"
                f"We will use all other features except for {self._label} and {self._query_id}.")
            strCL = df.select_dtypes(exclude=["object"]).columns.tolist()
            checkColumn = lambda t, c_list: next((c for c in c_list if re.search(t, c, re.I)), t)
            self._config.feature.query_id = checkColumn(self._query_id, strCL)
            self._config.feature.label    = checkColumn(self._label, strCL)
            self._config.feature.features = list( set(strCL) - {self._label, self._query_id} )
        logger.debug("[%s] DataFrame validated. Shape: %s", name, df.shape)


    def _duckdb_sort_and_extract(self,
            df          : pd.DataFrame,
            split_label : str = "data",
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sort *df* by query_id via DuckDB and extract LightGBM arrays.
        df: Raw unsorted DataFrame.
        split_label: Logging tag.
        """
        logger.debug("[%s] DuckDB sort -- %d rows", split_label, len(df))
        with duckdb_connection(':memory:') as con:
            con.register_dataframe("raw_df", df)
            df_sorted = con.query(f'SELECT * FROM raw_df ORDER BY "{self._query_id}"')
        X     = df_sorted[self._features].to_numpy(dtype=np.float32)
        y     = df_sorted[self._label].to_numpy(dtype=np.int32)
        group = (df_sorted.groupby(self._query_id, sort=False)
                 .size()
                 .reindex(df_sorted[self._query_id].unique())
                 .to_numpy(dtype=np.int32))
        logger.debug("[%s] Extracted -- X: %s  y: %s  groups: %d",
            split_label, X.shape, y.shape, len(group))
        return X, y, group


    def _parallel_prepare(self,
            train_df : pd.DataFrame,
            test_df  : pd.DataFrame,
        ) -> None:
        """Prepare train and test splits in parallel worker processes.
        Uses ``ProcessPoolExecutor`` with two workers (one per split).
        Serialises DataFrames as Parquet bytes for inter-process transfer.
        Updates ``self`` in-place.
        """
        logger.info("Large dataset detected (> %d rows)."
        "\nActivating parallel processing for data preparation.",
        self._threshold)

        def _serialise(df: pd.DataFrame) -> bytes:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            return buf.getvalue()

        futures_map: dict = dict()
        results: dict     = dict()
        parker = min(_cfg.getint('duckdb', 'threads'), int(os.cpu_count()))
        with ProcessPoolExecutor(max_workers = parker) as executor:
            for split_label, df in [("train", train_df), ("test", test_df)]:
                future = executor.submit(
                            _worker_prepare,
                            _serialise(df),
                            self._features,
                            self._label,
                            self._query_id,
                            split_label)
                futures_map[future] = split_label
            pbar = tqdm(as_completed(futures_map),
                total       = len(futures_map),
                desc        = 'Parallel data prep',
                unit        = 'split',
                colour      = _cfg.get('tqdm', 'colour'),
                ncols       = _cfg.getint('tqdm', 'ncols'),
                bar_format  = _cfg.get('tqdm', 'BarFormats'))
            for future in pbar:
                X, y, group, split = future.result()
                results[split] = (X, y, group)
                pbar.set_postfix(done = split)

        self.X_train, self.y_train, self.group_train = results["train"]
        self.X_test,  self.y_test,  self.group_test  = results["test"]
        logger.info("Parallel data preparation complete.")


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self, 
                train_df : pd.DataFrame, 
                test_df  : pd.DataFrame,
        ) -> None:
        """Main entry-point. Populates train/test arrays on `self`."""
        logger.info("DataProcessor.prepare() — started.")
        logger.info("Train shape: %s | Test shape: %s",
            train_df.shape, test_df.shape)
        self._validate_dataframe(train_df, "train")
        self._validate_dataframe(test_df,  "test")
        use_parallel = (
            len(train_df) > self._threshold
            or len(test_df) > self._threshold)
        logger.debug(f'About the use_parallel : {use_parallel}.')

        if use_parallel:
            self._parallel_prepare(train_df, test_df)

        else:
            logger.debug("Standard (single-process) data preparation.")
            steps = [("train", train_df, "train"), ("test",  test_df,  "test")]
            for split_label, df, attr_prefix in tqdm(
                    steps,
                    desc        = "Preparing splits", 
                    unit        = "split",
                    colour      = _cfg.get('tqdm', 'colour'),
                    ncols       = _cfg.getint('tqdm', 'ncols'),
                    bar_format  = _cfg.get('tqdm', 'BarFormats')):
                X, y, group = self._duckdb_sort_and_extract(df, split_label)
                setattr(self, f"X_{attr_prefix}", X)
                setattr(self, f"y_{attr_prefix}", y)
                setattr(self, f"group_{attr_prefix}", group)

        logger.info(
            "DataProcessor.prepare() -- Complete. "
            "Train X: %s | Test X: %s",
            self.X_train.shape, self.X_test.shape)

if __name__ == '__main__':
    print(f'Pass on {dates}.')