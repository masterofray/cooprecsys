#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-28"

"""
load.py
Production-grade universal loader
Supported:
- CSV
- Parquet
- DuckDB (.db / .duckdb)
"""

import os
import gc
import pandas as pd
from pathlib import Path

from ..db import DuckDBManager, duckdb_connection, logger


# ==========================================================
# Validators
# ==========================================================
def _validate_exists(path: str) -> None:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")


def _validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        logger.error("Returned dataframe is None.")
        raise ValueError("Returned dataframe is None.")

    if df.empty:
        logger.warning("Loaded dataframe is empty.")


def _log_dataframe(df: pd.DataFrame) -> None:
    logger.info(f"Loaded rows={len(df):,}, cols={len(df.columns)}")
    logger.debug(f"Columns: {list(df.columns)}")


# ==========================================================
# CSV Loader
# ==========================================================
def _load_csv(path: str, **kwargs) -> pd.DataFrame:
    logger.info(f"Loading CSV file: {path}")

    defaults = {
        "low_memory": False
    }
    defaults.update(kwargs)

    return pd.read_csv(path, **defaults)


# ==========================================================
# Parquet Loader
# ==========================================================
def _load_parquet(path: str, **kwargs) -> pd.DataFrame:
    logger.info(f"Loading Parquet file: {path}")
    return pd.read_parquet(path, **kwargs)


# ==========================================================
# DuckDB Loader
# ==========================================================
def _load_duckdb(
    path: str,
    table_name: str = None,
    query: str = None,
    read_only: bool = True,
    threads: int = None,
    memory_limit: str = "4GB"
) -> pd.DataFrame:

    logger.info(f"Loading DuckDB database: {path}")

    with duckdb_connection(
        db_path=path,
        read_only=read_only,
        threads=threads,
        memory_limit=memory_limit
    ) as db:

        # --------------------------------------------------
        # Direct SQL Query
        # --------------------------------------------------
        if query:
            logger.info("Executing custom SQL query.")
            return db.query(query)

        # --------------------------------------------------
        # Detect tables
        # --------------------------------------------------
        tables = db.get_tables()

        if not tables:
            logger.error("No tables found in DuckDB database.")
            raise ValueError("No tables found in DuckDB database.")

        logger.info(f"Detected tables: {tables}")

        # --------------------------------------------------
        # Auto choose first table
        # --------------------------------------------------
        if table_name is None:
            table_name = tables[0]
            logger.warning(
                f"No table_name specified. "
                f"Using first detected table: {table_name}"
            )

        if table_name not in tables:
            logger.error(f"Table '{table_name}' not found.")
            raise ValueError(
                f"Table '{table_name}' not found. "
                f"Available tables: {tables}"
            )

        logger.info(f"Loading table: {table_name}")

        return db.query(f"SELECT * FROM {table_name}")


# ==========================================================
# Main Loader API
# ==========================================================
def load_data(
    data_path: str,
    table_name: str = None,
    query: str = None,
    sample_n: int = None,
    threads: int = None,
    memory_limit: str = "4GB",
    **kwargs
) -> pd.DataFrame:
    """
    Universal production data loader.

    Parameters
    ----------
    data_path : str
        Path to data file

    table_name : str, optional
        DuckDB table name

    query : str, optional
        Custom SQL query for DuckDB

    sample_n : int, optional
        Return first N rows only

    threads : int, optional
        DuckDB threads

    memory_limit : str
        DuckDB memory limit

    kwargs :
        forwarded to pandas loader

    Returns
    -------
    pd.DataFrame
    """

    logger.info("=" * 70)
    logger.info("LOAD DATA START")
    logger.info("=" * 70)

    _validate_exists(data_path)

    ext = Path(data_path).suffix.lower()

    logger.info(f"Detected extension: {ext}")

    try:

        # --------------------------------------------------
        # CSV
        # --------------------------------------------------
        if ext == ".csv":
            df = _load_csv(data_path, **kwargs)

        # --------------------------------------------------
        # Parquet
        # --------------------------------------------------
        elif ext == ".parquet":
            df = _load_parquet(data_path, **kwargs)

        # --------------------------------------------------
        # DuckDB
        # --------------------------------------------------
        elif ext in [".db", ".duckdb"]:
            df = _load_duckdb(
                path=data_path,
                table_name=table_name,
                query=query,
                threads=threads,
                memory_limit=memory_limit
            )

        else:
            logger.error(f"Unsupported extension: {ext}")
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported: .csv .parquet .db .duckdb"
            )

        # --------------------------------------------------
        # Sampling
        # --------------------------------------------------
        if sample_n is not None:
            logger.info(f"Applying sample_n={sample_n}")
            df = df.head(sample_n)

        _validate_dataframe(df)
        _log_dataframe(df)

        logger.info("LOAD DATA SUCCESS")
        logger.info("=" * 70)

        gc.collect()

        return df

    except Exception as e:
        logger.exception(f"LOAD DATA FAILED: {e}")
        raise

if __name__ == "__main__":

    try:
        # Example CSV
        df = load_data(
            data_path="train.csv",
            sample_n=5
        )

        logger.info(df.head())

    except Exception:
        logger.exception("Local loader test failed.")
