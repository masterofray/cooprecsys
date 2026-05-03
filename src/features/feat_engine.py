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

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cloudpickle as cpk
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
pd.set_option('display.max_columns', None)

try:
    LocDir = Path(__file__).resolve().parents[1]
    sys.path.append(str(LocDir))
    from db import DuckDBManager, duckdb_connection
    from config import logger
except Exception:
    import duckdb
    DuckDBManager = None
    duckdb_connection = None

@dataclass
class AutoFeatureEngineer(
        BaseEstimator, 
        TransformerMixin):
    entity          : str
    mode            : str = 'auto'
    leakage_safe    : bool = True
    n_jobs          : int = 1
    datetime_cols   : List[str] = field(default_factory=list)
    logger_name     : str = 'autofeat'
    backend         : str = 'duckdb'
    db_path         : str = ':memory:'
    temp_dir        : str = './autofeat_tmp'
    chunk_size      : int = 1000000
    exclude_cols    : List[str] = field(default_factory=list)
    custom_features : Dict[str, Callable] = field(default_factory=dict)
    agg_map_        : Dict[str, List[str]] = field(default_factory=dict)
    feature_columns_: List[str] = field(default_factory=list)
    fitted_         : bool = False

    def __post_init__(self):
        self.logger = logger
        self.logger.info('Initializing AutoFeat')
        try:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f'Temp directory ready: {self.temp_dir}')
        except Exception as arc:
            self.logger.exception(f'Failed creating temp dir: {arc}')
            raise ValueError()

    def _infer_types(self, df):
        num = df.select_dtypes(include=np.number).columns.tolist()
        cat = [c for c in df.columns if c not in num]
        num = [c for c in num if c != self.entity and c not in self.exclude_cols]
        cat = [c for c in cat if c != self.entity and c not in self.exclude_cols]
        return num, cat

    def _build_agg_map(self, df):
        num, cat = self._infer_types(df)
        agg = dict()
        for c in num:
            nunq = df[c].dropna().nunique()
            if nunq <= 2:
                agg[c] = ['sum','mean']
            else:
                agg[c] = ['mean','stddev','min','max','median']
        for c in cat:
            agg[c] = ['nunique']
        return agg

    def _duckdb_register(self, 
                         df : pd.DataFrame, 
                         name : str ='source_df',
                        ):
        self.logger.info(f'Registering dataframe rows={len(df):,} into {name}')
        if DuckDBManager is not None:
            self.db = DuckDBManager(self.db_path, read_only=False, threads=self.n_jobs or None)
            self.con = self.db.conn
            self.db.register_dataframe(name, df)
        else:
            self.con = duckdb.connect(self.db_path)
            self.con.register(name, df)
        return name

    def _duckdb_group_features(self, df):
        self.logger.info('Starting DuckDB aggregation engine')
        tbl = self._duckdb_register(df)
        exprs = list()

        for col, aggs in self.agg_map_.items():
            for agg in aggs:
                if agg == 'nunique':
                    sql_func = f"COUNT(DISTINCT {col})"
                elif agg == 'stddev':
                    sql_func = f"STDDEV({col})"
                else:
                    sql_func = f"{agg.upper()}({col})"
                exprs.append(f"{sql_func} AS {self.entity}__{col}__{agg}")
        sql = f"SELECT {self.entity}, {', '.join(exprs)} FROM {tbl} GROUP BY {self.entity}"
        logger.debug(f'Group Feature query : \n{sql}')
        return self.con.execute(sql).df()

    def _prepare_datetime(self, df : pd.DataFrame) -> pd.DataFrame:
        for c in self.datetime_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
                df[c + '_year'] = df[c].dt.year
                df[c + '_month'] = df[c].dt.month
                df[c + '_day'] = df[c].dt.day
                df[c + '_dow'] = df[c].dt.dayofweek
        return df

    def fit(self, 
            data:pd.DataFrame,
        ) -> pd.DataFrame:
        self.logger.info('Fit started')
        if self.entity not in data.columns:
            raise KeyError(f'{self.entity} not found')
        
        df = self._prepare_datetime(data.copy())
        self.agg_map_ = self._build_agg_map(df)
        self.fitted_ = True
        
        self.logger.info('AutoFeat fitted successfully')
        return self

    def transform(self, 
                  data : pd.DataFrame,
        ) -> pd.DataFrame:
        self.logger.info('Transform started')
        if not self.fitted_:
            raise RuntimeError('Call fit first')
        df = self._prepare_datetime(data.copy())
        
        if not self.agg_map_:
            raise RuntimeError('Call fit first')
            
        if self.backend == 'duckdb':
            # DuckDB sudah menghasilkan kolom datar (flat columns)
            feat = self._duckdb_group_features(df)
        else:
            # Pandas menghasilkan MultiIndex columns
            feat = df.groupby(self.entity, observed=True).agg(self.agg_map_)
            # Kita hanya ubah nama kolom jika menggunakan backend Pandas
            feat.columns = [f'{self.entity}__{a}__{b}' for a, b in feat.columns]
            feat = feat.reset_index()

        for name, fn in self.custom_features.items():
            vals = df.groupby(self.entity).apply(fn)
            feat = feat.merge(vals.rename(name).reset_index(), on=self.entity, how='left')
        self.feature_columns_ = [c for c in feat.columns if c != self.entity]
        self.logger.info(f'Generated {len(self.feature_columns_)} features')

        # Cleaning inf dan nan
        feat[self.feature_columns_] = feat[self.feature_columns_].replace([np.inf, -np.inf], np.nan)
        
        # Isi nan dengan median
        if not feat[self.feature_columns_].empty:
            feat[self.feature_columns_] = feat[self.feature_columns_].fillna(
                feat[self.feature_columns_].median(numeric_only=True)
            )
        return feat

    def to_parquet(self, df, path):
        df.to_parquet(path, index=False)
        self.logger.info(f'Saved parquet: {path}')

    def sql(self, query):
        self.logger.debug(f'Executing SQL: {query[:300]}')

    def sql_core(self, query):
        if hasattr(self, 'db') and self.db is not None:
            return self.db.query(query)
        return self.con.execute(query).df()

    def list_tables(self):
        if hasattr(self, 'db') and self.db is not None:
            return self.db.ListedTable()
        return self.con.execute("SELECT table_name FROM duckdb_tables()").df()

    def schema(self, table_name):
        if hasattr(self, 'db') and self.db is not None:
            return self.db.get_schema(table_name)
        return self.con.execute(f"DESCRIBE {table_name}").df()

    def profile(self, df):
        self.logger.info(f'Profiling dataframe rows={len(df):,} cols={len(df.columns)}')
        rep = pd.DataFrame({
            'column': df.columns,
            'dtype': df.dtypes.astype(str).values,
            'null_pct': (df.isnull().mean()*100).values,
            'nunique': [df[c].nunique() for c in df.columns]})
        return rep

    def get_feature_names_out(self):
        return self.feature_columns_

    def detect_drift(self, train_df, new_df):
        report = dict()
        for c in self.feature_columns_:
            if c in train_df.columns and c in new_df.columns:
                report[c] = abs(train_df[c].mean() - new_df[c].mean())
        return pd.Series(report).sort_values(ascending=False)

    def fit_transform(self, Data : pd.DataFrame) -> pd.DataFrame:
        return self.fit(Data).transform(Data)

    def register_custom_feature(self, name, func):
        self.custom_features[name] = func
        return self

    def save(self, path):
        with open(path,'wb') as f:
            cpk.dump(self,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return cpk.load(f)

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("AUTOFEAT - LOCAL TEST START")
    logger.info("=" * 70)
    try:
        import os
        # ==========================================================
        # 1. Create Dummy Dataset
        # ==========================================================
        np.random.seed(2)
        n_rows = 10_000
        df = pd.DataFrame({
            "user_id"     : np.random.randint(1, 1001, n_rows),
            "product_id"  : np.random.randint(100, 500, n_rows),
            "amount"      : np.random.uniform(5, 500, n_rows).round(2),
            "qty"         : np.random.randint(1, 8, n_rows),
            "discount"    : np.random.choice([0, 5, 10, 15], n_rows),
            "is_reordered": np.random.choice([0, 1], n_rows),
            "trx_date"    : pd.date_range(start="2025-01-01",
                                          periods=n_rows,
                                          freq="h",)
            })
        logger.info(f"Dummy dataset created: {df.shape}")
        logger.debug(df.head())

        # ==========================================================
        # 2. Initialize AutoFeat
        # ==========================================================
        fe = AutoFeatureEngineer(
            entity        = "user_id",
            backend       = "duckdb",
            db_path       = ":memory:",
            datetime_cols = ["trx_date"],
            n_jobs        = 4,)
        logger.info("AutoFeat object initialized.")

        # ==========================================================
        # 3. Register Custom Feature
        # ==========================================================
        fe.register_custom_feature(
            "repeat_product_ratio",
            lambda g: g["product_id"].duplicated().mean())
        logger.info("Custom feature registered.")

        # ==========================================================
        # 4. Fit Transform
        # ==========================================================
        feat = fe.fit_transform(df)
        logger.info(f"Feature engineering success. Shape: {feat.shape}")
        logger.info("Top 5 rows:")
        logger.info(feat.head())

        # ==========================================================
        # 5. Feature Names
        # ==========================================================
        logger.info(f"Total generated features: {len(fe.get_feature_names_out())}")

        # ==========================================================
        # 6. DuckDB SQL Test
        # ==========================================================
        sql_result = fe.sql_core("""
            SELECT COUNT(*) AS total_rows,
                   COUNT(DISTINCT user_id) AS total_users
            FROM source_df
        """)
        logger.info("SQL test result:")
        logger.info(sql_result)

        # ==========================================================
        # 7. Profile Test
        # ==========================================================
        profile = fe.profile(df)
        logger.info("Dataset profile:")
        logger.info(profile.head(10))

        # ==========================================================
        # 8. Save Output
        # ==========================================================
        fe.to_parquet(feat, "./autofeat_output.parquet")
        logger.info("Parquet export success.")
        logger.info("=" * 70)
        logger.info("AUTOFEAT TEST FINISHED SUCCESSFULLY")
        logger.info("=" * 70)
        os.remove('./autofeat_output.parquet')

    except Exception as e:
        logger.exception("AUTOFEAT TEST FAILED")
        raise ValueError()
