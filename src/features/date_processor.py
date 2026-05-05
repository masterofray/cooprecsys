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
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import permutations
from typing import Optional, Dict, List, Tuple, Any

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(LocDir)
from configs import _cfg, logger


class DateProcessor(object):
    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        logger.info("Initializing DateProcessor")
        self._data            : pd.DataFrame = pd.DataFrame()
        self.date_keywords    : List[str] = [
            "date", "tanggal", "tgl",
            "year", "month", "week", "day",
            "created", "updated", "modified",
            "valid", "booked"]
        self.time_keywords    : List[str] = [
            "time", "hour", "minute",
            "second", "clock", "jam"]
        self.datetime_keywords: List[str] = [
            "datetime", "timestamp","pickup", "dropoff",
            "arrival", "departure","checkin", "checkout"]
        self.unix_keywords    : List[str] = [
            "unix", "epoch", "ts"]
        if data is not None:
            self.data = data
        self._result  = dict()
        logger.debug("DateProcessor initialized successfully")

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        logger.debug("Setting internal dataframe")
        if not isinstance(value, pd.DataFrame):
            logger.error("Input data is not pandas DataFrame")
            raise TypeError("data must be pandas DataFrame")
        if value.empty:
            raise ValueError("Your dataframe is empty")
        if value.shape[0] < 2:
            raise ValueError("Your dataframe is not sufficient")
        logger.debug("Data updated successfully.")
        self._data = deepcopy(value)
        logger.info("Data assigned with shape=%s", self._data.shape)

      
    # _________________________________________________________________
    # BASIC UTILS
    # _________________________________________________________________
    def _normalize(self, text: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    def _to_category(self, col: str) -> None:
        try:
            if col in self._data.columns:
                self._data[col] = self._data[col].astype("category")
                logger.debug("Converted %s to category", col)
            else:
                logger.info(f'There is no {col} column in data.')
        except Exception as exc:
            logger.warning("Failed convert %s to category: %s", col, exc)
            raise ValueError()

    def _drop_if_low_unique(self, col: str) -> None:
        try:
            if col in self._data.columns:
                nunique = self._data[col].nunique(dropna = True)
                if nunique < 2:
                    self._data.drop(columns=[col], inplace=True)
                    logger.info("Dropped low variance column = %s unique = %s", col, nunique)
        except Exception as exc:
            logger.warning("Failed checking unique for %s: %s", col, exc)


    # _________________________________________________________________
    # DETECTION HELPERS
    # _________________________________________________________________
    def _is_unix_timestamp(self, series: pd.Series) -> bool:
        try:
            numeric = pd.to_numeric(series, errors = "coerce").dropna()
            if numeric.empty:
                return False
            med = numeric.median()
            return ((1e9 <= med <= 2e10) or (1e12 <= med <= 2e13))
        except Exception as exc:
            logger.debug("Unix detection failed: %s", exc)
            return False

    def _is_date_like(self, series: pd.Series) -> bool:
        try:
            sample = series.dropna().head(30)
            if sample.empty:
                return False
            parsed = pd.to_datetime(sample, errors="coerce")
            score  = parsed.notna().mean()
            return score >= 0.70
        except Exception as exc:
            logger.debug("Date-like detection failed: %s", exc)
            return False


    # _________________________________________________________________
    # COLUMN DETECTION
    # _________________________________________________________________
    def detect_column_types(self) -> Dict[str, List[str]]:
        logger.info("Detecting column types")
        date_cols     : List[str] = list()
        time_cols     : List[str] = list()
        datetime_cols : List[str] = list()
        unix_cols     : List[str] = list()
        for item in tqdm(self.data.columns, 
                         desc   = 'Detect columns',
                         colour = _cfg.get('tqdm', 'colour'),
                         ncols  = _cfg.getinteger('tqdm', 'ncols'),
                         unit   = 'Column',
                         mininterval = 0.1):
            try:
                series = self.data[item]
                norm = self._normalize(item)
                if any(k in norm for k in self.unix_keywords):
                    unix_cols.append(item)
                elif any(k in norm for k in self.datetime_keywords):
                    datetime_cols.append(item)
                elif any(k in norm for k in self.time_keywords):
                    time_cols.append(item)
                elif any(k in norm for k in self.date_keywords):
                    date_cols.append(item)
                elif np.issubdtype(series.dtype, np.datetime64):
                    datetime_cols.append(item)
                else:
                    if self._is_unix_timestamp(series):
                        unix_cols.append(item)
                    elif self._is_date_like(series):
                        datetime_cols.append(item)
            except Exception as exc:
                logger.warning("Failed detect column = %s error = %s", item, exc)
                continue
        self._result = {'date'     : list(set(date_cols)),
                        'time'     : list(set(time_cols)),
                        'datetime' : list(set(datetime_cols)),
                        'unix'     : list(set(unix_cols))}
        logger.debug("Detection result: %s", result)


    # _________________________________________________________________
    # UNIX CONVERTER
    # _________________________________________________________________
    def convert_unix_columns(self) -> None:
        logger.info("Converting unix columns")
        UnixColumn = self._result["unix"]
        for item in tqdm(UnixColumn, 
                 desc   = 'Unix convert',
                 colour = _cfg.get('tqdm', 'colour'),
                 ncols  = _cfg.getinteger('tqdm', 'ncols'),
                 unit   = 'Column',
                 mininterval = 0.1):
            try:
                num = pd.to_numeric(self.data[item], errors = "coerce")
                med = num.dropna().median()
                if pd.isna(med):
                    logger.debug("Median NaN for %s skip", item)
                    continue
                if med < 1e11:
                    self._data[item] = pd.to_datetime(num, unit="s", errors="coerce")
                else:
                    self._data[item] = pd.to_datetime(num, unit="ms", errors="coerce")
                logger.info("Converted unix column = %s", item)
            except Exception as arch:
                logger.error("Failed convert unix %s: %s", item, arch)
                continue


    # _________________________________________________________________
    # DATE FEATURES
    # _________________________________________________________________
    def process_date_features(self) -> None:
        logger.info("Processing date features")
        datecolumn = self._result["date"]
        for item in tqdm(datecolumn,
                 desc   = 'Date Features Process',
                 colour = _cfg.get('tqdm', 'colour'),
                 ncols  = _cfg.getinteger('tqdm', 'ncols'),
                 unit   = 'Column',
                 mininterval = 0.1):
            try:
                dt = pd.to_datetime(self.data[item], errors="coerce")
                features = {f"{item}_year": dt.dt.year,
                            f"{item}_month": dt.dt.month,
                            f"{item}_week": dt.dt.isocalendar().week.astype(float),
                            f"{item}_day": dt.dt.day,
                            f"{item}_dayofweek": dt.dt.dayofweek,
                            f"{item}_quarter": dt.dt.quarter,
                            f"{item}_is_weekend":
                                (dt.dt.dayofweek >= 5).astype(int),
                            f"{item}_is_workday":
                                (dt.dt.dayofweek < 5).astype(int)}

                for new_col, values in features.items():
                    self._data[new_col] = values
                    self._drop_if_low_unique(new_col)
                    if new_col in self._data.columns:
                        self._to_category(new_col)
            except Exception as arch:
                logger.error("Date feature failed %s: %s", item, arch)
                continue

    # _________________________________________________________________
    # TIME FEATURES
    # _________________________________________________________________
    def process_time_features(self) -> None:
        logger.info("Processing time features")
        timecolumn = self._result["time"]
        for item in tqdm(timecolumn,
                 desc   = 'Time Features Process',
                 colour = _cfg.get('tqdm', 'colour'),
                 ncols  = _cfg.getinteger('tqdm', 'ncols'),
                 unit   = 'Column',
                 mininterval = 0.1):
            try:
                tm = pd.to_datetime(self.data[item], errors="coerce")
                hour = tm.dt.hour.fillna(0).astype(int)
                self._data[f"{item}_hour"] = hour
                self._drop_if_low_unique(f"{item}_hour")
                if f"{item}_hour" in self._data.columns:
                    self._to_category(f"{item}_hour")
                segment = np.select([(hour >= 0) & (hour < 5),
                                     (hour >= 5) & (hour < 10),
                                     (hour >= 10) & (hour < 14),
                                     (hour >= 14) & (hour < 18),
                                     (hour >= 18)],
                                    ["dawn", "morning", "afternoon", "evening", "night"],
                            default = "night")
                seg_col = f"{item}_segment"
                self._data[seg_col] = segment
                self._to_category(seg_col)
                bh_col = f"{item}_is_business_hours"
                self._data[bh_col] = ((hour >= 9) & (hour <= 17)).astype(int)
                self._drop_if_low_unique(bh_col)
                if bh_col in self._data.columns:
                    self._to_category(bh_col)
            except Exception as arch:
                logger.error("Time feature failed %s: %s", item, arch)
                continue


    # _________________________________________________________________
    # DATETIME FEATURES
    # _________________________________________________________________
    def process_datetime_features(self) -> None:
        logger.info("Processing datetime features")
        datetimecol = list(set(self._result["datetime"] + self._result["unix"]))
        for item in tqdm(datetimecol,
                         desc   = 'DateTime Features Process',
                         colour = _cfg.get('tqdm', 'colour'),
                         ncols  = _cfg.getinteger('tqdm', 'ncols'),
                         unit   = 'Column',
                         mininterval = 0.1):
            try:
                dt = pd.to_datetime(self.data[item], errors="coerce")
                features = {f"{item}_year": dt.dt.year,
                            f"{item}_month": dt.dt.month,
                            f"{item}_day": dt.dt.day,
                            f"{item}_hour": dt.dt.hour,
                            f"{item}_dayofweek":
                                dt.dt.dayofweek,
                            f"{item}_is_weekend":
                                (dt.dt.dayofweek >= 5).astype(int)}

                for new_col, values in features.items():
                    self._data[new_col] = values
                    self._drop_if_low_unique(new_col)
                    if new_col in self._data.columns:
                        self._to_category(new_col)
            except Exception as arch:
                logger.error("Datetime feature failed %s: %s", item, arch)
                continue


    # _________________________________________________________________
    # DURATION PAIRS
    # _________________________________________________________________
    def detect_duration_pairs(self) -> List[Tuple[str, str]]:
        logger.info("Detecting duration pairs")
        templates = [("start", "end"),
                     ("departure", "arrival"),
                     ("pickup", "dropoff"),
                     ("clockin", "clockout"),
                     ("shiftstart", "shiftend"),
                     ("checkin", "checkout"),
                     ("request", "response"),
                     ("started", "finished"),
                     ("sessionstart", "sessionend"),
                     ("booked", "completed"),
                     ("validfrom", "validuntil")]
        norm_map = {c: self._normalize(c) for c in self.data.columns}
        found = [(c1, c2) for c1, c2 in permutations(self.data.columns, 2)
                 if any(a in norm_map[c1] and b in norm_map[c2] for a, b in templates)]
        logger.debug("Found %s duration pairs", len(found))
        return found


    # _________________________________________________________________
    # DURATION FEATURES
    # _________________________________________________________________
    def process_duration_features(self) -> None:
        logger.debug("Processing duration features")
        pairs = self.detect_duration_pairs()
        for mystart, myend in tqdm(pairs,
                         desc   = 'Duration Features',
                         colour = _cfg.get('tqdm', 'colour'),
                         ncols  = _cfg.getinteger('tqdm', 'ncols'),
                         unit   = 'Column',
                         mininterval = 0.1):
            try:
                start = pd.to_datetime(self.data[mystart], errors="coerce")
                end   = pd.to_datetime(self.data[myend], errors = "coerce")
                diff  = end - start
                base  = f"{mystart}_to_{myend}"
                self._data[f"{base}_seconds"] = diff.dt.total_seconds()
                self._data[f"{base}_minutes"] = (diff.dt.total_seconds() / 60)
                self._data[f"{base}_hours"] = (diff.dt.total_seconds() / 3600)
                logger.debug("Created duration %s", base)
            except Exception as arch:
                logger.error("Duration failed %s -> %s : %s", mystart, myend, arch)
                continue


    # _________________________________________________________________
    # MAIN
    # _________________________________________________________________
    def fit_transform(self) -> pd.DataFrame:
        logger.info("Starting full feature process")
        try:
            self.convert_unix_columns()
            self.process_date_features()
            self.process_time_features()
            self.process_datetime_features()
            self.process_duration_features()
            logger.info("Feature engineering completed shape = %s", self.data.shape)
            return self.data
        except Exception as arch:
            logger.exception("Pipeline failed: %s", arch)
            raise ValueError()


    # _________________________________________________________________
    # CALL
    # _________________________________________________________________
    def __call__(self,
                 data: Optional[pd.DataFrame] = None
                ) -> pd.DataFrame:
        logger.debug("__call__ invoked")
        if data is not None:
            self.data = data
        return self.fit_transform()


if __name__ == '__main__':
    datapath = LocDir.parents[0] / 'data' / 'sampledata.parquet'
    datatest = pd.read_parquet(str(datapath))
    test = DateProcessor()
    test.data = datatest
    test()