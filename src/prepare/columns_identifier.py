#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-06"


import re
import sys
import numpy   as np
import pandas  as pd
from pathlib   import Path
from tqdm.auto import tqdm
from typing    import Dict, Optional


LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs   import logger, _cfg


def validate_cardinality(
        Dataprocess : pd.DataFrame, 
        col_name    : str, 
        target_key  : str, 
        MinUnique   : float,
    ) -> bool:
    """
    Validates if a specific column meets the minimum unique value ratio 
    required for entity identifiers (User/Item). Returns is boolean type.
    """
    try:
        if target_key in ["user_col", "item_col"]:
            col_lower = col_name.lower()
            if "date" in col_lower or "price" in col_lower:
                logger.info(
                    f"SEMANTIC MISMATCH: Column '{col_name}' matched regex for '{target_key}' "
                    f"but was REJECTED because it contains forbidden keywords ('date' or 'price').")
                return False
            
            total_rows = len(Dataprocess)
            if total_rows == 0:
                logger.info(f"Cardinality check skipped: DataFrame contains 0 rows for column '{col_name}'.")
                return False
            unique_count = Dataprocess[col_name].nunique()
            unique_ratio = unique_count / total_rows
            if unique_ratio <= MinUnique:
                logger.info(
                    f"CRITICAL MISMATCH: Column '{col_name}' matched regex for '{target_key}' "
                    f"but failed cardinality validation. Unique ratio: {unique_ratio:.2%} "
                    f"(Minimum required: {MinUnique:.2%})."
                )
                return False
        return True
        
    except Exception as e:
        logger.error(
            f"Error occurred during cardinality validation on column '{col_name}' "
            f"for target '{target_key}': {str(e)}", 
            exc_info = True)
        return False

def DetectReco_Identifier(
        Dataprocess : pd.DataFrame,
        MinUnique   : float = None,
    ) -> Dict[str, Optional[str]]:
    """
    Identifies specific target columns using a two‑tier regex matching system.
    Phase 1 - Exact fullmatch (preferred identifiers).
    Phase 2 - Partial search with relaxed word boundaries (semantic fallback).
    Returns a dictionary mapping:
    - user column
    - item column
    - quantity column
    - total column
    - discount column
    - sales_date column
    to their actual column names in the DataFrame (or None if not found).
    """
    logger.debug("Scanning dataframe columns for target variables with decoupled validation.")
    MinUnique = MinUnique or _cfg.getfloat('TRAINING', 'min_cardinal')
    MinUnique  = min(max(abs(MinUnique), _cfg.getfloat('TRAINING', 'lowercap_cardinal')), 0.4)
    result_map = {"user_col"       : None,
                  "item_col"       : None,
                  "quantity_col"   : None,
                  "total_col"      : None,
                  "discount_col"   : None,
                  "sales_date_col" : None}

    try:
        if Dataprocess is None or not isinstance(Dataprocess, pd.DataFrame):
            raise TypeError("Input 'Dataprocess' must be a valid pandas DataFrame instance.")
        DataColumns = Dataprocess.columns.values
        Candidate = {
            "user_col": {
                "exact": [
                    r"(?i)^customer_?id$", r"(?i)^user_?id$", r"(?i)^cust_?id$", 
                    r"(?i)^client_?id$", r"(?i)^customer$", r"(?i)^user$", 
                    r"(?i)^member_?id$", r"(?i)^buyer_?id$", r"(?i)^cust$", 
                    r"(?i)^client$", r"(?i)^member$", r"(?i)^buyer$"
                ],
                "partial": [
                    r"(?i)(?:^|_)(customer|user|account|subscriber|client|member|buyer|cust)(?:$|_)",
                    r"(?i)(customer|user|client|cust|member|buyer|account|subscriber)"
                ]
            },
            "item_col": {
                "exact": [
                    r"(?i)^category_?id$", r"(?i)^cat_?id$", r"(?i)^item_?id$", 
                    r"(?i)^product_?id$", r"(?i)^prod_?id$", r"(?i)^category$", 
                    r"(?i)^sku$", r"(?i)^product$", r"(?i)^item$"
                ],
                "partial": [
                    r"(?i)(?:^|_)(category|department|cat|item|product|prod|sku|segment|class|group)(?:$|_)",
                    r"(?i)(category|segment|class|group|cat|item|product|prod|sku|department)"
                ]
            },
            "quantity_col": {
                "exact": [
                    r"(?i)^quantity$", r"(?i)^qty$", r"(?i)^count$", r"(?i)^volume$", 
                    r"(?i)^units$", r"(?i)^unit_?count$", r"(?i)^item_?count$", 
                    r"(?i)^order_?qty$", r"(?i)^purchase_?qty$", r"(?i)^pieces$"
                ],
                "partial": [
                    r"(?i)(?:^|_)(quantity|qty|count|volume|pieces|units)(?:$|_)",
                    r"(?i)(quantity|qty|count|volume|pieces|units)"
                ]
            },
            "total_col": {
                "exact": [
                    r"(?i)^total_?price$", r"(?i)^total$", r"(?i)^price$", r"(?i)^revenue$", 
                    r"(?i)^sales$", r"(?i)^amount$", r"(?i)^grand_?total$", r"(?i)^total_?amount$", 
                    r"(?i)^transaction_?amount$", r"(?i)^invoice_?total$", r"(?i)^order_?total$", 
                    r"(?i)^sales_?value$", r"(?i)^gross_?sales$"
                ],
                "partial": [
                    r"(?i)(?:^|_)(total_?price|total|price|revenue|amount|cost|value)(?:$|_)",
                    r"(?i)(total|price|revenue|amount|cost|value)"
                ]
            },
            "discount_col": {
                "exact": [
                    r"(?i)^discount$", r"(?i)^discount_?amount$", r"(?i)^discount_?value$", 
                    r"(?i)^discount_?price$", r"(?i)^rebate$", r"(?i)^markdown$", 
                    r"(?i)^deduction$", r"(?i)^reduction$", r"(?i)^promo_?amount$", 
                    r"(?i)^coupon$", r"(?i)^savings$", r"(?i)^saving$", 
                    r"(?i)^allowance$", r"(?i)^concession$", r"(?i)^price_?off$", r"(?i)^percent_?off$"
                ],
                "partial" : [r"(?i)(?:^|_)(discount|rebate|markdown|deduction"
                     r"|reduction|promo|promotion|coupon|saving|savings"
                     r"|allowance|concession|off)(?:$|_)",
                     r"(?i)(discount|rebate|markdown|promo|coupon|saving|"
                     r"savings|off|deduction|reduction|allowance|concession)"]
                    },
            "sales_date_col": {
                "exact": [
                    r"(?i)^date$", r"(?i)^sales_?date$", r"(?i)^sale_?date$", 
                    r"(?i)^transaction_?date$", r"(?i)^order_?date$", r"(?i)^purchase_?date$", 
                    r"(?i)^invoice_?date$", r"(?i)^trans_?date$", r"(?i)^order_?dt$", 
                    r"(?i)^time$", r"(?i)^timestamp$", r"(?i)^datetime$", 
                    r"(?i)^event_?date$", r"(?i)^date_?of_?sale$", r"(?i)^transaction_?time$"
                ],
                "partial": [
                    r"(?i)(?:^|_)(date|time|timestamp|datetime|dt)(?:$|_)",
                    r"(?i)(date|time|timestamp|datetime)"
                ]
            }
        }

        # Loop utama pemetaan meta-kolom
        for Target, rules in tqdm(Candidate.items(),
                                  desc        = "Resolving Meta-Columns Architecture",
                                  colour      = _cfg.get('tqdm', 'colour'),
                                  ncols       = _cfg.getint('tqdm', 'ncols'),
                                  bar_format  = _cfg.get('tqdm', 'BarFormats'),
                                  unit        = 'process',
                                  mininterval = 0.1):
            matched_col = None

            # Phase 1 - Exact fullmatch
            for pattern in rules["exact"]:
                for col in DataColumns:
                    if re.fullmatch(pattern, str(col), re.IGNORECASE):
                        if validate_cardinality(Dataprocess, str(col), Target, MinUnique):
                            matched_col = col
                            break
                if matched_col:
                    break

            # Phase 2 - Partial search
            if matched_col is None:
                for pattern in rules["partial"]:
                    for col in DataColumns:
                        if re.search(pattern, str(col), re.IGNORECASE):
                            if validate_cardinality(Dataprocess, str(col), Target, MinUnique):
                                matched_col = col
                                break
                    if matched_col:
                        break
            result_map[Target] = matched_col
    except Exception as fatal_err:
        logger.critical(
            f"Fatal error encountered inside Identifier: {str(fatal_err)}",
            exc_info = True)
    finally:
        logger.debug(f"Target identifier identification complete. Mapping state: {result_map}")
        return result_map


if __name__ == '__main__':
    import pandas as pd
    pathdf     = LocDir.parents[0] / 'data' / 'sampledata.parquet'
    assert pathdf.exists(), 'data is not exist.'
    data       = pd.read_parquet(str(pathdf))
    check      = DetectReco_Identifier(
                 Dataprocess = data,
                 MinUnique   = 0.2)
    logger.warning('Test Result is here: ', check)
