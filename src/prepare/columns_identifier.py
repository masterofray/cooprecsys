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
import numpy as np
from pathlib import Path
from typing import Dict, Optional

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger


def DetectReco_Identifier(
        DataColumns : np.ndarray
        ) -> Dict[str, Optional[str]]:
    """
    Identifies specific target columns using a two‑tier regex matching system.
    Phase 1 - Exact fullmatch (preferred identifiers).
    Phase 2 - Partial search with relaxed word boundaries (semantic fallback).
    Returns a dictionary mapping:
    - user_col
    - item_col 
    - quantity_col
    - total_col
    to their actual column names in the DataFrame (or None if not found).
    """
    logger.debug("Scanning dataframe columns for target variables.")
    result_map  = {"user_col"     : None,
                   "item_col"     : None,
                   "quantity_col" : None,
                   "total_col"    : None}
    Candidate   = {
    "user_col": {
        "exact"   : [r"(?i)^customer_?id$",
                     r"(?i)^user_?id$",
                     r"(?i)^cust_?id$",
                     r"(?i)^client_?id$",
                     r"(?i)^customer$",
                     r"(?i)^user$",
                     r"(?i)^member_?id$",
                     r"(?i)^buyer_?id$",
                     r"(?i)^customer$",
                     r"(?i)^cust$",
                     r"(?i)^user$",
                     r"(?i)^client$",
                     r"(?i)^member$",
                     r"(?i)^buyer$",],
        "partial" : [r"(?i)(?:^|_)(customer|user|account|subscriber|client|member|buyer)(?:$|_)",
                     r"(?i)(customer|user|client|cust)",],
                    },

    "item_col": {
        "exact"   : [r"(?i)^category_?id$",
                     r"(?i)^cat_?id$",
                     r"(?i)^item_?id$",
                     r"(?i)^product_?id$",
                     r"(?i)^prod_?id$",
                     r"(?i)^category$",
                     r"(?i)^sku$",
                     r"(?i)^product$",
                     r"(?i)^item$",],
        "partial" : [r"(?i)(?:^|_)(category|department|cat|item|product|prod|sku)(?:$|_)",
                     r"(?i)(category|segment|class|group|cat|item|product|prod)",],
                    },

    "quantity_col": {
        "exact"   : [r"(?i)^quantity$",
                     r"(?i)^qty$",
                     r"(?i)^count$",
                     r"(?i)^volume$",
                     r"(?i)^units$",
                     r"(?i)^unit_?count$",
                     r"(?i)^item_?count$",
                     r"(?i)^order_?qty$",
                     r"(?i)^purchase_?qty$",],
        "partial" : [r"(?i)(?:^|_)(quantity|qty|count|volume|pieces)(?:$|_)",
                     r"(?i)(quantity|qty)",],
                    },

    "total_col": {
        "exact"   : [r"(?i)^total_?price$",
                     r"(?i)^total$",
                     r"(?i)^price$",
                     r"(?i)^revenue$",
                     r"(?i)^sales$",
                     r"(?i)^amount$",
                     r"(?i)^grand_?total$",
                     r"(?i)^total_?amount$",
                     r"(?i)^transaction_?amount$",
                     r"(?i)^invoice_?total$",
                     r"(?i)^order_?total$",
                     r"(?i)^sales_?value$",
                     r"(?i)^gross_?sales$",
                     r"(?i)^revenue$",],
        "partial" : [r"(?i)(?:^|_)(total_?price|total|price|revenue|sales|amount|cost)(?:$|_)",
                     r"(?i)(total|price|revenue|amount)",],
                    }}

    for Target, rules in Candidate.items():
        logger.info("Evaluating rules for target = %s", Target)
        matched_col = None
        # Phase 1 - Exact fullmatch
        for pattern in rules["exact"]:
            for col in DataColumns:
                if re.fullmatch(pattern, str(col), re.IGNORECASE):
                    matched_col = col
                    break
            if matched_col:
                break
        # Phase 2 - Partial search (only if no exact match)
        if matched_col is None:
            logger.info(
            "No exact match for %s, trying partial patterns.",
            Target)
            for pattern in rules["partial"]:
                for col in DataColumns:
                    if re.search(pattern, str(col), re.IGNORECASE):
                        matched_col = col
                        break
                if matched_col:
                    break
        result_map[Target] = matched_col
    logger.info(f'This is the result: {result_map}.')
    return result_map


if __name__ == '__main__':
    import pandas as pd
    pathdf     = LocDir.parents[0] / 'data' / 'sampledata.parquet'
    assert pathdf.exists(), 'data is not exist.'
    data       = pd.read_parquet(str(pathdf))
    datacolumn = data.columns.to_numpy()
    check      = DetectReco_Identifier(datacolumn)
