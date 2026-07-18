#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-07"


import re
import sys
import pandas as pd
from pathlib  import Path
from datetime import datetime
from typing   import Any, Dict, List, Optional

ConfDir = Path(__file__).resolve()
sys.path.append(str(ConfDir.parents[1]))
from configs import logger, _cfg


def Gen_MiniStats(data: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
    """
    Generate mini statistics cards using two‑tier regex column matching.
    Phase 1 – Exact fullmatch (preferred identifiers).
    Phase 2 – Partial search with relaxed word boundaries (semantic fallback).
    All patterns are case‑insensitive.
    """
    logger.debug("Generating stat mini cards.")
    if data.empty:
        logger.warning("Prediction dataframe empty.")
        return list()

    stats: List[Dict[str, Any]] = list()
    try:
        candidate_rules = {
            "Users": {
                "exact": [
                    r"(?i)^customer_?id$",
                    r"(?i)^user_?id$",
                    r"(?i)^cust_?id$",
                    r"(?i)^customer$",
                    r"(?i)^user$",
                ],
                "partial": [
                    r"(?i)(?:^|_)(customer|user|client|member|buyer)(?:$|_)",
                    r"(?i)(customer|user|client)",
                ],
            },
            "Products": {
                "exact": [
                    r"(?i)^product_?id$",
                    r"(?i)^prod_?id$",
                    r"(?i)^product$",
                ],
                "partial": [
                    r"(?i)(?:^|_)(product|prod|item|sku)(?:$|_)",
                    r"(?i)(product|prod|item)",
                ],
            },
            "Categories": {
                "exact": [
                    r"(?i)^category_?id$",
                    r"(?i)^cat_?id$",
                    r"(?i)^category$",
                ],
                "partial": [
                    r"(?i)(?:^|_)(category|department|cat|class|segment|group)(?:$|_)",
                    r"(?i)(category|department|cat|class|segment)",
                ],
            }}
        for label, rules in candidate_rules.items():
            logger.debug("Evaluating stat label = %s", label)
            matched_col = None

            # Phase 1 – exact fullmatch
            for pattern in rules["exact"]:
                for col in data.columns:
                    if re.fullmatch(pattern, str(col)):
                        matched_col = col
                        break
                if matched_col:
                    break

            # Phase 2 – partial search (only if no exact match)
            if matched_col is None:
                logger.debug("No exact match for %s, trying partial patterns.", label)
                for pattern in rules["partial"]:
                    for col in data.columns:
                        if re.search(pattern, str(col)):
                            matched_col = col
                            break
                    if matched_col:
                        break

            if matched_col is None:
                logger.warning(
                    "No matching column for label=%s. Available columns: %s",
                    label, list(data.columns))
                continue
            logger.debug("Using column=%s for label=%s", matched_col, label)
            stats.append({"label"   : label,
                          "value"   : str(data[matched_col].nunique()),
                          "percent" : 100})
        stats.append({"label"   : "Rows",
                      "value"   : str(len(data)),
                      "percent" : 100})
        logger.info("Generated %d mini stat cards.", len(stats))
    except Exception as exc:
        logger.error("Mini stat generation failed.", exc_info=True)
        raise RuntimeError("Failed generating mini stats.") from exc
    finally:
        return stats


if __name__ == '__main__':
    logger.warning('Statrender here!')

