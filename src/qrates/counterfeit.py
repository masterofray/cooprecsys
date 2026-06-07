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


import gc
import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy  as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg


# ══════════════════════════════════════════════════════════════════════════════
# Public function 1 – pseudo-rating generator
# ══════════════════════════════════════════════════════════════════════════════

def make_pseudo_rating(
    data            : pd.DataFrame,
    user_col        : str                        = "CustomerID",
    item_col        : str                        = "CategoryID",
    quantity_col    : str                        = "Quantity",
    total_price_col : str                        = "TotalPrice",
    discount_col    : str                        = "Discount",
    date_col        : str                        = "SalesDate",
    rating_range    : Tuple[float, float]        = (1.0, 5.0),
    weights         : Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Derive implicit **pseudo-ratings** from transactional data when no
    explicit rating column is present.

    Five behavioural signals are computed per (user, item) pair inside
    DuckDB, min-max normalised via window functions, then blended with
    configurable weights and rescaled to *rating_range*.

    Signals
    ─────────────────────────────────────────────────────────────────────
    frequency  │ log1p(# transactions)        higher buy count → stronger pref
    quantity   │ log1p(Σ Quantity)             bulk purchasing signal
    spend      │ log1p(Σ TotalPrice)           monetary commitment (weight 0.30)
    recency    │ 1 − normalised days-since-last-buy  (1 = most recent)
    loyalty    │ 1 − mean_Discount              willingness to pay full price

    Parameters
    ──────────────────────────────────────────────────────────────────────
    data            : pd.DataFrame  – raw transactional records.
    user_col        : column for the user identifier.
    item_col        : column for the item / category identifier.
    quantity_col    : column holding purchase quantity per row.
    total_price_col : column holding total price (after discount) per row.
    discount_col    : column holding discount fraction in [0, 1].
    date_col        : column holding transaction date (castable to DATE).
    rating_range    : (min, max) output rating values; default (1.0, 5.0).
    weights         : dict with keys ``{'frequency','quantity','spend',
                      'recency','loyalty'}``; values do *not* need to sum
                      to 1 – they are auto-normalised.  Defaults to
                      ``_DEFAULT_RATING_WEIGHTS``.

    Returns
    ───────────────────────────────────────────────────────────────────────
    pd.DataFrame with exactly three columns:
        [user_col, item_col, 'pseudo_rating']
    One row per unique (user, item) pair.
    """
    logger.info(
        "make_pseudo_rating: data.shape=%s  user='%s'  item='%s'  "
        "rating_range=%s",
        data.shape, user_col, item_col, rating_range,
    )

    # ── 0. validate inputs ────────────────────────────────────────────────────
    required = {user_col, item_col, quantity_col, total_price_col,
                discount_col, date_col}
    missing  = required - set(data.columns)
    if missing:
        logger.error("make_pseudo_rating: required columns missing: %s", missing)
        raise ValueError(f"DataFrame is missing columns: {missing}")

    r_min, r_max = float(rating_range[0]), float(rating_range[1])
    if r_min >= r_max:
        raise ValueError(
            f"rating_range must be (min, max) with min < max; got {rating_range}"
        )
    logger.debug("make_pseudo_rating: rating_range validated [%.2f, %.2f]", r_min, r_max)

    # ── normalise / merge weights ─────────────────────────────────────────────
    w = {**_DEFAULT_RATING_WEIGHTS, **(weights or {})}
    w_total = sum(w.values())
    if abs(w_total - 1.0) > 1e-6:
        logger.warning(
            "make_pseudo_rating: provided weights sum to %.4f ≠ 1.0 – "
            "auto-normalising each weight", w_total,
        )
        w = {k: v / w_total for k, v in w.items()}
    logger.debug("make_pseudo_rating: effective signal weights = %s", w)

    n_pairs_raw = data.groupby([user_col, item_col]).ngroups
    logger.debug(
        "make_pseudo_rating: %d raw rows → %d unique (user, item) pairs",
        len(data), n_pairs_raw,
    )

    # ── 1. DuckDB aggregation + normalisation ─────────────────────────────────
    with duckdb_connection() as con:
        con.register("RAW", data)
        logger.debug("make_pseudo_rating: DataFrame registered in DuckDB as RAW")

        sql = f"""
        WITH
        -- Step A: aggregate raw transactions → one row per (user, item)
        raw_agg AS (
            SELECT
                "{user_col}"                                AS uid,
                "{item_col}"                               AS iid,
                COUNT(*)                                   AS freq,
                SUM(CAST("{quantity_col}"    AS DOUBLE))   AS total_qty,
                SUM(CAST("{total_price_col}" AS DOUBLE))   AS total_spend,
                1.0 - AVG(CAST("{discount_col}" AS DOUBLE)) AS loyalty,
                MAX(CAST("{date_col}" AS DATE))            AS last_date
            FROM RAW
            GROUP BY "{user_col}", "{item_col}"
        ),

        -- Step B: compute recency; days_ago = 0 for the most recent purchase
        with_recency AS (
            SELECT *,
                DATEDIFF('day', last_date,
                         MAX(last_date) OVER ())        AS days_ago,
                MAX(DATEDIFF('day', last_date,
                             MAX(last_date) OVER ())) OVER ()
                                                        AS max_days_ago
            FROM raw_agg
        ),

        -- Step C: log-transform heavy-tailed signals + compute recency score
        signals AS (
            SELECT
                uid,
                iid,
                LN(1.0 + CAST(freq AS DOUBLE))         AS log_freq,
                LN(1.0 + total_qty)                    AS log_qty,
                LN(1.0 + total_spend)                  AS log_spend,
                GREATEST(0.0, LEAST(1.0, loyalty))     AS loyalty,
                CASE
                    WHEN max_days_ago = 0 THEN 1.0
                    ELSE 1.0 - CAST(days_ago AS DOUBLE)
                             / CAST(max_days_ago AS DOUBLE)
                END                                    AS recency
            FROM with_recency
        ),

        -- Step D: min-max normalise log signals via window functions
        normalized AS (
            SELECT
                uid, iid, loyalty, recency,
                (log_freq  - MIN(log_freq)  OVER ())
                    / NULLIF(MAX(log_freq)  OVER ()
                           - MIN(log_freq)  OVER (), 0.0) AS n_freq,
                (log_qty   - MIN(log_qty)   OVER ())
                    / NULLIF(MAX(log_qty)   OVER ()
                           - MIN(log_qty)   OVER (), 0.0) AS n_qty,
                (log_spend - MIN(log_spend) OVER ())
                    / NULLIF(MAX(log_spend) OVER ()
                           - MIN(log_spend) OVER (), 0.0) AS n_spend
            FROM signals
        )

        -- Step E: weighted blend → raw_score ∈ [0, 1]
        SELECT
            uid                      AS "{user_col}",
            iid                      AS "{item_col}",
            CAST(
                {w['frequency']} * COALESCE(n_freq,   0.5) +
                {w['quantity']}  * COALESCE(n_qty,    0.5) +
                {w['spend']}     * COALESCE(n_spend,  0.5) +
                {w['recency']}   * COALESCE(recency,  0.5) +
                {w['loyalty']}   * COALESCE(loyalty,  0.5)
            AS DOUBLE)               AS raw_score
        FROM normalized
        ORDER BY "{user_col}", "{item_col}"
        """

        logger.debug("make_pseudo_rating: executing 5-stage SQL pipeline")
        result_df = con.execute(sql).df()
        con.unregister("RAW")

    logger.debug(
        "make_pseudo_rating: raw_score stats  "
        "min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
        result_df["raw_score"].min(), result_df["raw_score"].max(),
        result_df["raw_score"].mean(), result_df["raw_score"].std(),
    )
    if result_df["raw_score"].isna().any():
        n_null = result_df["raw_score"].isna().sum()
        logger.warning(
            "make_pseudo_rating: %d raw_score values are NaN – "
            "filling with midpoint 0.5.  Check signal weight configuration.",
            n_null,
        )
        result_df["raw_score"] = result_df["raw_score"].fillna(0.5)

    # ── 2. scale to [r_min, r_max] ─────────────────────────────────────────
    result_df["pseudo_rating"] = (
        r_min + result_df["raw_score"] * (r_max - r_min)
    ).round(4).astype(np.float32)
    result_df = result_df.drop(columns=["raw_score"])

    # ── 3. summary stats ───────────────────────────────────────────────────
    n_pairs = len(result_df)
    n_users = result_df[user_col].nunique()
    n_items = result_df[item_col].nunique()
    pr      = result_df["pseudo_rating"]
    logger.info(
        "make_pseudo_rating: done  pairs=%d  users=%d  items=%d  "
        "rating[min=%.3f  mean=%.3f  max=%.3f  std=%.3f]",
        n_pairs, n_users, n_items,
        pr.min(), pr.mean(), pr.max(), pr.std(),
    )

    cold_users = (result_df.groupby(user_col)["pseudo_rating"].count() == 1).sum()
    cold_items = (result_df.groupby(item_col)["pseudo_rating"].count() == 1).sum()
    if cold_users:
        logger.warning(
            "make_pseudo_rating: %d user(s) appear in only one (user,item) "
            "pair → cold-start risk", cold_users,
        )
    if cold_items:
        logger.warning(
            "make_pseudo_rating: %d item(s) appear in only one (user,item) "
            "pair → cold-start risk", cold_items,
        )

    return result_df
