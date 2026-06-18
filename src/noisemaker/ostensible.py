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
import sys
import numpy  as np
import pandas as pd
import scipy.sparse as sp
from   tqdm.auto import tqdm
from   pathlib   import Path
from   typing    import List, NamedTuple, Optional
from  .ersetz    import ExchangeResult

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg
from db      import duckdb_connection

DType = _cfg.get('model', 'dtype')

def extended_norm_exchange(
        data              : pd.DataFrame,
        user_col          : str                 = "CustomerID",
        item_col          : str                 = "CategoryID",
        rating_col        : Optional[str]       = None,
        dtype             : str                 = DType,
        user_feature_cols : Optional[List[str]] = None,
        item_feature_cols : Optional[List[str]] = None,
    ) -> ExchangeResult:
    """
    Convert a Pandas DataFrame of (user, item[, rating]) rows into a
    complete set of sparse tensors for collaborative-filtering models
    such as LightFM. Interaction matrix encoding follows the original
    implementation: DENSE_RANK is used inside DuckDB to produce 
    contiguous integer indices starting from 0.

    New outputs (vs original)
    ──────────────────────────
    user_features : CSR matrix (n_users × F_u)
        One row per user.  Built by aggregating *user_feature_cols*
        per CustomerID (MODE for categoricals, AVG for numericals),
        then one-hot encoding categoricals and min-max normalising
        numericals.  Default columns: ``CityName``, ``CountryName``.

    item_features : CSR matrix (n_items × F_i)
        One row per item (category).  Default columns: ``Class``,
        ``Resistant``, ``IsAllergic``, ``VitalityDays``, ``ProductPrice``.

    sample_weight : COO matrix (n_users × n_items)
        Same sparsity pattern as *interactions*.  Values are
        normalised ``log1p(Σ TotalPrice) × log1p(freq)`` per
        (user, item) pair — capturing both monetary commitment and
        interaction frequency as a confidence score.

    Parameters
    ───────────────────────────────────────────────────────────────────────
    data              : pd.DataFrame  – raw transactional records.
    user_col          : user identifier column (default 'CustomerID').
    item_col          : item identifier column (default 'CategoryID').
    rating_col        : explicit rating column; if None, interactions = 1.0.
    dtype             : numpy dtype for sparse matrix data arrays.
    user_feature_cols : columns to include in user_features matrix.
                        Defaults to ``_DEFAULT_USER_FEAT_COLS``.
    item_feature_cols : columns to include in item_features matrix.
                        Defaults to ``_DEFAULT_ITEM_FEAT_COLS``.

    Returns
    ───────────────────────────────────────────────────────────────────────
    :class:`ExchangeResult` NamedTuple — unpack as:
    ``interactions, user_ids, item_ids, user_features,
      item_features, sample_weight = norm_exchange(...)``
    """
    u_feat_cols = list(user_feature_cols or _DEFAULT_USER_FEAT_COLS)
    i_feat_cols = list(item_feature_cols or _DEFAULT_ITEM_FEAT_COLS)

    logger.info(
        "norm_exchange: shape=%s  user_col='%s'  item_col='%s'  "
        "rating_col=%s  user_feature_cols=%s  item_feature_cols=%s",
        data.shape, user_col, item_col, rating_col,
        u_feat_cols, i_feat_cols,
    )

    # ── validate ───────────────────────────────────────────────────────────────
    required = {user_col, item_col}
    if rating_col:
        required.add(rating_col)
    missing = required - set(data.columns)
    if missing:
        logger.error("norm_exchange: missing required columns: %s", missing)
        raise ValueError(f"DataFrame is missing columns: {missing}")
    for col_list, label in [(u_feat_cols, "user_feature_cols"),
                             (i_feat_cols, "item_feature_cols")]:
        extra = set(col_list) - set(data.columns)
        if extra:
            logger.warning(
            "%s contains columns not in DataFrame: %s "
            "– they will be dropped during aggregation", label, extra)
    rating_expr = (f'CAST("{rating_col}" AS DOUBLE)' if rating_col else "1.0")
    with tqdm(total      = 6,
              desc       = "norm_exchange",
              colour     = _cfg.get("tqdm", "colour"),
              ncols      = _cfg.getint("tqdm", "ncols"),
              bar_format = _cfg.get("tqdm", "BarFormats"),
              unit       = "process",
              mininterval= 0.1,
             ) as pbar:
        with duckdb_connection() as con:
            # ── Step 1 : register DataFrame, encode IDs ────────────────────
            pbar.set_postfix_str("registering DataFrame + encoding IDs")
            logger.debug("norm_exchange: registering DataFrame in DuckDB as RAW")
            con.register("RAW", data)
            encoded = con.query(f"""
                SELECT
                    DENSE_RANK() OVER (ORDER BY "{user_col}") - 1  AS user_idx,
                    DENSE_RANK() OVER (ORDER BY "{item_col}") - 1  AS item_idx,
                    {rating_expr}                                   AS rating,
                    "{user_col}"                                    AS user_id,
                    "{item_col}"                                    AS item_id
                FROM RAW
            """)

            n_rows = len(encoded)
            logger.debug(
                "norm_exchange [step 1]: encoded %d rows  "
                "user_idx range=[%d,%d]  item_idx range=[%d,%d]",
                n_rows,
                encoded["user_idx"].min(), encoded["user_idx"].max(),
                encoded["item_idx"].min(), encoded["item_idx"].max(),
            )
            if encoded["rating"].isna().any():
                n_null = encoded["rating"].isna().sum()
                logger.warning(
                    "norm_exchange: %d NaN values in rating column '%s' – "
                    "these interactions will appear as NaN in the sparse matrix.",
                    n_null, rating_col,
                )
            pbar.update(1)

            # ── Step 2 : extract ID maps ────────────────────────────────────
            pbar.set_postfix_str("extracting ID maps")
            user_map = con.query("""
                SELECT DISTINCT user_idx, user_id FROM encoded ORDER BY user_idx
            """)
            item_map = con.query("""
                SELECT DISTINCT item_idx, item_id FROM encoded ORDER BY item_idx
            """)
            n_users = int(user_map["user_idx"].max()) + 1
            n_items = int(item_map["item_idx"].max()) + 1
            logger.debug(
            "norm_exchange [step 2]: n_users=%d | n_items=%d | expected_nnz=%d | sparsity=%.4f%%",
            n_users, n_items, n_rows, 100.0 * n_rows / max(n_users * n_items, 1))
            pbar.update(1)

            # ── Step 3 : compute sample weights ────────────────────────────
            pbar.set_postfix_str("computing sample weights")
            logger.debug("norm_exchange [step 3]: computing confidence weights")
            if "TotalPrice" in data.columns:
                weight_sql = f"""
                    SELECT
                        DENSE_RANK() OVER (ORDER BY "{user_col}") - 1 AS user_idx,
                        DENSE_RANK() OVER (ORDER BY "{item_col}") - 1 AS item_idx,
                        LN(1.0 + SUM(CAST("TotalPrice" AS DOUBLE)))
                            * LN(1.0 + COUNT(*))                      AS raw_weight
                    FROM RAW
                    GROUP BY "{user_col}", "{item_col}"
                """
                logger.debug(
                "norm_exchange [step 3]: weight = log1p(Σ TotalPrice) x log1p(freq)")
            else:
                weight_sql = f"""
                    SELECT
                        DENSE_RANK() OVER (ORDER BY "{user_col}") - 1 AS user_idx,
                        DENSE_RANK() OVER (ORDER BY "{item_col}") - 1 AS item_idx,
                        LN(1.0 + COUNT(*))                            AS raw_weight
                    FROM RAW
                    GROUP BY "{user_col}", "{item_col}"
                """
                logger.warning(
                "norm_exchange [step 3]: 'TotalPrice' not found – "
                "sample_weight will use log1p(frequency) only")
            w_df         = con.execute(weight_sql).df()
            raw_w        = w_df["raw_weight"].values.astype(np.float64)
            w_min, w_max = raw_w.min(), raw_w.max()
            span         = w_max - w_min
            normed_w     = (np.full(len(raw_w), 0.5, dtype=np.float32)
                            if span < 1e-10 else ((raw_w - w_min) / span).astype(np.float32))
            logger.debug(
            "norm_exchange [step 3]: weight stats min=%.4f | max = %.4f | mean = %.4f",
            normed_w.min(), normed_w.max(), normed_w.mean())
            pbar.update(1)

            # ── Step 4 : build interactions + sample_weight COO matrices ───
            pbar.set_postfix_str("building COO matrices")
            row_arr       = encoded["user_idx"].values.astype(np.int32)
            col_arr       = encoded["item_idx"].values.astype(np.int32)
            data_arr      = encoded["rating"].values.astype(dtype)
            interactions  = sp.coo_matrix(
                            (data_arr, (row_arr, col_arr)),
                            shape=(n_users, n_items),
                            dtype=np.dtype(dtype))
            sample_weight = sp.coo_matrix(
                            (normed_w,
                            (w_df["user_idx"].values.astype(np.int32),
                             w_df["item_idx"].values.astype(np.int32))),
                            shape= (n_users, n_items),
                            dtype= np.float32,)
            user_ids      = user_map["user_id"].values
            item_ids      = item_map["item_id"].values
            logger.info(
                "norm_exchange [step 4]: interactions shape=%s  nnz=%d  "
                "sample_weight nnz=%d",
                interactions.shape, interactions.nnz, sample_weight.nnz)
            if interactions.nnz != sample_weight.nnz:
                logger.warning(
                "norm_exchange: interactions.nnz (%d) ≠ sample_weight.nnz (%d). "
                "This can happen when multiple rows share the same (user, item) "
                "pair (interactions stores duplicates as separate entries; "
                "sample_weight aggregates them).",
                interactions.nnz, sample_weight.nnz)
            pbar.update(1)

            # ── Step 5 : build user_features ───────────────────────────────
            pbar.set_postfix_str("building user_features matrix")
            logger.debug("norm_exchange [step 5]: aggregating user features  cols=%s",u_feat_cols)
            u_agg, u_cat, u_num = _aggregate_features(data, user_col, u_feat_cols, con)
            user_features       = _encode_to_sparse(u_agg, user_col, user_ids, u_cat, u_num, dtype = dtype)
            logger.info("norm_exchange [step 5]: user_features shape=%s  nnz=%d",
                         user_features.shape, user_features.nnz)
            del u_agg
            pbar.update(1)

            # ── Step 6 : build item_features ───────────────────────────────
            pbar.set_postfix_str("building item_features matrix")
            logger.debug("norm_exchange [step 6]: aggregating item features  cols=%s",
                          i_feat_cols)
            i_agg, i_cat, i_num = _aggregate_features(data, item_col, i_feat_cols, con)
            item_features = _encode_to_sparse(i_agg, item_col, item_ids, i_cat, 
                                              i_num, dtype = dtype)
            logger.info("norm_exchange [step 6]: item_features shape=%s  nnz=%d",
                         item_features.shape, item_features.nnz)
            del i_agg

    del encoded, row_arr, col_arr, data_arr, raw_w, normed_w, w_df
    gc.collect()
    logger.info(
        "norm_exchange: DONE\n"
        "  interactions  : %s  nnz=%d\n"
        "  user_features : %s  nnz=%d\n"
        "  item_features : %s  nnz=%d\n"
        "  sample_weight : %s  nnz=%d",
        interactions.shape,  interactions.nnz,
        user_features.shape, user_features.nnz,
        item_features.shape, item_features.nnz,
        sample_weight.shape, sample_weight.nnz)
    logger.debug("norm_exchange: user_ids[:5]=%s  item_ids[:5]=%s",
                 user_ids[:5], item_ids[:5])
    return ExchangeResult(
        interactions  = interactions,
        user_ids      = user_ids,
        item_ids      = item_ids,
        user_features = user_features,
        item_features = item_features,
        sample_weight = sample_weight,
    )

if __name__ == "__main__":
    print('This is Ostensible!')