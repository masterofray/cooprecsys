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


"""
Collaborative-filtering data-preparation utilities.

Public API (new / extended)
────────────────────────────
make_pseudo_rating(data, ...)
    Derives a synthetic rating column from five implicit behavioural
    signals (frequency, quantity, spend, recency, loyalty).
    All signals are computed inside DuckDB, min-max normalised via
    window functions, then blended with configurable weights and
    mapped to a user-defined [min, max] range.

norm_exchange(data, ...)          ← extended signature
    Original interaction-matrix builder, now also returns:
        • user_features  : sp.csr_matrix  (n_users × F_u)
        • item_features  : sp.csr_matrix  (n_items × F_i)
        • sample_weight  : sp.coo_matrix  (n_users × n_items)
    Wrapped in ExchangeResult (NamedTuple) for ergonomic unpacking.

Private helpers
───────────────
_split_cat_num        – dtype-based column partitioning
_aggregate_features   – per-entity DuckDB aggregation (MODE / AVG)
_encode_to_sparse     – one-hot + min-max -> CSR sparse matrix
"""

import gc
import sys
import numpy  as np
import pandas as pd
import scipy.sparse as sp
from pathlib   import Path
from tqdm.auto import tqdm
from typing    import Dict, List, NamedTuple, Tuple

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg
from db      import duckdb_connection

DType = _cfg.get('model', 'dtype')


class ExchangeResult(NamedTuple):
    """
    All tensors produced by :func:`norm_exchange`.

    Attributes
    ----------
    interactions  : coo_matrix  shape (n_users, n_items)
                    Observed user-item interactions (ratings or 1.0).
    user_ids      : ndarray     int-index -> original user identifier.
    item_ids      : ndarray     int-index -> original item identifier.
    user_features : csr_matrix  shape (n_users, F_u)
                    One feature row per user; columns = one-hot + normalised
                    numeric features derived from *user_feature_cols*.
    item_features : csr_matrix  shape (n_items, F_i)
                    One feature row per item; same encoding as user_features.
    sample_weight : coo_matrix  shape (n_users, n_items)
                    Interaction-level confidence weights; same sparsity pattern
                    as *interactions*.  Values = normalised
                    log1p(Σ spend) × log1p(frequency).
    """
    interactions  : sp.coo_matrix
    user_ids      : np.ndarray
    item_ids      : np.ndarray
    user_features : sp.csr_matrix
    item_features : sp.csr_matrix
    sample_weight : sp.coo_matrix


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _split_cat_num(
    df     : pd.DataFrame,
    cols   : List[str],
    id_col : str,
) -> Tuple[List[str], List[str]]:
    """
    Partition *cols* into (categorical_cols, numerical_cols) by pandas dtype.

    Parameters
    ----------
    df     : source DataFrame used only for dtype inspection.
    cols   : candidate columns to classify.
    id_col : entity-ID column; always excluded from output lists.

    Returns
    -------
    cat_cols, num_cols
    """
    cat, num = list(), list()
    for c in cols:
        if c == id_col:
            logger.debug("skipping id column '%s'", c)
            continue
        if c not in df.columns:
            logger.warning(
                "column '%s' not found in DataFrame – skipping", c
            )
            continue
        if df[c].dtype == object or str(df[c].dtype) == "category":
            cat.append(c)
            logger.debug("'%s' -> categorical", c)
        else:
            num.append(c)
            logger.debug("'%s' -> numerical (dtype=%s)", c, df[c].dtype)

    logger.debug(
        "[id=%s]: cat=%s  num=%s", id_col, cat, num
    )
    return cat, num


# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_features(
    data         : pd.DataFrame,
    id_col       : str,
    feature_cols : List[str],
    con,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Collapse multiple transaction rows into **one feature row per entity**
    using DuckDB (assumes ``RAW`` is already registered on *con*).

    Aggregation strategy
    ─────────────────────
    Categorical  ->  ``MODE()``  — most frequent non-null value.
    Numerical    ->  ``AVG()``   — mean across all transactions.

    Parameters
    ----------
    data         : original DataFrame (used only for dtype inspection).
    id_col       : grouping key (user_col or item_col).
    feature_cols : candidate columns to include.
    con          : open DuckDB connection with ``RAW`` registered.

    Returns
    -------
    agg_df   : aggregated DataFrame, one row per unique *id_col* value.
    cat_cols : categorical columns actually included.
    num_cols : numerical  columns actually included.
    """
    cat_cols, num_cols = _split_cat_num(data, feature_cols, id_col)

    if not cat_cols and not num_cols:
        logger.warning(
            "_aggregate_features [%s]: no valid feature columns found; "
            "returning bare id-only DataFrame", id_col,
        )
        return pd.DataFrame({id_col: data[id_col].unique()}), list(), list()

    cat_exprs = ",\n            ".join(
        [f'MODE("{c}") AS "{c}"' for c in cat_cols]
    )
    num_exprs = ",\n            ".join(
        [f'AVG(CAST("{c}" AS DOUBLE)) AS "{c}"' for c in num_cols]
    )
    all_exprs = ",\n            ".join(filter(None, [cat_exprs, num_exprs]))

    sql = f"""
        SELECT
            "{id_col}",
            {all_exprs}
        FROM RAW
        GROUP BY "{id_col}"
        ORDER BY "{id_col}"
    """
    logger.debug(
        "_aggregate_features [%s]: executing SQL  cat=%d  num=%d",
        id_col, len(cat_cols), len(num_cols),
    )
    agg_df = con.execute(sql).df()

    n_rows    = len(agg_df)
    nan_frac  = agg_df.isnull().values.mean()
    logger.debug(
        "_aggregate_features [%s]: result rows=%d  NaN=%.2f%%",
        id_col, n_rows, 100 * nan_frac,
    )
    if nan_frac > 0.05:
        logger.warning(
            "_aggregate_features [%s]: %.1f%% NaN values in aggregated "
            "feature frame – verify source data quality (sparse records, "
            "wrong column mappings, or MODE on all-null groups)",
            id_col, 100 * nan_frac,
        )
    if n_rows == 0:
        logger.error(
            "_aggregate_features [%s]: aggregation returned 0 rows! "
            "Check that '%s' column exists in RAW and contains data.",
            id_col, id_col,
        )
        raise RuntimeError(f"_aggregate_features: empty result for id_col='{id_col}'")

    return agg_df, cat_cols, num_cols


# ─────────────────────────────────────────────────────────────────────────────

def _encode_to_sparse(
    agg_df   : pd.DataFrame,
    id_col   : str,
    id_map   : np.ndarray,
    cat_cols : List[str],
    num_cols : List[str],
    dtype    : str = DType,
) -> sp.csr_matrix:
    """
    Convert an aggregated feature DataFrame into a **CSR sparse matrix**.

    Row order is forced to match *id_map* — the DENSE_RANK ordering
    produced by :func:`norm_exchange` so that matrix rows align perfectly
    with interaction-matrix rows/columns.

    Encoding rules
    ──────────────
    Categorical  ->  ``pd.get_dummies`` one-hot expansion.
                    NaN -> imputed with column mode (or ``'__unknown__'``).
    Numerical    ->  min-max normalisation to [0, 1].
                    NaN -> imputed with column median.
                    Constant columns -> set to 0.5 with a warning.

    Parameters
    ----------
    agg_df   : aggregated DataFrame (one row per entity).
    id_col   : entity-ID column name in *agg_df*.
    id_map   : ordered array of entity IDs (from norm_exchange).
    cat_cols : categorical columns to one-hot encode.
    num_cols : numerical  columns to normalise.
    dtype    : numpy dtype string for the output sparse matrix.

    Returns
    -------
    sp.csr_matrix of shape ``(len(id_map), Σ n_features)``
    """
    logger.debug(
        "_encode_to_sparse [%s]: aligning %d entities to id_map (len=%d)",
        id_col, len(agg_df), len(id_map),
    )

    # ── align row order to norm_exchange's DENSE_RANK ordering ───────────────
    id_frame = pd.DataFrame({id_col: id_map})
    agg_df   = id_frame.merge(agg_df, on=id_col, how="left")

    orphan = int(agg_df.isnull().any(axis=1).sum())
    if orphan:
        logger.warning(
            "_encode_to_sparse [%s]: %d / %d entities had no matching row "
            "in aggregated features after merge – they will be imputed from "
            "column statistics.  Possible cause: id_map contains IDs not "
            "present in the feature aggregation (data leakage or subset split).",
            id_col, orphan, len(id_map),
        )
    else:
        logger.debug("_encode_to_sparse [%s]: merge OK – no orphan entities", id_col)

    blocks: List[sp.csr_matrix] = list()
    feature_names: List[str]    = list()

    # ── categorical -> one-hot ─────────────────────────────────────────────────
    for col in cat_cols:
        n_na     = int(agg_df[col].isna().sum())
        n_unique = agg_df[col].nunique(dropna=True)
        logger.debug(
            "_encode_to_sparse: '%s'  NaN=%d  unique(non-null)=%d",
            col, n_na, n_unique,
        )
        if n_na:
            fill_val = (
                agg_df[col].mode().iat[0]
                if agg_df[col].notna().any() else "__unknown__"
            )
            logger.warning(
                "_encode_to_sparse: '%s' – %d NaN -> imputed with mode='%s'",
                col, n_na, fill_val,
            )
            agg_df[col] = agg_df[col].fillna(fill_val)

        agg_df[col] = agg_df[col].astype(str)
        dummies     = pd.get_dummies(agg_df[col], prefix=col, dtype=np.float32)
        logger.debug(
            "_encode_to_sparse: '%s' -> %d one-hot dims: %s",
            col, dummies.shape[1], list(dummies.columns),
        )
        feature_names.extend(dummies.columns.tolist())
        blocks.append(sp.csr_matrix(dummies.values, dtype=dtype))

    # ── numerical -> min-max normalise ────────────────────────────────────────
    for col in num_cols:
        n_na = int(agg_df[col].isna().sum())
        logger.debug(
            "_encode_to_sparse: '%s'  dtype=%s  NaN=%d",
            col, agg_df[col].dtype, n_na,
        )
        if n_na:
            med = float(np.nanmedian(agg_df[col].values.astype(float)))
            logger.warning(
                "_encode_to_sparse: '%s' – %d NaN -> imputed with median=%.4f",
                col, n_na, med,
            )
            agg_df[col] = agg_df[col].fillna(med)

        vals = agg_df[col].values.astype(np.float64)
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin

        if span < 1e-10:
            logger.warning(
                "_encode_to_sparse: '%s' is constant (value=%.4f) – "
                "zero-variance feature; setting all values to 0.5.  "
                "Consider dropping this column.",
                col, vmin,
            )
            normed = np.full(len(vals), 0.5, dtype=np.float32)
        else:
            normed = ((vals - vmin) / span).astype(np.float32)
            logger.debug(
                "_encode_to_sparse: '%s' normalised to [%.4f, %.4f]  "
                "mean=%.4f  std=%.4f",
                col, normed.min(), normed.max(), normed.mean(), normed.std(),
            )

        feature_names.append(col)
        blocks.append(sp.csr_matrix(normed.reshape(-1, 1), dtype=dtype))

    # ── assemble ──────────────────────────────────────────────────────────────
    if not blocks:
        n = len(id_map)
        logger.warning(
            "_encode_to_sparse [%s]: no feature blocks produced; "
            "returning %d×1 constant matrix.  Downstream model may "
            "ignore user/item features entirely.", id_col, n,
        )
        return sp.csr_matrix(np.ones((n, 1), dtype=dtype))

    mat = sp.hstack(blocks, format="csr", dtype=dtype)
    density = 100.0 * mat.nnz / max(mat.shape[0] * mat.shape[1], 1)
    logger.debug(
        "_encode_to_sparse [%s]: output shape=%s  nnz=%d  density=%.2f%%  "
        "features=%s",
        id_col, mat.shape, mat.nnz, density, feature_names,
    )
    if density < 1.0:
        logger.warning(
            "_encode_to_sparse [%s]: extremely sparse feature matrix "
            "(density=%.2f%%) – verify column selection", id_col, density,
        )
    return mat

if __name__ == "__main__":
    print('This is Ersetz!')