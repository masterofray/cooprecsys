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
import numpy     as np
import pandas    as pd
from   pathlib   import Path
from   enum      import Enum
from   tqdm.auto import tqdm
from   copy      import deepcopy
from   typing    import Dict, Optional, Tuple, List
from   sklearn.linear_model  import Lasso, Ridge
from   sklearn.preprocessing import MinMaxScaler
from   sklearn.decomposition import PCA

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger, _cfg, _cfglist
from db import duckdb_connection
from prepare import DetectReco_Identifier



class ScenarioType(Enum):
    """Scenario based on available columns."""
    FULL               = "full"                 # All columns present
    NO_PRICE           = "no_price"             # Missing total_price_col
    NO_DISCOUNT        = "no_discount"          # Missing discount_col
    NO_DATE            = "no_date"              # Missing date_col
    NO_PRICE_DISCOUNT  = "no_price_discount"    # Missing price + discount
    NO_PRICE_DATE      = "no_price_date"        # Missing price + date
    NO_DISCOUNT_DATE   = "no_discount_date"     # Missing discount + date
    MINIMAL            = "minimal"              # Only user, item, quantity


class ScoringMethod(Enum):
    """Scoring algorithm to use."""
    WEIGHTED = "weighted"    # Original weighted-average blend
    LASSO    = "lasso"       # L1-regularised regression (sparse weights)
    RIDGE    = "ridge"       # L2-regularised regression
    PCA      = "pca"         # First principal component
    EQUAL    = "equal"       # Simple unweighted mean



class CounterFeit_Core:
    """
    Derive implicit *pseudo-ratings* from transactional data when no explicit
    rating column is present.

    The engine automatically detects which columns are available, adapts its
    SQL aggregation pipeline, and offers several scoring algorithms.

    Scenarios
    ─────────────────────────────────────────────────────────────────────────
    FULL               – all 5 signals (freq, qty, spend, recency, loyalty)
    NO_PRICE           – 4 signals (no spend)
    NO_DISCOUNT        – 4 signals (no loyalty)
    NO_DATE            – 4 signals (no recency)
    NO_PRICE_DISCOUNT  – 3 signals (freq, qty, recency)
    NO_PRICE_DATE      – 3 signals (freq, qty, loyalty)
    NO_DISCOUNT_DATE   – 3 signals (freq, qty, spend)
    MINIMAL            – 2 signals (freq, qty only)

    Scoring Methods
    ─────────────────────────────────────────────────────────────────────────
    WEIGHTED  – configurable weighted average (original approach)
    LASSO     – L1 regression learns sparse weights from data
    RIDGE     – L2 regression learns dense weights
    PCA       – first principal component as composite score
    EQUAL     – simple arithmetic mean of all normalised signals
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        'frequency': _cfg.getfloat('RATING', 'frequency'),
        'quantity' : _cfg.getfloat('RATING', 'quantity'),
        'spend'    : _cfg.getfloat('RATING', 'spend'),
        'recency'  : _cfg.getfloat('RATING', 'recency'),
        'loyalty'  : _cfg.getfloat('RATING', 'loyalty'),
    }

    def __init__(
            self,
            data            : pd.DataFrame,
            user_col        : str                        = "CustomerID",
            item_col        : str                        = "CategoryID",
            quantity_col    : str                        = "Quantity",
            total_price_col : Optional[str]              = "TotalPrice",
            discount_col    : Optional[str]              = "Discount",
            date_col        : Optional[str]              = "SalesDate",
            rating_range    : Tuple[float, float]        = None,
            weights         : Optional[Dict[str, float]] = None,
            method          : str                        = None,
            lasso_alpha     : float                      = 0.01,
            ridge_alpha     : float                      = 1.0,
            pca_components  : int                        = 1,
        ):
        if rating_range is None:
            rating_range     = _cfglist(_cfg, 'RATING', 'range')
        if method is None:
            method           = _cfg.get('RATING', 'methodscore')
        logger.debug("Initialized data.shape = %s | method = %s",
                      data.shape, method)
        logger.debug("columns -> user = '%s' item = '%s' \n"
                     "qty = '%s' price = '%s' disc='%s' date='%s'",
                      user_col, item_col, quantity_col,
                      total_price_col, discount_col, date_col)
        # store raw references
        self.data            = deepcopy(data)
        self.user_col        = user_col
        self.item_col        = item_col
        self.quantity_col    = quantity_col
        self.total_price_col = total_price_col
        self.discount_col    = discount_col
        self.date_col        = date_col
        self.rating_range    = rating_range
        self.lasso_alpha     = lasso_alpha
        self.ridge_alpha     = ridge_alpha
        self.pca_components  = pca_components

        # validate & coerce
        self.method = ScoringMethod(method)
        self._validate_inputs()

        # detect scenario
        self.scenario = self._detect_scenario()
        logger.info("detected scenario: %s", self.scenario.value)

        # resolve weights
        self.available_signals = self._get_available_signals()
        self.weights = self._setup_weights(weights)
        logger.debug("effective weights = %s", self.weights)

        logger.info(
            "engine ready  signals=%s  scenario=%s",
            self.available_signals, self.scenario.value,
        )

    # ── validation ───────────────────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        """Check base columns exist; coerce missing optional cols to None."""
        logger.debug("validating input columns")

        base_required = {self.user_col, self.item_col, self.quantity_col}
        missing_base  = base_required - set(self.data.columns)
        if missing_base:
            logger.error("required base columns missing: %s", missing_base)
            raise ValueError(
                f"Required base columns missing from dataframe: {missing_base}"
            )

        # optional columns – set to None when absent
        _optional = {
            'total_price_col': self.total_price_col,
            'discount_col':    self.discount_col,
            'date_col':        self.date_col,
        }
        for attr, col_name in _optional.items():
            if col_name is not None and col_name not in self.data.columns:
                logger.warning(
                    "column '%s' (%s) not found in dataframe – disabling",
                    col_name, attr,
                )
                setattr(self, attr, None)

        # rating range
        r_min, r_max = float(self.rating_range[0]), float(self.rating_range[1])
        if r_min >= r_max:
            logger.error("invalid rating_range: %s", self.rating_range)
            raise ValueError(
                f"rating_range must have min < max; got {self.rating_range}"
            )
        logger.debug("rating_range validated [%.2f, %.2f]", r_min, r_max)

    # ── scenario detection ───────────────────────────────────────────────────

    def _detect_scenario(self) -> ScenarioType:
        """Return the ScenarioType matching the available columns."""
        logger.debug("detecting scenario")
        hp = self.total_price_col is not None
        hd = self.discount_col    is not None
        ht = self.date_col        is not None

        if   hp and hd and ht: return ScenarioType.FULL
        elif not hp and not hd and not ht: return ScenarioType.MINIMAL
        elif not hp and not hd: return ScenarioType.NO_PRICE_DISCOUNT
        elif not hp and not ht: return ScenarioType.NO_PRICE_DATE
        elif not hd and not ht: return ScenarioType.NO_DISCOUNT_DATE
        elif not hp:            return ScenarioType.NO_PRICE
        elif not hd:            return ScenarioType.NO_DISCOUNT
        elif not ht:            return ScenarioType.NO_DATE
        return ScenarioType.FULL                       # pragma: no cover

    # ── signal bookkeeping ───────────────────────────────────────────────────

    def _get_available_signals(self) -> List[str]:
        """List of signal names that can be computed."""
        logger.debug("resolving available signals")
        sigs = ['frequency', 'quantity']               # always present
        if self.total_price_col is not None: sigs.append('spend')
        if self.date_col        is not None: sigs.append('recency')
        if self.discount_col    is not None: sigs.append('loyalty')
        logger.debug("available signals = %s", sigs)
        return sigs

    def _setup_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Merge user weights with defaults, prune unavailable, normalise."""
        logger.debug("setting up weights")
        w = {**self._DEFAULT_WEIGHTS, **(weights or {})}

        # keep only available signals
        avail = set(self.available_signals)
        w = {k: v for k, v in w.items() if k in avail}

        w_total = sum(w.values())
        if w_total == 0:
            w = {k: 1.0 / len(avail) for k in avail}
            logger.warning("all weights were zero → equal weights")
        elif abs(w_total - 1.0) > 1e-6:
            logger.warning(
                "weights sum to %.4f ≠ 1.0 – auto-normalising", w_total,
            )
            w = {k: v / w_total for k, v in w.items()}
        return w

    # ── SQL builder ──────────────────────────────────────────────────────────

    def _build_sql(self) -> str:
        """Build the aggregation SQL dynamically based on available columns."""
        logger.debug("building SQL for scenario=%s", self.scenario.value)

        # ── Step A: raw aggregation ──────────────────────────────────────────
        agg_parts = [
            f'"{self.user_col}"                              AS uid',
            f'"{self.item_col}"                             AS iid',
            'COUNT(*)                                       AS freq',
            f'SUM(CAST("{self.quantity_col}" AS DOUBLE))    AS total_qty',
        ]
        if self.total_price_col:
            agg_parts.append(
                f'SUM(CAST("{self.total_price_col}" AS DOUBLE)) AS total_spend'
            )
        if self.discount_col:
            agg_parts.append(
                f'1.0 - AVG(CAST("{self.discount_col}" AS DOUBLE)) AS loyalty'
            )
        if self.date_col:
            agg_parts.append(
                f'MAX(CAST("{self.date_col}" AS DATE))      AS last_date'
            )
        agg_sql = ",\n                ".join(agg_parts)

        # ── Step B: recency CTE (only when date present) ─────────────────────
        recency_cte   = ""
        signals_from  = "raw_agg"
        if self.date_col:
            recency_cte = """
        ,
        with_recency AS (
            SELECT *,
                DATEDIFF('day', last_date,
                         MAX(last_date) OVER ())            AS days_ago,
                MAX(DATEDIFF('day', last_date,
                             MAX(last_date) OVER ())) OVER () AS max_days_ago
            FROM raw_agg
        )"""
            signals_from = "with_recency"

        # ── Step C: signal computation ───────────────────────────────────────
        sig_parts = [
            'uid', 'iid',
            'LN(1.0 + CAST(freq AS DOUBLE))   AS log_freq',
            'LN(1.0 + total_qty)              AS log_qty',
        ]
        if self.total_price_col:
            sig_parts.append('LN(1.0 + total_spend)          AS log_spend')
        if self.discount_col:
            sig_parts.append('GREATEST(0.0, LEAST(1.0, loyalty)) AS loyalty')
        if self.date_col:
            sig_parts.append("""
                CASE
                    WHEN max_days_ago = 0 THEN 1.0
                    ELSE 1.0 - CAST(days_ago AS DOUBLE)
                             / CAST(max_days_ago AS DOUBLE)
                END                                        AS recency""")
        signals_sql = ",\n                ".join(sig_parts)

        # ── Step D: min-max normalisation ────────────────────────────────────
        norm_parts = ['uid', 'iid']
        if self.discount_col:
            norm_parts.append('loyalty')
        if self.date_col:
            norm_parts.append('recency')

        norm_parts.append("""
            (log_freq - MIN(log_freq) OVER ())
                / NULLIF(MAX(log_freq) OVER ()
                       - MIN(log_freq) OVER (), 0.0)       AS n_freq""")
        norm_parts.append("""
            (log_qty - MIN(log_qty) OVER ())
                / NULLIF(MAX(log_qty) OVER ()
                       - MIN(log_qty) OVER (), 0.0)        AS n_qty""")
        if self.total_price_col:
            norm_parts.append("""
            (log_spend - MIN(log_spend) OVER ())
                / NULLIF(MAX(log_spend) OVER ()
                       - MIN(log_spend) OVER (), 0.0)      AS n_spend""")
        normalized_sql = ",\n                ".join(norm_parts)

        # ── Step E: weighted blend ───────────────────────────────────────────
        score_terms = []
        if 'frequency' in self.weights:
            score_terms.append(
                f"{self.weights['frequency']} * COALESCE(n_freq, 0.5)"
            )
        if 'quantity' in self.weights:
            score_terms.append(
                f"{self.weights['quantity']} * COALESCE(n_qty, 0.5)"
            )
        if 'spend' in self.weights:
            score_terms.append(
                f"{self.weights['spend']} * COALESCE(n_spend, 0.5)"
            )
        if 'recency' in self.weights:
            score_terms.append(
                f"{self.weights['recency']} * COALESCE(recency, 0.5)"
            )
        if 'loyalty' in self.weights:
            score_terms.append(
                f"{self.weights['loyalty']} * COALESCE(loyalty, 0.5)"
            )
        score_expr = " + ".join(score_terms) if score_terms else "0.5"

        sql = f"""
        WITH
        raw_agg AS (
            SELECT
                {agg_sql}
            FROM RAW
            GROUP BY "{self.user_col}", "{self.item_col}"
        )
        {recency_cte},
        signals AS (
            SELECT
                {signals_sql}
            FROM {signals_from}
        ),
        normalized AS (
            SELECT
                {normalized_sql}
            FROM signals
        )
        SELECT
            uid   AS "{self.user_col}",
            iid   AS "{self.item_col}",
            CAST({score_expr} AS DOUBLE) AS raw_score
        FROM normalized
        ORDER BY "{self.user_col}", "{self.item_col}"
        """
        logger.debug("SQL built (%d chars)", len(sql))
        return sql

    # ── SQL features query (for ML methods) ──────────────────────────────────

    def _build_features_sql(self) -> Tuple[str, List[str]]:
        """
        Build SQL that returns all normalised signals as separate columns
        so they can be fed into sklearn models.
        Returns (sql_string, list_of_feature_column_names).
        """
        logger.debug("building features SQL")

        agg_parts = [
            f'"{self.user_col}"                              AS uid',
            f'"{self.item_col}"                             AS iid',
            'COUNT(*)                                       AS freq',
            f'SUM(CAST("{self.quantity_col}" AS DOUBLE))    AS total_qty',
        ]
        if self.total_price_col:
            agg_parts.append(
                f'SUM(CAST("{self.total_price_col}" AS DOUBLE)) AS total_spend'
            )
        if self.discount_col:
            agg_parts.append(
                f'1.0 - AVG(CAST("{self.discount_col}" AS DOUBLE)) AS loyalty'
            )
        if self.date_col:
            agg_parts.append(
                f'MAX(CAST("{self.date_col}" AS DATE))      AS last_date'
            )
        agg_sql = ", ".join(agg_parts)

        recency_cte  = ""
        signals_from = "raw_agg"
        if self.date_col:
            recency_cte = """
            , with_recency AS (
                SELECT *,
                    DATEDIFF('day', last_date, MAX(last_date) OVER ()) AS days_ago,
                    MAX(DATEDIFF('day', last_date, MAX(last_date) OVER ())) OVER () AS max_days_ago
                FROM raw_agg
            )"""
            signals_from = "with_recency"

        sig_parts = [
            'uid', 'iid',
            'LN(1.0 + CAST(freq AS DOUBLE)) AS log_freq',
            'LN(1.0 + total_qty) AS log_qty',
        ]
        if self.total_price_col:
            sig_parts.append('LN(1.0 + total_spend) AS log_spend')
        if self.discount_col:
            sig_parts.append('GREATEST(0.0, LEAST(1.0, loyalty)) AS loyalty_raw')
        if self.date_col:
            sig_parts.append("""
                CASE WHEN max_days_ago = 0 THEN 1.0
                     ELSE 1.0 - CAST(days_ago AS DOUBLE)
                              / CAST(max_days_ago AS DOUBLE)
                END AS recency""")
        signals_sql = ", ".join(sig_parts)

        norm_parts = ['uid', 'iid']
        feature_cols = ['n_freq', 'n_qty']

        norm_parts.append("""
            (log_freq - MIN(log_freq) OVER ())
                / NULLIF(MAX(log_freq) OVER () - MIN(log_freq) OVER (), 0.0) AS n_freq""")
        norm_parts.append("""
            (log_qty - MIN(log_qty) OVER ())
                / NULLIF(MAX(log_qty) OVER () - MIN(log_qty) OVER (), 0.0) AS n_qty""")

        if self.total_price_col:
            norm_parts.append("""
            (log_spend - MIN(log_spend) OVER ())
                / NULLIF(MAX(log_spend) OVER () - MIN(log_spend) OVER (), 0.0) AS n_spend""")
            feature_cols.append('n_spend')

        if self.discount_col:
            norm_parts.append('loyalty_raw AS n_loyalty')
            feature_cols.append('n_loyalty')

        if self.date_col:
            norm_parts.append('recency AS n_recency')
            feature_cols.append('n_recency')

        normalized_sql = ", ".join(norm_parts)

        sql = f"""
        WITH
        raw_agg AS (
            SELECT {agg_sql} FROM RAW
            GROUP BY "{self.user_col}", "{self.item_col}"
        )
        {recency_cte},
        signals AS (
            SELECT {signals_sql} FROM {signals_from}
        ),
        normalized AS (
            SELECT {normalized_sql} FROM signals
        )
        SELECT * FROM normalized
        """
        logger.debug("features SQL built, feature_cols=%s", feature_cols)
        return sql, feature_cols

    # ── DuckDB execution helpers ─────────────────────────────────────────────

    def _run_sql(self, sql: str, desc: str = "executing SQL") -> pd.DataFrame:
        """Register dataframe, run SQL, return result."""
        logger.debug("_run_sql: %s", desc)
        with tqdm(total=3, desc=desc, leave=False, ncols=90) as pbar:
            con = duckdb.connect(":memory:")
            try:
                con.register("RAW", self.data)
                pbar.update(1)
                logger.debug("RAW registered in DuckDB")

                result = con.execute(sql).df()
                pbar.update(1)
                logger.debug("SQL executed → %d rows", len(result))
            finally:
                con.unregister("RAW")
                con.close()
                pbar.update(1)
        return result

    # ── WEIGHTED scoring ─────────────────────────────────────────────────────

    def _score_weighted(self) -> pd.DataFrame:
        """Original weighted-average approach via DuckDB."""
        logger.info("scoring method = WEIGHTED")

        n_pairs = self.data.groupby([self.user_col, self.item_col]).ngroups
        logger.debug("%d rows → %d (user,item) pairs", len(self.data), n_pairs)

        sql = self._build_sql()
        result_df = self._run_sql(sql, desc="weighted scoring")

        self._check_nan_scores(result_df)
        logger.debug(
            "raw_score stats  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
            result_df["raw_score"].min(), result_df["raw_score"].max(),
            result_df["raw_score"].mean(), result_df["raw_score"].std(),
        )
        return result_df

    # ── feature extraction (shared by ML methods) ────────────────────────────

    def _extract_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Pull normalised signal columns into a pandas DataFrame."""
        logger.info("extracting feature matrix for ML scoring")
        sql, feat_cols = self._build_features_sql()
        df = self._run_sql(sql, desc="feature extraction")

        for col in feat_cols:
            if col in df.columns and df[col].isna().any():
                n_null = int(df[col].isna().sum())
                logger.warning(
                    "feature '%s' has %d NaN → filling 0.5", col, n_null,
                )
                df[col] = df[col].fillna(0.5)

        logger.info("extracted %d features for %d pairs", len(feat_cols), len(df))
        return df, feat_cols

    # ── LASSO scoring ────────────────────────────────────────────────────────

    def _score_lasso(self) -> pd.DataFrame:
        """L1-regularised regression – learns sparse signal weights."""
        logger.info("scoring method = LASSO  alpha=%.4f", self.lasso_alpha)

        feat_df, feat_cols = self._extract_features()
        X = feat_df[feat_cols].values.astype(np.float64)

        # synthetic target = sum of signals (engagement proxy)
        y = X.sum(axis=1)

        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)

        with tqdm(total=3, desc="Lasso fitting", leave=False, ncols=90) as pbar:
            lasso = Lasso(
                alpha=self.lasso_alpha, positive=True,
                max_iter=10_000, tol=1e-4,
            )
            lasso.fit(X_s, y)
            pbar.update(1)

            coefs = dict(zip(feat_cols, lasso.coef_))
            logger.debug("Lasso coefficients: %s", coefs)

            n_zero = int(np.sum(lasso.coef_ == 0))
            if n_zero:
                logger.warning(
                    "Lasso zeroed %d/%d coefficients (feature selection)",
                    n_zero, len(feat_cols),
                )

            scores = lasso.predict(X_s)
            pbar.update(1)

            scores = self._minmax_1d(scores)
            pbar.update(1)

        out = feat_df[[self.user_col, self.item_col]].copy()
        out['raw_score'] = scores.astype(np.float32)
        self._log_score_stats("Lasso", scores)
        return out

    # ── RIDGE scoring ────────────────────────────────────────────────────────

    def _score_ridge(self) -> pd.DataFrame:
        """L2-regularised regression – keeps all signals with shrunk weights."""
        logger.info("scoring method = RIDGE  alpha=%.4f", self.ridge_alpha)

        feat_df, feat_cols = self._extract_features()
        X = feat_df[feat_cols].values.astype(np.float64)
        y = X.sum(axis=1)

        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)

        with tqdm(total=3, desc="Ridge fitting", leave=False, ncols=90) as pbar:
            ridge = Ridge(alpha=self.ridge_alpha, positive=True)
            ridge.fit(X_s, y)
            pbar.update(1)

            coefs = dict(zip(feat_cols, ridge.coef_))
            logger.debug("Ridge coefficients: %s", coefs)

            scores = ridge.predict(X_s)
            pbar.update(1)

            scores = self._minmax_1d(scores)
            pbar.update(1)

        out = feat_df[[self.user_col, self.item_col]].copy()
        out['raw_score'] = scores.astype(np.float32)
        self._log_score_stats("Ridge", scores)
        return out

    # ── PCA scoring ──────────────────────────────────────────────────────────

    def _score_pca(self) -> pd.DataFrame:
        """First principal component as composite engagement score."""
        logger.info("scoring method = PCA  components=%d", self.pca_components)

        feat_df, feat_cols = self._extract_features()
        X = feat_df[feat_cols].values.astype(np.float64)

        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)

        n_comp = min(self.pca_components, X_s.shape[1])

        with tqdm(total=3, desc="PCA fitting", leave=False, ncols=90) as pbar:
            pca = PCA(n_components=n_comp)
            raw = pca.fit_transform(X_s)
            pbar.update(1)

            logger.debug(
                "PCA explained_variance_ratio_ = %s",
                pca.explained_variance_ratio_,
            )

            # take first PC, shift to non-negative
            scores = raw[:, 0].copy()
            if scores.min() < 0:
                scores -= scores.min()
            pbar.update(1)

            scores = self._minmax_1d(scores)
            pbar.update(1)

        out = feat_df[[self.user_col, self.item_col]].copy()
        out['raw_score'] = scores.astype(np.float32)
        self._log_score_stats("PCA", scores)
        return out

    # ── EQUAL scoring ────────────────────────────────────────────────────────

    def _score_equal(self) -> pd.DataFrame:
        """Simple arithmetic mean of all available normalised signals."""
        logger.info("scoring method = EQUAL (unweighted mean)")

        feat_df, feat_cols = self._extract_features()
        X = feat_df[feat_cols].values.astype(np.float64)

        with tqdm(total=2, desc="equal-weight scoring", leave=False, ncols=90) as pbar:
            scores = X.mean(axis=1)
            pbar.update(1)
            scores = np.clip(scores, 0.0, 1.0)
            pbar.update(1)

        out = feat_df[[self.user_col, self.item_col]].copy()
        out['raw_score'] = scores.astype(np.float32)
        self._log_score_stats("Equal", scores)
        return out

    # ── shared utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _minmax_1d(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise a 1-D array to [0, 1]."""
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            logger.warning("constant score array → filling 0.5")
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _check_nan_scores(df: pd.DataFrame) -> None:
        """Fill NaN raw_scores with 0.5 and warn."""
        if df["raw_score"].isna().any():
            n = int(df["raw_score"].isna().sum())
            logger.warning(
                "%d raw_score NaN → filling 0.5", n,
            )
            df["raw_score"] = df["raw_score"].fillna(0.5)

    def _log_score_stats(self, label: str, scores: np.ndarray) -> None:
        logger.info(
            "%s scores  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
            label, scores.min(), scores.max(), scores.mean(), scores.std(),
        )

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self) -> pd.DataFrame:
        """
        Compute pseudo-ratings and return a DataFrame with columns
        ``[user_col, item_col, 'pseudo_rating']``.
        """
        logger.info("fit() starting  method=%s  scenario=%s",
                     self.method.value, self.scenario.value)

        dispatch = {
            ScoringMethod.WEIGHTED: self._score_weighted,
            ScoringMethod.LASSO:    self._score_lasso,
            ScoringMethod.RIDGE:    self._score_ridge,
            ScoringMethod.PCA:      self._score_pca,
            ScoringMethod.EQUAL:    self._score_equal,
        }
        scorer = dispatch.get(self.method)
        if scorer is None:
            logger.error("unknown method: %s", self.method)
            raise ValueError(f"Unknown scoring method: {self.method}")

        result_df = scorer()

        # scale to [r_min, r_max]
        r_min, r_max = float(self.rating_range[0]), float(self.rating_range[1])
        with tqdm(total=1, desc="scaling to rating range", leave=False, ncols=90) as pbar:
            result_df["pseudo_rating"] = (
                r_min + result_df["raw_score"] * (r_max - r_min)
            ).round(4).astype(np.float32)
            result_df = result_df.drop(columns=["raw_score"])
            pbar.update(1)

        self._log_summary(result_df)
        gc.collect()
        logger.info("fit() completed  rows=%d", len(result_df))
        return result_df

    # ── summary logging ──────────────────────────────────────────────────────

    def _log_summary(self, df: pd.DataFrame) -> None:
        """Log final statistics and cold-start warnings."""
        logger.debug("computing summary statistics")
        n_pairs = len(df)
        n_users = df[self.user_col].nunique()
        n_items = df[self.item_col].nunique()
        pr      = df["pseudo_rating"]

        logger.info(
            "summary  pairs=%d  users=%d  items=%d  "
            "rating[min=%.3f  mean=%.3f  max=%.3f  std=%.3f]",
            n_pairs, n_users, n_items,
            pr.min(), pr.mean(), pr.max(), pr.std(),
        )

        cold_users = int(
            (df.groupby(self.user_col)["pseudo_rating"].count() == 1).sum()
        )
        cold_items = int(
            (df.groupby(self.item_col)["pseudo_rating"].count() == 1).sum()
        )
        if cold_users:
            logger.warning(
                "%d user(s) in only 1 pair → cold-start risk", cold_users,
            )
        if cold_items:
            logger.warning(
                "%d item(s) in only 1 pair → cold-start risk", cold_items,
            )

    # ── convenience class methods ────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        data: pd.DataFrame,
        method: str = "weighted",
        **kwargs,
    ) -> pd.DataFrame:
        """One-shot convenience: instantiate, fit, return."""
        logger.info("CounterFeitRatingEngine.run()  method=%s", method)
        engine = cls(data=data, method=method, **kwargs)
        return engine.fit()



def CounterFeit_RateGen(
        data            : pd.DataFrame,
        discount_col    : str = None,
        date_col        : str = "SalesDate",
        weights         : Dict[str, float] = None,
    ):
    #DetectReco_Identifier
    #
    pass



if __name__ == "__main__":
    pass