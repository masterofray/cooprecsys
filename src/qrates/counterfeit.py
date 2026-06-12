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
from   tqdm.auto import tqdm
from   enum      import Enum
from   typing    import Dict, Optional, Tuple, List
from   jinja2                import Environment, BaseLoader
from   sklearn.linear_model  import Lasso, Ridge
from   sklearn.preprocessing import MinMaxScaler
from   sklearn.decomposition import PCA

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[1]))
from configs import logger, _cfg, _cfglist
from prepare import DetectReco_Identifier
from db      import DuckDBManager


fp01 = LocDir.parent/"Weighted_Score.sql"
with fp01.open("r", encoding = "utf-8") as f01:
    WEIGHTED_SCORING_SQL = f01.read()

fp02 = LocDir.parent/"Features_Aggs.sql"
with fp02.open("r", encoding = "utf-8") as f02:
    FEATURES_SQL = f02.read()


class ScenarioType(Enum):
    FULL               = "full"
    NO_PRICE           = "no_price"
    NO_DISCOUNT        = "no_discount"
    NO_DATE            = "no_date"
    NO_PRICE_DISCOUNT  = "no_price_discount"
    NO_PRICE_DATE      = "no_price_date"
    NO_DISCOUNT_DATE   = "no_discount_date"
    MINIMAL            = "minimal"

class ScoringMethod(Enum):
    WEIGHTED = "weighted"
    LASSO    = "lasso"
    RIDGE    = "ridge"
    PCA      = "pca"
    EQUAL    = "equal"


class CounterFeitRatingEngine:
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
        data: pd.DataFrame,
        user_col: str = "CustomerID",
        item_col: str = "CategoryID",
        quantity_col: str = "Quantity",
        total_price_col: Optional[str] = "TotalPrice",
        discount_col: Optional[str] = "Discount",
        date_col: Optional[str] = "SalesDate",
        rating_range: Tuple[float, float] = (1.0, 5.0),
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted",
        lasso_alpha: float = 0.01,
        ridge_alpha: float = 1.0,
        pca_components: int = 1,
    ):
        if rating_range is None:
            rating_range     = _cfglist(_cfg, 'RATING', 'range')
        if method is None:
            method           = _cfg.get('RATING', 'methodscore')
        logger.info(
            "CounterFeitRatingEngine.__init__  data.shape=%s | method=%s",
            data.shape, method,
        )

        self.data            = data.copy()
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

        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=BaseLoader())
        logger.debug("Jinja2 environment initialized")

        self.method = ScoringMethod(method)
        self._validate_inputs()

        self.scenario = self._detect_scenario()
        logger.info("detected scenario: %s", self.scenario.value)

        self.available_signals = self._get_available_signals()
        self.weights = self._setup_weights(weights)
        logger.debug("effective weights = %s", self.weights)

        logger.info("engine ready  signals=%s  scenario=%s",
                    self.available_signals, self.scenario.value)

    def _validate_inputs(self) -> None:
        logger.debug("validating input columns")

        base_required = {self.user_col, self.item_col, self.quantity_col}
        missing_base  = base_required - set(self.data.columns)
        if missing_base:
            logger.error("required base columns missing: %s", missing_base)
            raise ValueError(f"Required base columns missing: {missing_base}")

        _optional = {
            'total_price_col': self.total_price_col,
            'discount_col':    self.discount_col,
            'date_col':        self.date_col,
        }
        for attr, col_name in _optional.items():
            if col_name is not None and col_name not in self.data.columns:
                logger.warning(
                    "column '%s' (%s) not found – disabling", col_name, attr,
                )
                setattr(self, attr, None)

        r_min, r_max = float(self.rating_range[0]), float(self.rating_range[1])
        if r_min >= r_max:
            logger.error("invalid rating_range: %s", self.rating_range)
            raise ValueError(f"rating_range must have min < max; got {self.rating_range}")
        logger.debug("rating_range validated [%.2f, %.2f]", r_min, r_max)

    def _detect_scenario(self) -> ScenarioType:
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
        return ScenarioType.FULL

    def _get_available_signals(self) -> List[str]:
        logger.debug("resolving available signals")
        sigs = ['frequency', 'quantity']
        if self.total_price_col is not None: sigs.append('spend')
        if self.date_col        is not None: sigs.append('recency')
        if self.discount_col    is not None: sigs.append('loyalty')
        logger.debug("available signals = %s", sigs)
        return sigs

    def _setup_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        logger.debug("setting up weights")
        w = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        avail = set(self.available_signals)
        w = {k: v for k, v in w.items() if k in avail}

        w_total = sum(w.values())
        if w_total == 0:
            w = {k: 1.0 / len(avail) for k in avail}
            logger.warning("all weights were zero ->> equal weights")
        elif abs(w_total - 1.0) > 1e-6:
            logger.warning("weights sum to %.4f ≠ 1.0 – auto-normalising", w_total)
            w = {k: v / w_total for k, v in w.items()}
        return w

    def _build_score_expression(self) -> str:
        """Build the weighted score expression for Jinja2 template."""
        logger.debug("building score expression")
        terms = list()
        if 'frequency' in self.weights:
            terms.append(f"{self.weights['frequency']} * COALESCE(n_freq, 0.5)")
        if 'quantity' in self.weights:
            terms.append(f"{self.weights['quantity']} * COALESCE(n_qty, 0.5)")
        if 'spend' in self.weights:
            terms.append(f"{self.weights['spend']} * COALESCE(n_spend, 0.5)")
        if 'recency' in self.weights:
            terms.append(f"{self.weights['recency']} * COALESCE(recency, 0.5)")
        if 'loyalty' in self.weights:
            terms.append(f"{self.weights['loyalty']} * COALESCE(loyalty, 0.5)")
        
        expr = " + ".join(terms) if terms else "0.5"
        logger.debug("score expression: %s", expr)
        return expr

    def _render_weighted_sql(self) -> str:
        """Render the weighted scoring SQL using Jinja2."""
        logger.debug("rendering weighted SQL template")
        template = self.jinja_env.from_string(WEIGHTED_SCORING_SQL)
        
        context = {
            'user_col':        self.user_col,
            'item_col':        self.item_col,
            'quantity_col':    self.quantity_col,
            'total_price_col': self.total_price_col,
            'discount_col':    self.discount_col,
            'date_col':        self.date_col,
            'score_expression': self._build_score_expression(),
        }
        
        sql = template.render(**context)
        logger.debug("rendered SQL (%d chars)", len(sql))
        return sql

    def _render_features_sql(self) -> str:
        """Render the features SQL using Jinja2."""
        logger.debug("rendering features SQL template")
        template = self.jinja_env.from_string(FEATURES_SQL)
        
        context = {
            'user_col':        self.user_col,
            'item_col':        self.item_col,
            'quantity_col':    self.quantity_col,
            'total_price_col': self.total_price_col,
            'discount_col':    self.discount_col,
            'date_col':        self.date_col,
        }
        
        sql = template.render(**context)
        logger.debug("rendered features SQL (%d chars)", len(sql))
        return sql

    def _run_sql(self, sql: str, desc: str = "executing SQL") -> pd.DataFrame:
        logger.debug("_run_sql: %s", desc)
        with tqdm(total=3, desc=desc, leave=False, ncols=90) as pbar:
            con = duckdb.connect(":memory:")
            try:
                con.register("RAW", self.data)
                pbar.update(1)
                logger.debug("RAW registered in DuckDB")

                result = con.execute(sql).df()
                pbar.update(1)
                logger.debug("SQL executed ->> %d rows", len(result))
            finally:
                con.unregister("RAW")
                con.close()
                pbar.update(1)
        return result

    def _score_weighted(self) -> pd.DataFrame:
        logger.info("scoring method = WEIGHTED")
        n_pairs = self.data.groupby([self.user_col, self.item_col]).ngroups
        logger.debug("%d rows ->> %d (user,item) pairs", len(self.data), n_pairs)

        sql = self._render_weighted_sql()
        result_df = self._run_sql(sql, desc="weighted scoring")

        self._check_nan_scores(result_df)
        logger.debug(
            "raw_score stats  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
            result_df["raw_score"].min(), result_df["raw_score"].max(),
            result_df["raw_score"].mean(), result_df["raw_score"].std(),
        )
        return result_df

    def _extract_features(self) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("extracting feature matrix for ML scoring")
        sql = self._render_features_sql()
        df = self._run_sql(sql, desc="feature extraction")

        feat_cols = ['n_freq', 'n_qty']
        if self.total_price_col: feat_cols.append('n_spend')
        if self.discount_col:    feat_cols.append('n_loyalty')
        if self.date_col:        feat_cols.append('n_recency')

        for col in feat_cols:
            if col in df.columns and df[col].isna().any():
                n_null = int(df[col].isna().sum())
                logger.warning("feature '%s' has %d NaN ->> filling 0.5", col, n_null)
                df[col] = df[col].fillna(0.5)

        logger.info("extracted %d features for %d pairs", len(feat_cols), len(df))
        return df, feat_cols

    def _score_lasso(self) -> pd.DataFrame:
        logger.info("scoring method = LASSO  alpha=%.4f", self.lasso_alpha)

        feat_df, feat_cols = self._extract_features()
        X = feat_df[feat_cols].values.astype(np.float64)
        y = X.sum(axis=1)

        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)

        with tqdm(total=3, desc="Lasso fitting", leave=False, ncols=90) as pbar:
            lasso = Lasso(alpha=self.lasso_alpha, positive=True, max_iter=10_000, tol=1e-4)
            lasso.fit(X_s, y)
            pbar.update(1)

            coefs = dict(zip(feat_cols, lasso.coef_))
            logger.debug("Lasso coefficients: %s", coefs)

            n_zero = int(np.sum(lasso.coef_ == 0))
            if n_zero:
                logger.warning("Lasso zeroed %d/%d coefficients", n_zero, len(feat_cols))

            scores = lasso.predict(X_s)
            pbar.update(1)
            scores = self._minmax_1d(scores)
            pbar.update(1)

        out = feat_df[[self.user_col, self.item_col]].copy()
        out['raw_score'] = scores.astype(np.float32)
        self._log_score_stats("Lasso", scores)
        return out

    def _score_ridge(self) -> pd.DataFrame:
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

    def _score_pca(self) -> pd.DataFrame:
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

            logger.debug("PCA explained_variance_ratio_ = %s", pca.explained_variance_ratio_)

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

    def _score_equal(self) -> pd.DataFrame:
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

    @staticmethod
    def _minmax_1d(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            logger.warning("constant score array ->> filling 0.5")
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _check_nan_scores(df: pd.DataFrame) -> None:
        if df["raw_score"].isna().any():
            n = int(df["raw_score"].isna().sum())
            logger.warning("%d raw_score NaN ->> filling 0.5", n)
            df["raw_score"] = df["raw_score"].fillna(0.5)

    def _log_score_stats(self, label: str, scores: np.ndarray) -> None:
        logger.info(
            "%s scores  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
            label, scores.min(), scores.max(), scores.mean(), scores.std(),
        )

    def fit(self) -> pd.DataFrame:
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

    def _log_summary(self, df: pd.DataFrame) -> None:
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

        cold_users = int((df.groupby(self.user_col)["pseudo_rating"].count() == 1).sum())
        cold_items = int((df.groupby(self.item_col)["pseudo_rating"].count() == 1).sum())
        if cold_users:
            logger.warning("%d user(s) in only 1 pair ->> cold-start risk", cold_users)
        if cold_items:
            logger.warning("%d item(s) in only 1 pair ->> cold-start risk", cold_items)

    @classmethod
    def run(cls, data: pd.DataFrame, method: str = "weighted", **kwargs) -> pd.DataFrame:
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