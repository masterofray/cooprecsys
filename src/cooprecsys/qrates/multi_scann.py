#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-07"


import gc
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from jinja2 import Template
from typing import Dict, List, Tuple, Any
try:
    from scann import scann_ops_pybind as sopy
    SCANN_AVAILABLE = True
except ImportError:
    sopy = None
    SCANN_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture
from ..configs import _cfg, logger, _cfglist
from ..db      import duckdb_connection

LocDir = Path(__file__).resolve().parents[1]


def DFMerger(DataArray: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a collection of Quasi Rating DataFrames 
    using in-memory DuckDB. Ensures a strict 1-to-1
    relationship based on 'original_index' to prevent 
    Cartesian explosions while maintaining operational latency.
    The parameter is DataArray (List[pd.DataFrame]): 
    A list of Pandas DataFrames. Each DataFrame must 
    contain 'original_index', 'query_index', and one 
    column starting with 'Quasi_Rating_'.
    """
    logger.debug(f"Initiating merge engine for {len(DataArray)} DataFrames.")
    if not DataArray:
        logger.warning("Input DataArray is empty.")
        return pd.DataFrame([])

    Final      = pd.DataFrame([])
    QueryJinja = """
    WITH 
        {% for group in groups %}
        {{ group }}_dedup AS (
        SELECT 
            original_index,
            ANY_VALUE(Quasi_Rating_{{ group }}) AS Quasi_Rating_{{ group }}
        FROM 
            vw_{{ group }}
        GROUP BY 
            original_index
        ){% if not loop.last %},{% endif %}
        {% endfor %}

    SELECT
        *
    FROM 
        {{ groups[0] }}_dedup
    {% for group in groups[1:] %}
    FULL OUTER JOIN 
        {{ group }}_dedup USING (original_index)
    {% endfor %}
    """

    try:
        group_names: List[str] = list()
        with duckdb_connection() as con:
            for idx, KdfScann in enumerate(DataArray):
                RateColumn = next((c for c in KdfScann.columns 
                                   if c.startswith('Quasi_Rating_')),
                                   None)
                if not RateColumn:
                    logger.error(f"DataFrame at index {idx} ignored:"
                    " Missing column starting with 'Quasi_Rating_'.")
                    continue
                grop      = RateColumn.replace('Quasi_Rating_', '')
                group_names.append(grop)
                view_name = f"vw_{grop}"
                con.register_dataframe(view_name, KdfScann)
                logger.debug(f"View '{view_name}' successfully "
                             f"registered with {len(KdfScann)} rows.")

            if not group_names:
                logger.warning("No valid DataFrames with target "
                               "columns found. Aborting operation.")
                return pd.DataFrame([])
            logger.debug("Generating optimized query for "
                        f"{len(group_names)} matrices: {group_names}")
            DoSQL    = Template(QueryJinja)
            SQLquery = DoSQL.render(groups = group_names)
            Final    = con.query(SQLquery)
        logger.debug("Merge completed. Generated output DataFrame "
                    f"with {len(Final)} rows and "
                    f"{len(Final.columns)} columns.")

    except Exception as arc:
        logger.exception(f"Fatal failure in DuckDB execution: {str(arc)}")
        raise ValueError()

    finally:
        logger.debug("DuckDB memory connection closed gracefully.")
        return Final



class QuasiRate_ScaNN:
    """
    A multi-backend Quasi-Rating engine for heterogeneous feature groups.

    Uses ScaNN when installed, otherwise falls back to an sklearn
    Gaussian Mixture Model (GMM) backend. The fallback is probabilistic
    and unsupervised; it does not depend on KNN.
    """
    def __init__(self, 
                 feature_groups: Dict[str, List[str]], 
                 scann_config  : Dict[str, Any] = None,
                ):
        """
        Initializes the index architecture.
        feature_groups: A dictionary where keys are group names and values are lists 
                        of feature column names.
                        Example: {"financial": ["price", "discount"], 
                                  "demographic": ["age"]}
        scann_config  : Custom hyperparameters for ScaNN (optional).
        """
        self.feature_groups = feature_groups
        self.group_names    = list(feature_groups.keys())
        default_config = {
            "num_leaves_ratio": 0.15,
            "num_leaves_to_search": 10,
            "anisotropic_quantization_threshold": 0.2,
            # sklearn fallback configuration
            "gmm_components": 8,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1e-5,
            "gmm_max_iter": 200,
            "gmm_n_init": 1,
            "gmm_random_state": 42,
            "gmm_query_chunk_size": 128,
        }
        self.scann_config = {**default_config, **(scann_config or {})}
        self.scalers   : Dict[str, StandardScaler] = dict()
        self.searchers : Dict[str, Any]            = dict()
        # sklearn fallback: one unsupervised probabilistic model per feature group.
        self.models    : Dict[str, GaussianMixture] = dict()
        self._item_embeddings: Dict[str, np.ndarray] = dict()
        self._backend  = "scann" if SCANN_AVAILABLE else "gmm"
        self.unidata       = pd.DataFrame([])
        self._original     = pd.DataFrame([])
        self._is_fitted    = False
        self._reference_df = None


    def _l2_normalize(self, 
        vectors: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to vectors to ensure 
        Dot Product functions as Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
        norms[norms == 0] = 1e-10
        return vectors / norms


    def fit(self, Data: pd.DataFrame) -> 'QuasiRate_ScaNN':
        """
        Builds one model per feature group.

        Backend selection:
        - ScaNN, when the optional ``scann`` package is installed.
        - sklearn Gaussian Mixture Model (GMM), otherwise.

        The GMM fallback is intentionally *not* a KNN implementation.
        It learns a probability-density representation for each feature
        group and compares query/item posterior representations using
        cosine distance.
        """
        logger.debug(
            f"Starting model construction for {len(self.group_names)} "
            f"feature groups using backend='{self._backend}'."
        )

        require = [
            feat for group in self.feature_groups.values() for feat in group
        ]
        miss = [col for col in require if col not in Data.columns]
        if miss:
            logger.error(
                f"The following columns are missing from the dataset: {miss}"
            )
            raise ValueError(f"Missing required columns: {miss}")

        self._original = deepcopy(Data)
        self._reference_df = deepcopy(Data.reset_index(drop=True))

        num_rows = len(self._reference_df)
        if num_rows == 0:
            logger.error(
                "The input DataFrame is empty. Cannot build a model with zero records."
            )
            raise ValueError("Input DataFrame is empty.")

        # Existing ScaNN configuration is preserved for compatibility.
        num_leaves = max(
            10,
            int(num_rows * self.scann_config["num_leaves_ratio"])
        )
        if num_rows < num_leaves:
            num_leaves = max(1, num_rows // 2)
            logger.warning(
                f"Dataset size ({num_rows}) is smaller than configured "
                f"num_leaves. Adjusted num_leaves to {num_leaves}."
            )

        for group_name, features in tqdm(
            self.feature_groups.items(),
            desc="Building ScaNN/GMM Models",
            colour=_cfg.get('tqdm', 'colour'),
            ncols=_cfg.getint('tqdm', 'ncols'),
            bar_format=_cfg.get('tqdm', 'BarFormats'),
            unit='Group',
            mininterval=0.1
        ):
            total_dims = len(features)
            if total_dims == 0:
                logger.error(f"Feature group '{group_name}' has no features.")
                raise ValueError(f"Feature group '{group_name}' has no features.")

            RAWdata = self._reference_df[features].copy()
            if RAWdata.isnull().values.any():
                logger.warning(
                    f"NaN values detected in group '{group_name}'. "
                    "Filling with column medians."
                )
                RAWdata = RAWdata.fillna(RAWdata.median())

            RAW_data = RAWdata.values.astype(np.float32)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(RAW_data)
            self.scalers[group_name] = scaler

            if SCANN_AVAILABLE:
                # Keep the original ScaNN path intact when the optional
                # dependency is actually available.
                normdata = self._l2_normalize(scaled_data)
                dim_block = (
                    2 if (total_dims >= 2 and total_dims % 2 == 0) else 1
                )
                nleaf = self.scann_config["num_leaves_to_search"]
                threshold = self.scann_config[
                    "anisotropic_quantization_threshold"
                ]

                searcher = (
                    sopy.builder(
                        normdata,
                        num_neighbors=10,
                        distance_measure="squared_l2",
                    )
                    .tree(
                        num_leaves=num_leaves,
                        num_leaves_to_search=nleaf,
                        training_sample_size=min(num_rows, 100_000),
                    )
                    .score_ah(
                        dimensions_per_block=dim_block,
                        anisotropic_quantization_threshold=threshold,
                    )
                    .reorder(100)
                    .build()
                )
                self.searchers[group_name] = searcher

            else:
                # ---------- sklearn fallback: Gaussian Mixture Model ----------
                # Unlike KNN, GMM learns a latent probabilistic description of
                # the feature distribution.  The item embedding is the vector
                # of posterior component probabilities P(component | x).
                requested_components = int(
                    self.scann_config.get("gmm_components", 8)
                )
                n_components = max(
                    1, min(requested_components, num_rows)
                )

                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=self.scann_config.get(
                        "gmm_covariance_type", "diag"
                    ),
                    reg_covar=float(
                        self.scann_config.get("gmm_reg_covar", 1e-5)
                    ),
                    max_iter=int(
                        self.scann_config.get("gmm_max_iter", 200)
                    ),
                    n_init=int(
                        self.scann_config.get("gmm_n_init", 1)
                    ),
                    random_state=int(
                        self.scann_config.get("gmm_random_state", 42)
                    ),
                )
                gmm.fit(scaled_data)

                item_posteriors = gmm.predict_proba(scaled_data).astype(
                    np.float32, copy=False
                )
                self.models[group_name] = gmm
                self._item_embeddings[group_name] = normalize(
                    item_posteriors, norm="l2", axis=1
                ).astype(np.float32, copy=False)

                logger.debug(
                    f"GMM '{group_name}' fitted with "
                    f"{n_components} components, covariance_type="
                    f"'{gmm.covariance_type}'."
                )

        self._is_fitted = True
        logger.debug(
            f"Model construction completed successfully using "
            f"backend='{self._backend}'."
        )
        gc.collect()
        return self

    def RateUnified(self, 
                    aggweighted  : bool = False, 
                    weights      : Dict[str, float] = dict(), 
                    invert_score : bool = False,
                    rating_range : Tuple[float, float] = (1.0, 5.0),
                   ) -> pd.DataFrame:
        """Helper method to calculate the unified 'final_rquasi' column."""
        RateColumn = [c for c in self.unidata.columns
                      if c.startswith('Quasi_Rating_')]
        if not RateColumn:
            logger.warning("No rating columns available to "
                           "calculate final_rquasi.")
            self.unidata['final_rquasi'] = np.nan
            return

        if not aggweighted:
            self.unidata['final_rquasi'] = self.unidata[RateColumn].mean(axis = 1)
        else:
            if not weights:
                logger.error('Your weights parameter is empty.')
                raise ValueError()
            available_groups    = [c.replace('Quasi_Rating_', '') for c in RateColumn]
            global_weights      = {g: weights.get(g, 0.0) for g in available_groups}
            total_global_weight = sum(global_weights.values())
            if total_global_weight == 0:
                logger.warning("Total weight is zero. Setting final_rquasi to 0.0.")
                self.unidata['final_rquasi'] = 0.0
            else:
                norm_weights    = {g: w / total_global_weight
                                      for g, w in global_weights.items()}
                self.unidata['final_rquasi'] = sum(self.unidata[
                f'Quasi_Rating_{g}'] * w for g, w in norm_weights.items())

        if invert_score:
            #Convert distance to similarity score (1.0 is 
            #perfect match, approaches 0.0 for large distances)
            self.unidata['final_rquasi'] = 1.0 / (1.0 + self.unidata['final_rquasi'])
        else:
            # Formula: Score = Min + Span * e^(-0.5 * Distance)
            # This ensures: Dist = 0 -> 5.0, Dist-> Inf -> 1.0
            # if alpha be increased, you make scores drop faster as distance increases
            alpha                = 0.5  # Decay factor
            min_score, max_score = rating_range
            score_span           = max_score - min_score
            self.unidata['final_rquasi'] = min_score + score_span * np.exp(
                                           -alpha * self.unidata['final_rquasi'])


    def search(
        self,
        query_dict: Dict[str, float],
        k: int = 5,
        use_norm: bool = False,
        aggweighted: bool = False,
        weights: Dict[str, float] = None,
        invert_score: bool = False,
        rating_range: Tuple[float, float] = (1.0, 5.0),
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Executes queries against all feature-group models.

        ScaNN returns squared-L2 distances.  The sklearn GMM backend instead
        computes cosine distance between the query's posterior-probability
        embedding and each item's posterior embedding:

            distance = 1 - cosine_similarity

        Therefore both backends preserve the original contract where a lower
        ``Quasi_Rating_<group>`` value means a better match.
        """
        results = dict()
        DataMerge = list()

        if not self._is_fitted:
            logger.error(
                "The model has not been fitted yet. Please invoke .fit(Data) first."
            )
            raise RuntimeError("Model is not fitted.")

        k = max(1, min(int(k), len(self._reference_df)))

        for group_name, features in tqdm(
            self.feature_groups.items(),
            desc='Executing Search Queries',
            colour=_cfg.get('tqdm', 'colour'),
            ncols=_cfg.getint('tqdm', 'ncols'),
            bar_format=_cfg.get('tqdm', 'BarFormats'),
            unit='Group',
            mininterval=0.1
        ):
            misskey = [f for f in features if f not in query_dict]
            if misskey:
                logger.warning(
                    f"Query for group '{group_name}' is skipped. "
                    f"Missing keys: {misskey}"
                )
                continue

            q_vector = np.array(
                [[query_dict[f] for f in features]], dtype=np.float32
            )
            q_scaled = self.scalers[group_name].transform(q_vector)
            q_scaled = np.nan_to_num(q_scaled, nan=0.0)

            if SCANN_AVAILABLE:
                quasi = (
                    self._l2_normalize(q_scaled)[0]
                    if use_norm
                    else q_scaled[0].copy()
                )
                neighbors, distances = self.searchers[group_name].search(
                    quasi,
                    final_num_neighbors=k
                )
                distances = np.asarray(distances, dtype=np.float32)

            else:
                if use_norm:
                    logger.debug(
                        "GMM fallback uses standardized features internally; "
                        "use_norm is ignored to keep query/model space consistent."
                    )

                model = self.models[group_name]
                q_posterior = model.predict_proba(q_scaled).astype(
                    np.float32, copy=False
                )
                q_embedding = normalize(
                    q_posterior, norm="l2", axis=1
                )[0]

                item_embedding = self._item_embeddings[group_name]

                # Cosine similarity in posterior-probability space.
                similarities = item_embedding @ q_embedding
                top_idx = np.argpartition(
                    -similarities, k - 1
                )[:k]
                top_idx = top_idx[np.argsort(-similarities[top_idx])]

                neighbors = top_idx.astype(np.int64, copy=False)
                distances = (1.0 - similarities[neighbors]).astype(
                    np.float32, copy=False
                )

            res_df = self._reference_df.iloc[neighbors].copy()
            res_df['original_index'] = neighbors
            res_df[f'Quasi_Rating_{group_name}'] = distances
            results[group_name] = res_df
            DataMerge.append(res_df)

        logger.debug(
            f'Search results generated for {len(results)} groups.'
        )

        if not DataMerge:
            return results, pd.DataFrame()

        if aggweighted and not weights:
            logger.error(
                "Weights dictionary must be provided "
                "when aggregation_method is 'weighted'."
            )
            raise ValueError(
                "weights must be provided when aggweighted=True"
            )

        self.unidata = DFMerger(DataMerge)
        self.RateUnified(
            aggweighted=aggweighted,
            weights=weights,
            invert_score=invert_score,
            rating_range=rating_range
        )
        ASC = not invert_score
        self.unidata = self.unidata.sort_values(
            by=['final_rquasi'],
            ascending=[ASC]
        ).reset_index(drop=True)
        return results, self.unidata

    def search_batch(
        self,
        data: pd.DataFrame,
        k: int = 5,
        use_norm: bool = False,
        aggweighted: bool = False,
        weights: Dict[str, float] = None,
        invert_score: bool = False,
        rating_range: Tuple[float, float] = (1.0, 5.0),
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Executes batched queries against all feature-group models.

        In GMM fallback mode, posterior-probability embeddings are computed
        in batch and ranked by cosine distance to the learned item
        representations. This remains an ML-model-based approach and does
        not instantiate sklearn KNeighbors classes.
        """
        BATCH = dict()
        DataMerge = list()

        if not self._is_fitted:
            logger.error(
                "The model has not been fitted yet. Please invoke .fit(Data) first."
            )
            raise RuntimeError("Model is not fitted.")

        k = max(1, min(int(k), len(self._reference_df)))

        for group_name, features in tqdm(
            self.feature_groups.items(),
            desc='Executing Batched Search',
            colour=_cfg.get('tqdm', 'colour'),
            ncols=_cfg.getint('tqdm', 'ncols'),
            bar_format=_cfg.get('tqdm', 'BarFormats'),
            unit='Group',
            mininterval=0.1
        ):
            misskey = [f for f in features if f not in data.columns]
            if misskey:
                logger.warning(
                    f"Query for group '{group_name}' is skipped. "
                    f"Missing keys: {misskey}"
                )
                BATCH[group_name] = pd.DataFrame()
                continue

            q_vectors = data[features].values.astype(np.float32)
            q_scaled = self.scalers[group_name].transform(q_vectors)
            q_scaled = np.nan_to_num(q_scaled, nan=0.0)

            if SCANN_AVAILABLE:
                quasi = (
                    self._l2_normalize(q_scaled)
                    if use_norm
                    else q_scaled.copy()
                )
                neighbors, distances = self.searchers[group_name].search_batched(
                    quasi,
                    final_num_neighbors=k
                )

            else:
                if use_norm:
                    logger.debug(
                        "GMM fallback uses standardized features internally; "
                        "use_norm is ignored to keep query/model space consistent."
                    )

                model = self.models[group_name]
                q_posteriors = model.predict_proba(q_scaled).astype(
                    np.float32, copy=False
                )
                q_embeddings = normalize(
                    q_posteriors, norm="l2", axis=1
                ).astype(np.float32, copy=False)

                item_embeddings = self._item_embeddings[group_name]

                n_queries = len(data)
                all_items = len(item_embeddings)
                Items = []

                # Chunk over queries so Q x N similarity matrices do not
                # consume excessive memory on CI or large input batches.
                query_chunk = int(
                    self.scann_config.get("gmm_query_chunk_size", 128)
                )

                for q_start in range(0, n_queries, query_chunk):
                    q_end = min(q_start + query_chunk, n_queries)
                    sims = q_embeddings[q_start:q_end] @ item_embeddings.T

                    for local_i in range(q_end - q_start):
                        row_sim = sims[local_i]
                        top_idx = np.argpartition(
                            -row_sim, k - 1
                        )[:k]
                        top_idx = top_idx[np.argsort(-row_sim[top_idx])]

                        row_dist = (
                            1.0 - row_sim[top_idx]
                        ).astype(np.float32, copy=False)

                        res_df = self._reference_df.iloc[top_idx].copy()
                        res_df['query_index'] = q_start + local_i
                        res_df['original_index'] = top_idx
                        res_df[
                            f'Quasi_Rating_{group_name}'
                        ] = row_dist
                        Items.append(res_df)

                if Items:
                    Tempdata = pd.concat(Items, ignore_index=True)
                    BATCH[group_name] = Tempdata
                    DataMerge.append(Tempdata)
                else:
                    BATCH[group_name] = pd.DataFrame()

                continue

            # ScaNN batch path
            Items = []
            for i in range(len(data)):
                row_neighbors = neighbors[i]
                row_distances = distances[i]
                res_df = self._reference_df.iloc[row_neighbors].copy()
                res_df['query_index'] = i
                res_df['original_index'] = row_neighbors
                res_df[
                    f'Quasi_Rating_{group_name}'
                ] = row_distances
                Items.append(res_df)

            if Items:
                Tempdata = pd.concat(Items, ignore_index=True)
                BATCH[group_name] = Tempdata
                DataMerge.append(Tempdata)
            else:
                BATCH[group_name] = pd.DataFrame()

        logger.debug(
            f'Batch search results generated for {len(BATCH)} groups.'
        )

        if not DataMerge:
            logger.warning(
                "No groups were successfully processed. "
                "Returning empty unified DataFrame."
            )
            return BATCH, pd.DataFrame()

        if aggweighted and not weights:
            logger.error(
                "Weights dictionary must be provided "
                "when aggregation_method is 'weighted'."
            )
            raise ValueError(
                "weights must be provided when aggweighted=True"
            )

        self.unidata = DFMerger(DataMerge)
        self.RateUnified(
            aggweighted=aggweighted,
            weights=weights,
            invert_score=invert_score,
            rating_range=rating_range
        )
        ASC = not invert_score
        self.unidata = self.unidata.sort_values(
            by=['final_rquasi'],
            ascending=[ASC]
        ).reset_index(drop=True)
        return BATCH, self.unidata

    def __call__(self, Data : pd.DataFrame = None) -> pd.DataFrame:
        if not self._is_fitted:
            if Data is None:
                logger.error('The parameter Data is None.')
                raise ValueError()
            else:
                assert not Data.empty, 'This is empty dataframe.'
                self.fit(Data)
        rrange      = _cfglist(_cfg, 'RATING', 'range')
        _, dataBach = self.search_batch(
                        data         = self._original, 
                        k            = _cfg.getint('RATING', 'theK'),
                        use_norm     = False,
                        aggweighted  = False,
                        weights      = dict(),
                        invert_score = _cfg.getboolean('RATING', 'invert'),
                        rating_range = rrange,
                       )
        SecondBach  = dataBach[['original_index', 'final_rquasi']].copy()
        Finaldata   = SecondBach.set_index('original_index').join(self._original)
        Finaldata   = Finaldata.rename_axis("index").sort_index()
        gc.collect()
        return Finaldata



if __name__ == '__main__':
    datapath     = LocDir.parent / 'data' / 'sampledata.parquet'
    DataSample   = pd.read_parquet(datapath)
    numeric_cols = ['SalesID', 'CustomerID', 'ProductPrice', 
                    'Quantity', 'Discount', 'TotalPrice', 
                    'CategoryID', 'VitalityDays', 'EmployeeID', 
                    'EmployeeAge', 'YearsWorking']
    for col in numeric_cols:
        DataSample[col] = pd.to_numeric(DataSample[col], errors = 'coerce')
    logger.info(f"Dataset loaded and expanded to {len(DataSample)} "
                 "rows for ScaNN compatibility.\n")
    Feats   = {"transaction_metrics"   : ["ProductPrice", "Quantity", 
                                          "Discount", "TotalPrice"],
               "product_traits"        : ["CategoryID", "VitalityDays"],
               "employee_demographics" : ["EmployeeAge", "YearsWorking"]}
    model   = QuasiRate_ScaNN(feature_groups = Feats)
    model.fit(DataSample)

    logger.info("\n--- Executing Single Search Query ---")
    query   = {"ProductPrice" : 50.0,
               "Quantity"     : 15,
               "Discount"     : 0.1,
               "TotalPrice"   : 700.0,
               "CategoryID"   : 4,
               "VitalityDays" : 60,
               "EmployeeAge"  : 50,
               "YearsWorking" : 12}
    Single, UnifiedTest  = model.search(
                           query_dict   = query,
                           k            = 7,
                           use_norm     = False,
                           aggweighted  = False, 
                           weights      = dict(), 
                           invert_score = False,
                           rating_range = (1.0, 5.0),
                           )
    for group_name, res_df in Single.items():
        logger.info(f"\nTop 3 matches for '{group_name}':")
        dis01 = ['ProductName', 'EmployeeFirstName', f'Quasi_Rating_{group_name}']
        logger.info(res_df[dis01].head(3))
    logger.info(UnifiedTest.head(5))

    logger.info("\n\n--- Executing Batched Search Queries ---")
    # Lazy Mode
    Finaldata = model()
    
    pd.set_option('display.max_columns', None)
    logger.info(Finaldata.sample(20))
    logger.info(Finaldata.isna().sum())
