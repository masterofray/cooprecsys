#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-11"


"""
adaptive_fallback.py
__________________________
Professional‑grade adaptive fallback ranker with DuckDB acceleration,
joblib parallel processing, and full‑featured fallback strategies.
"""

import hashlib
import numpy as np
import pandas as pd
import cloudpickle as cp
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import combinations
from joblib import Parallel, delayed
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from typing import (Optional, List, Dict, Any, Callable, 
                    Union, Tuple)

from .infhandler import Process_Customer
from .infcore    import LTRModelInference
from .infsupport import (Joblibar, _CustomerCfg, ANNIndex, 
                         ContentBasedStrategy, PopularityStrategy, 
                         HybridStrategy, CollaborativeStrategy)
from ....db         import DuckDBManager
from ....configs    import logger, _cfg, FallbackConfig

LocDir = Path(__file__).resolve()

# ---------------------------------------------------------------------------
# Main Ranker with DuckDB & joblib
# ---------------------------------------------------------------------------
class AdaptiveFallbackRanker:
    """
    Ensures every query group gets exactly ``top_k`` ranked items.
    engine      : LTRModelInference, Initialized inference engine with data already loaded.
    config      : FallbackConfig is  Configuration instance. 
                  If None, auto-loads from _cfg (configuration.ini).
    ab_callback : Callable, A/B testing callback function (cannot be stored in INI).
    """
    def __init__(self,
            engine      : LTRModelInference,
            config      : Optional[FallbackConfig] = None,
            ab_callback : Optional[Callable[[pd.DataFrame], None]] = None,
       ) -> None:
        if engine is None:
            raise DataNotLoadedError("Engine cannot be None.")
        if engine.data is None or engine.data.empty:
            raise DataNotLoadedError(
            "Engine.data must be loaded before initializing Fallback.")
        self.engine = engine
        
        if config is None:
            try:
                self.config = FallbackConfig.from_configparser(_cfg)
                logger.debug("FallbackConfig loaded from _cfg internal")
            except Exception as Arc:
                logger.warning(
                    f"Could not load FallbackConfig from _cfg: {Arc}."
                    "Falling back to defaults.")
                self.config = FallbackConfig()
        else:
            self.config = config
        self.config.validate()
        self._k          = self.config.top_k if self.config.top_k is not None else engine.top_k
        self.ab_callback = ab_callback
        self.cache_dir   = LocDir.parents[4] / _cfg.get('PATHS', 'output_dir')
        self.cache_dir.mkdir(parents = True, exist_ok = True)
        
        # Lazy Inisialization
        self.score_mode    = _cfg.get('INFERENCE', 'scoremode')
        self._db           : Optional[DuckDBManager] = None
        self._catalog      : Optional[pd.DataFrame]  = None
        self._item_vectors : Optional[np.ndarray]    = None
        self._vector_index : Optional[pd.Index]      = None
        self._item_id_map  : Optional[pd.Series]     = None
        self._ann_index    : Optional[ANNIndex]      = None
        self._ranked_df    : Optional[pd.DataFrame]  = None
        self._popularity_scores    : Optional[pd.Series] = None
        self._collaborative_scores : Optional[Dict[Any, Dict[Any, float]]] = None
        if self.config.use_duckdb and DuckDBManager is not None:
            self._init_duckdb()
        logger.info(
        f"AdaptiveFallbackRanker initialized: "
        f"strategy = {self.config.strategy}, "
        f"top_k    = {self._k}, "
        f"duckdb   = {self.config.use_duckdb and DuckDBManager is not None}, "
        f"n_jobs   = {self.config.n_jobs}")

    
    def __getattr__(self, name: str):
        """
        Delegate attribute access to self.config if the attribute is not found
        in the instance. This allows transparent use of `self.max_candidates_scan`,
        `self.cold_start_threshold`, etc. without manual copy.
        """
        if name == "config" or name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        if "config" in self.__dict__ and hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
            f"Also not found in config. Available config attributes: {dir(self.config)}")

    
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def k(self) -> int:
        """Target number of items per customer."""
        return self._k
    
    @property
    def strategy(self):
        """Resolved fallback strategy (lazy)."""
        if not hasattr(self, '_strategy'):
            self._strategy = self._init_strategy(self.config.strategy)
        return self._strategy
    
    @property
    def cold_start_strategy(self):
        """Resolved cold-start strategy (lazy)."""
        if not hasattr(self, '_cold_start_strategy'):
            self._cold_start_strategy = self._init_strategy(self.config.cold_start_strategy)
        return self._cold_start_strategy

    @property
    def query_id_col(self) -> str:
        return self.engine.query_id_col

    @property
    def score_col(self) -> str:
        return self.engine.score_col

    @property
    def rank_col(self) -> str:
        return self.engine.rank_col

    @property
    def catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            self._build_catalog()
        return self._catalog

    @property
    def item_vectors(self) -> Optional[np.ndarray]:
        if self._item_vectors is None and \
            isinstance(self._strategy, (ContentBasedStrategy, HybridStrategy)):
            self._build_item_vectors()
        return self._item_vectors

    @property
    def popularity_scores(self) -> pd.Series:
        if self._popularity_scores is None:
            self._compute_popularity()
        return self._popularity_scores

    @property
    def collaborative_scores(self) -> Dict[Any, Dict[Any, float]]:
        if self._collaborative_scores is None:
            self._compute_collaborative_scores()
        return self._collaborative_scores

    @property
    def ranked_df(self) -> Optional[pd.DataFrame]:
        return self._ranked_df

    @property
    def item_id_map(self) -> pd.Series:
        '''Mapping posisi baris di catalog, 
        maka nilai item_id (series dengan index integer).'''
        if self._item_id_map is None:
            if self._catalog is None:
                self._build_catalog()
            id_col = self.item_id_col or 'index'
            if id_col in self._catalog.columns:
                self._item_id_map = self._catalog[id_col].reset_index(drop=True)
            else:
                self._item_id_map = self._catalog.index.to_series(
                                    ).reset_index(drop=True)
            logger.debug("item_id_map dibangun, panjang = %d", 
                         len(self._item_id_map))
        return self._item_id_map

    @property
    def vector_index(self) -> Optional[np.ndarray]:
        '''Indeks ANN yang sudah dibangun (Faiss/NMSLIB), 
        hanya untuk content/hybrid'''
        if self._vector_index is None and isinstance(
        self._strategy, (ContentBasedStrategy, HybridStrategy)):
            _ = self.item_vectors
        return self._vector_index

    # ------------------------------------------------------------------
    # DuckDB initialization
    # ------------------------------------------------------------------
    def _init_duckdb(self):
        logger.debug("Initializing DuckDB in-memory connection.")
        self._db = DuckDBManager(':memory:', read_only = False)
        data     = self.engine.data
        try:
            self._db.register_dataframe('candidates', data)
        except Exception:
            obj_cols = list(data.select_dtypes(include = ['object', 'string']).columns)
            df_safe  = deepcopy(data)
            for c in obj_cols:
                df_safe[c] = df_safe[c].astype(str)
            self._db.register_dataframe('candidates', df_safe)

        # Turn the view into a real table so we can ALTER it
        self._db.execute("CREATE OR REPLACE TABLE candidates_tmp AS SELECT * FROM candidates")
        self._db.execute("DROP VIEW IF EXISTS candidates")
        self._db.execute("ALTER TABLE candidates_tmp RENAME TO candidates")

        item_col = self.item_id_col if self.item_id_col else 'index'
        if item_col not in data.columns:
            self._db.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS __idx__ INTEGER")
            self._db.execute("UPDATE candidates SET __idx__ = rowid")
            item_col = '__idx__'
        self._item_col_db = item_col
        logger.debug(f"DuckDB initialized with {len(data)} rows, item column: {item_col}")


    # ------------------------------------------------------------------
    # Catalog building (DuckDB accelerated)
    # ------------------------------------------------------------------
    def _build_catalog(self):
        logger.debug("Building item catalog.")
        if self.use_duckdb and self._db:
            q = f"SELECT DISTINCT * FROM candidates"
            self._catalog = self._db.query(q)
            if self.item_id_col:
                self._item_id_map = self._catalog[self.item_id_col]
            else:
                self._item_id_map = pd.Series(self._catalog.index, index = self._catalog.index)
        else:
            df = self.engine.data
            if self.item_id_col and self.item_id_col in df.columns:
                self._catalog     = df.drop_duplicates(subset=[self.item_id_col]).copy()
                self._item_id_map = self._catalog[self.item_id_col]
            else:
                self._catalog     = df[~df.index.duplicated(keep='first')].copy()
                self._item_id_map = pd.Series(self._catalog.index, index = self._catalog.index)
        logger.debug(f"Catalog: {len(self._catalog)} unique items.")


    # ------------------------------------------------------------------
    # Item vectors (with caching)
    # ------------------------------------------------------------------
    def _build_item_vectors(self):
        feature_cols = self.engine._feature
        catalog = self.catalog
        if self.cache_dir and self.cache_key:
            cache_path = self.cache_dir / f"item_vectors_{self.cache_key}.npy"
            meta_path  = self.cache_dir / f"item_vectors_{self.cache_key}_meta.cloudpickle"
            if cache_path.exists() and meta_path.exists():
                try:
                    self._item_vectors = np.load(cache_path)
                    with open(meta_path, 'rb') as f:
                        meta = cp.load(f)
                    self._vector_index = pd.Index(meta['index'])
                    logger.info(f"Loaded item vectors from cache: {cache_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

        cat_matrices = list()
        num_matrices = list()
        for col in tqdm(feature_cols,
                        desc   = 'Item Feature Column',
                        colour = _cfg.get('tqdm', 'colour'),
                        ncols  = _cfg.getint('tqdm', 'ncols'),
                        unit   = 'Column',
                        mininterval = 0.1):
            if col not in catalog.columns:
                continue
            if col in self.engine.encman.encoders:
                try:
                    encoder = self.engine.encman.encoders[col]
                    encoded = encoder.transform(catalog[col].astype(str))
                    cat_matrices.append(encoded.reshape(-1, 1))
                except Exception as e:
                    logger.debug(f"Skip col '{col}': {e}")
            else:
                try:
                    vals = pd.to_numeric(catalog[col], errors='coerce').fillna(0).values
                    num_matrices.append(vals.reshape(-1, 1))
                except Exception as e:
                    logger.debug(f"Skip numeric col '{col}': {e}")

        cat_arr = np.hstack(cat_matrices).astype(np.float32
                  ) if cat_matrices else np.empty((len(catalog), 0), dtype = np.float32)
        num_arr = np.hstack(num_matrices).astype(np.float32
                  ) if num_matrices else np.empty((len(catalog), 0), dtype = np.float32)
        if num_arr.shape[1] > 0:
            num_arr = StandardScaler().fit_transform(num_arr)
        full = np.hstack([cat_arr, num_arr])
        if full.shape[1] == 0:
            logger.warning("No usable features for content vectors.\n"
            "Falling back to popularity.")
            self._strategy = PopularityStrategy()
            self._item_vectors = None
            return None
        elif 1 < full.shape[1] < 6:
            self._strategy = CollaborativeStrategy()

        self._item_vectors = full
        self._vector_index = catalog.index

        # Initialized Cache
        if self.cache_dir and self.cache_key:
            np.save(cache_path, self._item_vectors)
            with open(meta_path, 'wb') as metafile:
                cp.dump({'index': list(catalog.index)}, metafile)
            logger.info(f"Item vectors cached to {cache_path}.")

        # ANN models
        if self.use_ann and len(self._item_vectors) > self.ann_threshold:
            self._ann_index = ANNIndex(self._item_vectors,
                                       use_gpu     = self.use_gpu,
                                       force_brute = not _FAISS_AVAILABLE)


    # ------------------------------------------------------------------
    # Popularity & Collaborative via DuckDB
    # ------------------------------------------------------------------
    def _compute_popularity(self):
        if self.use_duckdb and self._db:
            logger.debug("Calculating popularity via DuckDB.")
            item_col = self._item_col_db
            item_que = f'''
                SELECT 
                    {item_col} as item, 
                    COUNT(*) as freq 
                FROM 
                    candidates 
                GROUP BY
                    1 ORDER BY freq DESC'''
            freq_df = self._db.query(item_que)
            freq_df = freq_df.set_index('item')['freq']
            freq_df = freq_df / freq_df.max()   # minmax normalized
            self._popularity_scores = freq_df
        else:
            if self.item_id_col:
                counts = self.engine.data[self.item_id_col].value_counts()
            else:
                counts = self.engine.data.index.value_counts()
            self._popularity_scores = counts / counts.max()
        logger.debug("Popularity scores computed.")
        logger.info(f'Popularity Score as {type(self._popularity_scores)}.\n')


    def _compute_collaborative_scores(self):
        if self.use_duckdb and self._db:
            logger.debug("Calculating collaborative scores via DuckDB.")
            item_col = self._item_col_db
            user_col = self.query_id_col
            # Co‑occurrence pairs
            co_query = f"""
                CREATE TEMP TABLE co_pairs AS
                SELECT
                    a.{item_col} as item1, 
                    b.{item_col} as item2, 
                    COUNT(*) as co
                FROM 
                    candidates AS a
                JOIN 
                    candidates AS b 
                ON 
                    a.{user_col} = b.{user_col} AND 
                    a.{item_col} < b.{item_col}
                GROUP BY 
                    1,2
                """
            self._db.execute(co_query)
            # Frekuensi item
            freq_query = f"""
                CREATE TEMP TABLE item_freq AS
                SELECT {item_col} as item, COUNT(*) as freq
                FROM candidates
                GROUP BY 1
            """
            self._db.execute(freq_query)

            # Jaccard score
            jaccard_query = """
                SELECT
                    c.item1, 
                    c.item2,
                    CAST(c.co AS DOUBLE) / (f1.freq + f2.freq - c.co) as jaccard
                FROM
                    co_pairs AS c
                JOIN 
                    item_freq f1 ON c.item1 = f1.item
                JOIN 
                    item_freq f2 ON c.item2 = f2.item
            """
            jaccard_df = self._db.query(jaccard_query)

            # Bangun dictionary
            collab = defaultdict(dict)
            for _, row in jaccard_df.iterrows():
                i1, i2, sim    = row['item1'], row['item2'], row['jaccard']
                collab[i1][i2] = sim
                collab[i2][i1] = sim
            self._collaborative_scores = dict(collab)
            logger.debug(f"Collaborative matrix built with {len(collab)} items.")

        # fallback pandas (heavy)
        else:
            logger.debug("Calculating collaborative scores via pandas.")
            df         = self.engine.data
            user_col   = self.query_id_col
            item_col   = self.item_id_col if self.item_id_col else df.index.name
            item_freq  = df[item_col].value_counts().to_dict()
            colour     = _cfg.get('tqdm', 'colour')
            ncols      = _cfg.getint('tqdm', 'ncols')
            bar        = _cfg.get('tqdm', 'BarFormats')
            user_items = df.groupby(user_col)[item_col].apply(lambda x: x.unique())

            # Hitung semua pasangan dengan Counter
            edge_counter = Counter()
            for items in tqdm(user_items,
                              desc   = 'User Item Combinations',
                              colour = colour,
                              ncols  = ncols,
                              unit   = 'User',
                              bar_format  = bar,
                              mininterval = 0.1):
                for i, j in combinations(items, 2):
                    edge_counter[(i, j)] += 1
                    edge_counter[(j, i)] += 1
            co_counts = defaultdict(lambda: defaultdict(int))
            for (i, j), co in edge_counter.items():
                co_counts[i][j] = co

            # Hitung similarity
            final = dict()
            for i1 in tqdm(co_counts.keys(),
                           desc   = 'Compute Similarities',
                           colour = colour,
                           ncols  = ncols,
                           unit   = 'Item',
                           bar_format  = bar,
                           mininterval = 0.1):
                total_i  = item_freq.get(i1, 1)
                sim_dict = dict()
                related  = co_counts[i1]
                for i2, co in related.items():
                    total_j      = item_freq.get(i2, 1)
                    sim_dict[i2] = co / (total_i + total_j - co + 1e-7)
                final[i1]        = sim_dict

            self._collaborative_scores = final
            logger.debug("Collaborative scores (pandas) done.")


    # ------------------------------------------------------------------
    # Fallback score
    # ------------------------------------------------------------------
    def _compute_fallback_score(self, 
        all_scores: np.ndarray) -> float:
        if self.score_mode == 'min':
            fallthemback = np.min(all_scores) - 1e-5 if \
                           len(all_scores) > 0 else 0.0
            return fallthemback
        elif self.score_mode == 'quantile':
            fallmeback   = np.quantile(all_scores, self.fallback_score_quantile
                           ) if len(all_scores) > 0 else 0.0
            return fallmeback
        else:
            return self.fallback_score_value


    # ------------------------------------------------------------------
    # Main ranking method
    # ------------------------------------------------------------------
    def _trigger_vector(self) -> None:
        try:
            self._build_catalog()
            self._build_item_vectors()
            _ = self.catalog
            _ = self.item_vectors
            _ = self.item_id_map
            _ = self.vector_index
            logger.debug('The Trigger of itemsmaps and items vector already done.')
        except Exception as Arc:
            logger.error('The Trigger was Fail, the program will be error.')
    
    def rank_with_fallback(self) -> pd.DataFrame:
        """
        Jalankan LTR ranking kemudian isi kekurangan item setiap customer
        dengan strategi fallback.  Pemrosesan per-customer dilakukan secara
        paralel menggunakan joblib, dengan progress-bar tqdm.auto.
        """
        logger.debug("Memulai rank_with_fallback.")

        #--- 1. LTR ranking ---------------------------------------------
        try:
            ranked = self.engine.rank_top_k(top_k = self.k)
        except Exception as exc:
            raise RuntimeError(f"LTR inference gagal: {exc}") from exc
        if ranked.empty:
            return ranked
        q_col     = self.query_id_col
        score_col = self.score_col
        rank_col  = self.rank_col
        item_col  = self.item_id_col or "__fallback_item_id__"
        if self.item_id_col is None:
            ranked[item_col] = ranked.index

        #--- 2. Identifikasi customer yang kekurangan item --------------
        counts         = ranked.groupby(q_col).size()
        deficient_list = counts[counts < self.k].index.tolist()
        if not deficient_list:
            logger.info("Semua customer sudah memiliki item yang cukup.")
            return self._final_rerank(
                ranked, item_col if self.item_id_col is None else None)
        logger.info("Fallback dibutuhkan untuk %d customer.", len(deficient_list))
        all_scores     = ranked[score_col].values
        fallback_score = self._compute_fallback_score(all_scores)

        #--- 3. Bangun _CustomerCfg sekali, dikirim ke semua worker -----
        #------ Ini menghindari serialisasi self yang tidak picklable.
        self._trigger_vector()
        cust_cfg       = _CustomerCfg.from_ranker(self)
        shared_kwargs  = dict(
            ranked         = ranked,
            q_col          = q_col,
            score_col      = score_col,
            item_col       = item_col,
            fallback_score = fallback_score,
            cold_threshold = self.cold_start_threshold,
            cfg            = cust_cfg)

        #--- 4. Eksekusi paralel / serial -------------------------------
        tqdm_bar = tqdm(total       = len(deficient_list),
                        desc        = "Fallback Process",
                        colour      = _cfg.get("tqdm", "colour"),
                        ncols       = _cfg.getint("tqdm", "ncols"),
                        bar_format  = _cfg.get('tqdm', 'BarFormats'),
                        unit        = "customer",
                        dynamic_ncols = False,
                        leave         = True,
                        position      = 0,
                        mininterval   = 0.4)
        # For debugging
        # fallback_parts: List[pd.DataFrame] = list()
        # for cust in deficient_list:
            # res = Process_Customer(cust_id = cust, **shared_kwargs)
            # if res is not None:
                # fallback_parts.append(res)
            # tqdm_bar.update(1)
        # tqdm_bar.close()
        
        #Original
        if self.n_jobs == 1 or len(deficient_list) <= 1:
            fallback_parts: List[pd.DataFrame] = list()
            for cust in deficient_list:
                res = Process_Customer(cust_id = cust, **shared_kwargs)
                if res is not None:
                    fallback_parts.append(res)
                tqdm_bar.update(1)
            tqdm_bar.close()
        
        else:
            # Paralel - Joblibar context manager meng-hook progress bar
            # ke joblib's BatchCompletionCallBack sehingga bar ter-update
            # setiap batch selesai (kompatibel dengan backend loky/threading).
            logger.debug(
                "Menggunakan joblib: n_jobs = %d, backend = '%s', batch_size= %s",
                self.n_jobs,
                getattr(self, "parallel_backend", "loky"),
                self.batch_size)
            with Joblibar(tqdm_bar):
                results     = Parallel(
                    n_jobs  = self.n_jobs,
                    backend = getattr(self, "parallel_backend", "loky"),
                    batch_size = self.batch_size,
                )(  delayed(Process_Customer)(cust_id = cust, **shared_kwargs)
                    for cust in deficient_list)
            fallback_parts  = [r for r in results if r is not None]

        #--- 5. Gabungkan hasil fallback ke ranked --------------------
        temp_drop = item_col if self.item_id_col is None else None
        if fallback_parts:
            fallback_df = pd.concat(fallback_parts, ignore_index = True)
            if temp_drop:
                ranked.drop(columns = [temp_drop],
                    errors = "ignore", inplace = True)
                fallback_df.drop(columns = [temp_drop], 
                    errors = "ignore", inplace = True)

            # Samakan kolom antara ranked dan fallback_df
            for col in ranked.columns:
                if col not in fallback_df.columns:
                    fallback_df[col] = np.nan
            fallback_df = fallback_df[ranked.columns]

            if self.mark_fallback and "is_fallback" not in ranked.columns:
                ranked["is_fallback"] = False
            ranked = pd.concat([ranked, fallback_df], ignore_index = True)
        else:
            if temp_drop:
                ranked.drop(columns = [temp_drop], 
                errors = "ignore", inplace = True)

        #--- 6. Final re-rank -----------------------------------------
        final           = self._final_rerank(ranked)
        self._ranked_df = final
        if self.ab_callback:
            self.ab_callback(final)
        logger.info("Ranking selesai, shape: %s", final.shape)
        return final

    def _final_rerank(self, df: pd.DataFrame) -> pd.DataFrame:
        q_col        = self.query_id_col
        score_col    = self.score_col
        rank_col     = self.rank_col
        df[rank_col] = df.groupby(q_col)[score_col].rank(
                       method = 'first', ascending = False).astype(int)
        return df.sort_values([q_col, rank_col]).reset_index(drop = True)


    # ------------------------------------------------------------------
    # Save / Convenience
    # ------------------------------------------------------------------
    def save_rankings(self, 
                      output_path : str, 
                      as_parquet  : bool = True,
                     ) -> str:
        if self._ranked_df is None:
            raise RuntimeError("Call rank_with_fallback() first.")
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents = True, exist_ok = True)
        if as_parquet:
            self._ranked_df.to_parquet(output_path, index = False, 
            engine = 'pyarrow', compression = 'gzip')
        else:
            self._ranked_df.to_csv(output_path, index = False)
        logger.info(f"Saved rankings to {output_path}")
        return output_path

    def __call__(self) -> pd.DataFrame:
        return self.rank_with_fallback()

    def __repr__(self) -> str:
        return f"AdaptiveFallbackRanker(strategy = {self._strategy.__class__.__name__}, top_k={self.k})"


if __name__ == '__main__':
    pass