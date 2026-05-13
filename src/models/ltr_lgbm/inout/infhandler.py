#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-13"


import sys
import joblib
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import logger, _cfg


@contextlib.contextmanager
def tqdm_joblib(tqdm_bar: tqdm):
    '''
    Context manager yang menghubungkan joblib's batch-completion callback
    ke sebuah tqdm progress bar. How to:
    with tqdm_joblib(tqdm(total=n, desc="Fallback")) as bar:
        results = Parallel(...)(delayed(fn)(...) for ...)
    '''
    class _TqdmBatchCallback:
        def __init__(self, bar: tqdm):
            self._bar = bar

        def __call__(self, out):
            self._bar.update(n=len(out) if hasattr(out, "__len__") else 1)

        def print_progress(self):
            pass
    try:
        old_cb = getattr(joblib.parallel, "_verbosity_filter", None)
        joblib.parallel.BatchCompletionCallBack = _TqdmBatchCallback
        yield tqdm_bar
    finally:
        if old_cb is not None:
            joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_bar.close()


#-------------------------------------------------------------------------------
# Dataclass: semua dependensi dari self
#-------------------------------------------------------------------------------
@dataclass
class _CustomerCfg:
    """
    Snapshot atribut AdaptiveFallbackRanker yang dibutuhkan oleh
    _process_customer.  Dikirim sebagai satu objek ke setiap worker
    sehingga semua field eksplisit dan picklable.
    """
    k                    : int
    catalog              : pd.DataFrame
    item_id_map          : pd.Series
    vector_index         : Optional[np.ndarray]
    item_vectors         : Optional[np.ndarray]
    strategy             : object
    cold_start_strategy  : object
    collaborative_scores : Dict[int, Dict[int, float]]
    popularity_scores    : Optional[pd.Series]
    max_candidates_scan  : Optional[int]
    random_state         : int
    mark_fallback        : bool
    item_id_col          : Optional[str]
    tqdm_colour          : str  = "green"
    tqdm_ncols           : int  = 100

    # ------------------------------------------------------------------ #
    @classmethod
    def from_ranker(cls, ranker: "AdaptiveFallbackRanker") -> "_CustomerCfg":
        """Buat _CustomerCfg dari instance AdaptiveFallbackRanker."""
        cfg = deepcopy(_cfg)
        return cls(
            k                    = ranker.k,
            catalog              = ranker._catalog,
            item_id_map          = ranker._item_id_map,
            vector_index         = ranker._vector_index,
            item_vectors         = ranker.item_vectors,
            strategy             = ranker._strategy,
            cold_start_strategy  = ranker._cold_start_strategy,
            collaborative_scores = ranker.collaborative_scores,
            popularity_scores    = ranker.popularity_scores,
            max_candidates_scan  = ranker.max_candidates_scan,
            random_state         = ranker.random_state,
            mark_fallback        = ranker.mark_fallback,
            item_id_col          = ranker.item_id_col,
            tqdm_colour          = cfg.get("tqdm", "colour"),
            tqdm_ncols           = cfg.getint("tqdm", "ncols"),
        )


#-------------------------------------------------------------------------------
# Fungsi "main standalone"
#-------------------------------------------------------------------------------
def Process_Customer(
    cust_id        : int,
    ranked         : pd.DataFrame,
    q_col          : str,
    score_col      : str,
    item_col       : str,
    fallback_score : float,
    cold_threshold : int,
    cfg            : _CustomerCfg,
) -> Optional[pd.DataFrame]:
    """
    Hitung fallback recommendations untuk satu customer.
    Fungsi ini berdiri sendiri di level modul (bukan metode) sehingga
    dapat di-pickle oleh joblib dengan backend 'loky' (true multiprocessing).
    Semua dependensi yang sebelumnya diakses via ``self`` dikirim melalui
    parameter ``cfg`` bertipe :class:`_CustomerCfg`.
    ________________________________________________________
    Argument
    cust_id        : ID customer yang sedang diproses.
    ranked         : DataFrame hasil LTR ranking seluruh customer.
    q_col          : Nama kolom query/customer ID.
    score_col      : Nama kolom skor.
    item_col       : Nama kolom item ID.
    fallback_score : Skor yang diberikan ke item fallback.
    cold_threshold : Batas jumlah item untuk cold-start.
    cfg            : Snapshot atribut ranker (lihat _CustomerCfg).
    """
    logger.debug("[%s] Memulai pemrosesan customer.", cust_id)

    # ── 1. Ekstrak data customer ─────────────────────────────────────────
    cust_df     = ranked[ranked[q_col] == cust_id]
    current_ids = cust_df[item_col].tolist()
    needed      = cfg.k - len(cust_df)
    if needed <= 0:
        logger.debug("[%s] Tidak perlu fallback (sudah >= k item).", cust_id)
        return None
    logger.debug("[%s] Membutuhkan %d item tambahan.", cust_id, needed)

    # ── 2. Tentukan strategi (cold-start atau normal) ────────────────────
    is_cold  = len(cust_df) <= cold_threshold
    strategy = cfg.cold_start_strategy if is_cold else cfg.strategy
    logger.debug("[%s] Menggunakan strategi '%s'.",
                 cust_id, "cold" if is_cold else "normal")

    # ── 3. Filter catalog yang tersedia ──────────────────────────────────
    catalog_ids       = cfg.item_id_map
    available_mask    = ~catalog_ids.isin(current_ids)
    available_catalog = cfg.catalog[available_mask].copy()
    if len(available_catalog) == 0:
        logger.debug("[%s] Tidak ada item catalog yang tersedia.", cust_id)
        return None
    logger.debug("[%s] %d item tersedia di catalog.",
    cust_id, len(available_catalog))

    # ── 4. User profile (content / hybrid saja) ──────────────────────────
    user_profile = None
    if (isinstance(strategy, (ContentBasedStrategy, HybridStrategy)) and \
        cfg.item_vectors is not None):
        existing_idx = catalog_ids[catalog_ids.isin(current_ids)].index
        if len(existing_idx) > 0:
            user_profile = cfg.item_vectors[existing_idx].mean(
                axis=0, keepdims = True)
            logger.debug("[%s] User profile shape: %s.",
                         cust_id, user_profile.shape)

    # ── 5. Siapkan candidate vectors ──────────────────────────────────────
    candidate_vecs = None
    if cfg.item_vectors is not None and len(available_catalog) > 0:
        vec_positions: List[int] = list()
        for idx in tqdm(
            available_catalog.index,
            desc        = f"[{cust_id}] Vector lookup",
            colour      = cfg.tqdm_colour,
            ncols       = cfg.tqdm_ncols,
            unit        = "item",
            mininterval = 0.1,
            leave       = False):
            pos = np.where(cfg.vector_index == idx)[0]
            if len(pos) > 0:
                vec_positions.append(pos[0])
        if vec_positions:
            candidate_vecs = cfg.item_vectors[vec_positions]

            # Sampling jika kandidat terlalu banyak
            if (cfg.max_candidates_scan and \
                len(candidate_vecs) > cfg.max_candidates_scan):
                logger.debug("[%s] Sampling %d dari %d kandidat.",
                    cust_id, cfg.max_candidates_scan, len(candidate_vecs))
                rng               = np.random.RandomState(cfg.random_state 
                                    + hash(str(cust_id)) % 10_000)
                sampled_idx       = rng.choice(len(candidate_vecs), 
                                    cfg.max_candidates_scan, replace = False)
                candidate_vecs    = candidate_vecs[sampled_idx]
                available_catalog = available_catalog.iloc[sampled_idx]

    # ── 6. Collaborative scores ───────────────────────────────────────────
    coll_dict: Dict[int, float] = dict()
    if isinstance(strategy, (CollaborativeStrategy, HybridStrategy)) or (
        is_cold and isinstance(cfg.cold_start_strategy, 
        (CollaborativeStrategy, HybridStrategy))):
        coll_matrix = cfg.collaborative_scores
        combined: Dict[int, float] = defaultdict(float)
        for ci in tqdm(current_ids,
            desc        = f"[{cust_id}] Collab scores",
            colour      = cfg.tqdm_colour,
            ncols       = cfg.tqdm_ncols,
            unit        = "item",
            mininterval = 0.1,
            leave       = False):
            if ci in coll_matrix:
                for item_j, score in coll_matrix[ci].items():
                    combined[item_j] += score
        coll_dict = dict(combined)
        logger.debug("[%s] %d collaborative neighbors ditemukan.", 
                      cust_id, len(coll_dict))

    # ── 7. Buat context dan jalankan strategi ────────────────────────────
    context = FallbackContext(
        candidate_items      = available_catalog,
        candidate_vectors    = candidate_vecs,
        current_item_ids     = current_ids,
        user_profile         = user_profile,
        popularity_scores    = cfg.popularity_scores,
        collaborative_scores = coll_dict,
        top_k                = needed,
        random_state         = cfg.random_state + hash(str(cust_id)) % 10_000)
    try:
        selected = strategy.select_items(context)
        logger.debug("[%s] Strategi memilih %d item.", cust_id, len(selected))
    except Exception as exc:
        logger.debug("[%s] Strategi gagal (%s), fallback ke random.", 
                      cust_id, exc)
        rng      = np.random.RandomState(context.random_state)
        selected = available_catalog.sample(
            min(needed, len(available_catalog)), random_state = rng)

    # ── 8. Finalisasi output ──────────────────────────────────────────────
    selected            = selected.head(needed).copy()
    selected[q_col]     = cust_id
    selected[score_col] = fallback_score
    selected[item_col]  = (catalog_ids[selected.index].values if \
                          cfg.item_id_col else selected.index)
    if cfg.mark_fallback:
        selected["is_fallback"] = True
    logger.debug("[%s] Selesai: %d item fallback ditambahkan.", cust_id, len(selected))
    return selected

if __name__ == '__main__':
    pass