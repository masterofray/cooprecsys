#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-25"


import gc
import sys
import numpy as np
import scipy.sparse as sp
from   tqdm.auto import tqdm
from   pathlib   import Path
from   typing    import Optional

LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import _cfg, logger


def _get_ranks(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix],
        user_features       : Optional[sp.spmatrix],
        item_features       : Optional[sp.spmatrix],
        num_threads         : int,
        check_intersections : bool,
        step_label          : str,
    ) -> sp.csr_matrix:
    """Shared rank-computation helper used by all metric functions."""
    logger.debug("Check the parameter: metric = %s num_threads = %d",
                  step_label, num_threads)
    ranks = model.predict_rank(
            test_interactions   = test_interactions,
            train_interactions  = train_interactions,
            user_features       = user_features,
            item_features       = item_features,
            num_threads         = num_threads,
            check_intersections = check_intersections)
    return ranks


def MRR_rank(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        item_features       : Optional[sp.spmatrix] = None,
        num_threads         : int = 4,
        check_intersections : bool = True,
        preserve_rows       : bool = False,
    ) -> np.ndarray:
    """Mean Reciprocal Rank — 1 / rank of the highest-ranked true positive.
       A perfect model scores 1.0 (true positive is rank-1).
       Users with no positives score 0.0.
       The primary objective of MRR is to evaluate how quickly 
       a recommendation or search system places the first 
       relevant item within the ranked results.
       """
    assert num_threads >= 1, "num_threads must be >= 1"
    logger.debug("MRR rank metric goes now: num_threads = %d", num_threads)
    ranks = _get_ranks(model               = model,
                       test_interactions   = test_interactions,
                       train_interactions  = train_interactions,
                       user_features       = user_features,
                       item_features       = item_features,
                       num_threads         = num_threads,
                       check_intersections = check_intersections,
                       step_label          = "MRR")
    with tqdm(total       = 2, 
              desc        = "Reciprocal Rank",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:
        pbar.set_postfix_str("converting ranks to reciprocals")
        ranks.data = 1.0 / (ranks.data + 1.0)
        pbar.update(1)

        pbar.set_postfix_str("taking per-user max")
        rr = np.squeeze(np.array(ranks.max(axis=1).todense()))
        if not preserve_rows:
            rr = rr[test_interactions.getnnz(axis=1) > 0]
        pbar.update(1)
    gc.collect()
    logger.debug("reciprocal_rank: mean = %.4f", float(rr.mean()))
    return rr.astype(np.float32)


def NDCG_rank(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        item_features       : Optional[sp.spmatrix] = None,
        k                   : int = 10,
        num_threads         : int = 4,
        check_intersections : bool = True,
        preserve_rows       : bool = False,
    ) -> np.ndarray:
    """Normalized Discounted Cumulative Gain at k (NDCG@k)
       for binary relevance. A perfect model will achieve 
       a score of 1.0 (all positive test items appear at 
       the top of the ranking). A user with no positive 
       items in test_interactions will obtain a score of 0.0.
    """
    assert num_threads >= 1, "num_threads must be >= 1"
    assert k >= 1, "k must be >= 1"
    logger.debug("Check the parameter: k=%d, num_threads=%d", k, num_threads)

    # 1. Ambil rank prediksi untuk item true positive di test set
    ranks = _get_ranks(model               = model,
                       test_interactions   = test_interactions,
                       train_interactions  = train_interactions,
                       user_features       = user_features,
                       item_features       = item_features,
                       num_threads         = num_threads,
                       check_intersections = check_intersections,
                       step_label          = f"NDCG@{k}")

    with tqdm(total       = 5, 
              desc        = f"NDCG@{k}",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:

        pbar.set_postfix_str("calculating DCG factors")
        in_top_k              = ranks.data < k
        dcg_contrib           = np.zeros_like(ranks.data, dtype=np.float32)
        dcg_contrib[in_top_k] = 1.0 / np.log2(ranks.data[in_top_k] + 2.0)
        ranks.data            = dcg_contrib
        dcg                   = np.squeeze(np.array(ranks.sum(axis=1).todense()))
        pbar.update(1)

        #Normalisasi
        pbar.set_postfix_str("normalizing")
        actual_positives = np.squeeze(np.array(test_interactions.getnnz(axis=1)))
        idcg_table       = np.zeros(k + 1, dtype=np.float32)
        pbar.update(1)
        
        #Hitung IDCG@k per user
        pbar.set_postfix_str("calculating IDCG")
        for j in range(k):
            idcg_table[j + 1] = idcg_table[j] + (1.0 / np.log2(j + 2.0))
        idcg_indices = np.minimum(actual_positives, k)
        idcg         = idcg_table[idcg_indices]
        pbar.update(1)
        
        ndcg = np.zeros_like(dcg, dtype=np.float32)
        valid_mask = idcg > 0
        ndcg[valid_mask] = dcg[valid_mask] / idcg[valid_mask]
        pbar.update(1)
        
        if not preserve_rows:
            ndcg = ndcg[actual_positives > 0]
        pbar.update(1)
    gc.collect()
    logger.debug("mean = %.4f", float(ndcg.mean()))
    return ndcg.astype(np.float32)

if __name__ == '__main__':
    pass