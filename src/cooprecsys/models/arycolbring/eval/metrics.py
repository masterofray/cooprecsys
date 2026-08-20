#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-01"

import gc
import numpy as np
import scipy.sparse as sp
from   tqdm.auto  import tqdm
from   typing     import Optional
from   .take_rank import _get_ranks
from cooprecsys.models.arycolbring.CLproximity import CSRMatrix, calculate_auc_from_rank
from ....configs import _cfg, logger


def precision_at_k(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        item_features       : Optional[sp.spmatrix] = None,
        k                   : int  = 10,
        num_threads         : int  = 1,
        preserve_rows       : bool = False,
        check_intersections : bool = True,
        ) -> np.ndarray:
    """Precision@k — fraction of the top-k recommendations that are true positives.
       _______________________________________________________
       model               : fitted AryColBring instance
       test_interactions   : sparse [n_users × n_items]  — ground-truth positives
       train_interactions  : optional sparse — known positives to exclude from ranking
       k                   : cut-off rank
       user_features       : optional CSR [n_users × n_user_features]
       item_features       : optional CSR [n_items × n_item_features]
       preserve_rows       : if False, drop users with no test interactions
       num_threads         : OpenMP thread count
       check_intersections : raise if test/train overlap
       _______________________________________________________
       Returns is np.ndarray float64, shape (n_active_users,) or (n_users,)
                  Precision@k score for each user (0.0 – 1.0).
    """
    assert num_threads >= 1, "num_threads must be >= 1"
    assert k >= 1, "num_threads must be >= 1"
    logger.info("precision_at_k: k = %d num_threads= %d", k, num_threads)
    ranks = _get_ranks(model               = model,
                       test_interactions   = test_interactions,
                       train_interactions  = train_interactions,
                       user_features       = user_features,
                       item_features       = item_features,
                       num_threads         = num_threads,
                       check_intersections = check_intersections,
                       step_label          = f"P@{k}")
    with tqdm(total       = 2, 
              desc        = f"Precision@{k}",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:
        pbar.set_postfix_str("masking top-k")
        ranks.data = np.less(ranks.data, k, ranks.data)
        precision  = np.squeeze(np.array(ranks.sum(axis=1))) / k
        pbar.update(1)

        pbar.set_postfix_str("filtering active users")
        if not preserve_rows:
            active    = test_interactions.getnnz(axis=1) > 0
            precision = precision[active]
        pbar.update(1)

    gc.collect()
    logger.debug("precision_at_k: mean = %.4f", float(precision.mean()))
    return precision.astype(np.float32)


def recall_at_k(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        item_features       : Optional[sp.spmatrix] = None,
        k                   : int  = 10,
        num_threads         : int  = 1,
        preserve_rows       : bool = False,
        check_intersections : bool = True,
        ) -> np.ndarray:
    """Recall@k - fraction of true positives recovered in the top-k results.
       A perfect score is 1.0. The return is np.ndarray float64 with shape 
       (n_active_users,) or (n_users,) for Recall@k per user.
       """
    assert num_threads >= 1, "num_threads must be >= 1"
    assert k >= 1, "num_threads must be >= 1"
    logger.info("recall_at_k: k = %d num_threads = %d", k, num_threads)
    ranks = _get_ranks(model               = model,
                       test_interactions   = test_interactions,
                       train_interactions  = train_interactions,
                       user_features       = user_features,
                       item_features       = item_features,
                       num_threads         = num_threads,
                       check_intersections = check_intersections,
                       step_label          = f"Recall@{k}")
    with tqdm(total       = 3, 
              desc        = f"Recall@{k}",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:
        pbar.set_postfix_str("masking top-k")
        ranks.data = np.less(ranks.data, k, ranks.data)
        retrieved  = np.squeeze(test_interactions.getnnz(axis=1)).astype(np.float32)
        hit        = np.squeeze(np.array(ranks.sum(axis=1), dtype=np.float32))
        pbar.update(1)

        pbar.set_postfix_str("filtering active users")
        if not preserve_rows:
            active    = test_interactions.getnnz(axis=1) > 0
            hit       = hit[active]
            retrieved = retrieved[active]
        pbar.update(1)

        # Guard against division by zero for users with 0 test positives
        with np.errstate(divide = "ignore", invalid = "ignore"):
            recall = np.where(retrieved > 0, hit / retrieved, 0.0)
        pbar.update(1)

    gc.collect()
    logger.debug("recall_at_k: mean=%.4f", float(recall.mean()))
    return recall.astype(np.float32)


def auc_score(
        model               : object,
        test_interactions   : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        item_features       : Optional[sp.spmatrix] = None,
        num_threads         : int  = 1,
        preserve_rows       : bool = False,
        check_intersections : bool = True,
        ) -> np.ndarray:
    """ROC AUC - probability that a random positive ranks above 
       a random negative. A perfect model scores 1.0; a random 
       model scores 0.5. Users with no test positives or where 
       all items are negative return 0.5. The return is np.ndarray
       float32, shape (n_active_users,) or (n_users,)
       """
    assert num_threads >= 1, "num_threads must be >= 1"
    logger.info("auc_score: num_threads=%d", num_threads)
    ranks = _get_ranks(model               = model,
                       test_interactions   = test_interactions,
                       train_interactions  = train_interactions,
                       user_features       = user_features,
                       item_features       = item_features,
                       num_threads         = num_threads,
                       check_intersections = check_intersections,
                       step_label          = "AUC")
    if not np.all(ranks.data >= 0):
        raise RuntimeError("Rank data contains negative values-"
                           "this should not happen.")
    with tqdm(total       = 3, 
              desc        = "AUC",
              colour      = _cfg.get('tqdm', 'colour'),
              ncols       = _cfg.getint('tqdm', 'ncols'),
              bar_format  = _cfg.get('tqdm', 'BarFormats'),
              unit        = 'process',
              mininterval = 0.1) as pbar:
        pbar.set_postfix_str("preparing train positive counts")
        auc = np.zeros(ranks.shape[0], dtype=np.float32)

        if train_interactions is not None:
            postive_num = np.squeeze(
                          np.array(train_interactions.getnnz(
                          axis=1)).astype(np.int32))
        else:
            postive_num = np.zeros(
                          test_interactions.shape[0], dtype = np.int32)
        pbar.update(1)

        pbar.set_postfix_str("computing AUC via Cython kernel")
        #Did in-place memory for AUC array (check function of flt cython)
        calculate_auc_from_rank(
            auc                 = auc,
            ranks               = CSRMatrix(ranks),
            num_train_positives = postive_num,
            rank_data           = ranks.data,
            num_threads         = num_threads)
        pbar.update(1)

        if not preserve_rows:
            auc = auc[test_interactions.getnnz(axis=1) > 0]
        pbar.update(1)
    gc.collect()
    logger.debug("auc_score: mean=%.4f", float(auc.mean()))
    return auc.astype(np.float32)

if __name__ == '__main__':
    pass