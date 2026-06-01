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
import numpy as np
import scipy.sparse as sp
from   tqdm.auto import tqdm
from   typing    import Optional

LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))
from configs import _cfg, logger


def CCC_k(
        model              : object,
        test_interactions  : sp.spmatrix,
        train_interactions : Optional[sp.spmatrix] = None,
        user_features      : Optional[sp.spmatrix] = None,
        item_features      : Optional[sp.spmatrix] = None,
        k                  : int = 10,
        batch_size         : int = 500,
        num_threads        : int = 4,
    ) -> float:
    """
    Calculate Catalog Coverage at k (Catalog@k)
    __________________________________________________
    Catalog Coverage metric measures the proportion of unique 
    items from the entire catalog that are recommended to 
    at least one user within their top-k recommendation list.
    High coverage implies the model avoids severe popularity
    bias and effectively leverages the long-tail catalog.
    The Return is float number as catalog coverage score 
    bounded between 0.0 and 1.0.
    """
    assert num_threads >= 1, "num_threads must be >= 1"
    assert k >= 1, "k must be >= 1"
    assert batch_size >= 1, "batch_size must be >= 1"
    n_users, n_items = test_interactions.shape
    logger.info("catalog_coverage_at_k: k = %d, users = %d,"
                "items = %d, batch_size = %d",
                 k, n_users, n_items, batch_size)

    active_users   = np.where(test_interactions.getnnz(axis=1) > 0)[0]
    n_active_users = len(active_users)
    logger.debug("Identify users with active profiles in "
                 "the test matrix to evaluate")
    if n_active_users == 0:
        logger.warning("No active users found in test_interactions.")
        return 0.0

    reco_itemmask = np.zeros(n_items, dtype=bool)
    all_item_ids  = np.arange(n_items, dtype=np.int32)
    total_batches = int(np.ceil(n_active_users / batch_size))
    for batch_idx in tqdm(range(total_batches), 
                          desc        = f"Catalog Coverage@{k}",
                          colour      = _cfg.get('tqdm', 'colour'),
                          ncols       = _cfg.getint('tqdm', 'ncols'),
                          bar_format  = _cfg.get('tqdm', 'BarFormats'),
                          unit        = 'batch',
                          mininterval = 0.1):
        # Materialize a dense score matrix for the current 
        # user block: shape (current, n_items)
        # This handles models expecting a vectorized sequence 
        # of users matched against all items
        start_u     = batch_idx * batch_size
        end_u       = min(start_u + batch_size, n_active_users)
        batch_users = active_users[start_u:end_u]
        current     = len(batch_users)
        user_ids_extended = np.repeat(batch_users, n_items)
        item_ids_extended = np.tile(all_item_ids, current)
        raw_scores  = model.predict(
                        user_ids_extended, 
                        item_ids_extended, 
                        user_features = user_features, 
                        item_features = item_features, 
                        num_threads   = num_threads)
        scores      = raw_scores.reshape(current, n_items)

        # Mask out training items to prevent them from 
        # entering the top-k evaluation pool
        if train_interactions is not None:
            train_sliced = train_interactions[batch_users]
            if train_sliced.nnz > 0:
                row_indices = train_sliced.tocoo().row
                col_indices = train_sliced.tocoo().col
                scores[row_indices, col_indices] = -np.inf
        partition = np.argpartition(scores, -k, axis=1)[:, -k:]
        reco_itemmask[partition.ravel()] = True

    # Calculate final coverage ratio
    total_unique   = int(reco_itemmask.sum())
    coverage_score = float(total_unique) / float(n_items)
    gc.collect()
    logger.debug("unique_recommended = %d, global_coverage = %.4f",
                 total_unique, coverage_score)
    return coverage_score


def safe_normalize(
        array : object, 
        axis  : int = 1,
    ) -> object:
    """
    Perform row-wise or column-wise L2 
    normalization on dense or sparse matrices.
    This is a lightweight, zero-sklearn 
    replacement for row/col-wise L2 normalization 
    that maintains strict memory efficiency 
    and execution speed.
    """
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (column-wise) or 1 (row-wise).")
    if sp.issparse(array):
        logger.debug("Processing sparse matrix along axis %d", axis)
        sum_of_squares = np.array(array.power(2).sum(axis=axis)).ravel()
        norms = np.sqrt(sum_of_squares)
        norms[norms == 0] = 1.0
        inv_norms = 1.0 / norms
        if axis == 1:
            return sp.diags(inv_norms) @ array
        else:
            return array @ sp.diags(inv_norms)
    else:
        logger.debug("Processing dense matrix along axis %d", axis)
        if axis == 1:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return array / norms
        else:
            norms = np.linalg.norm(array, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            return array / norms


def ILD_k(
        model               : object,
        test_interactions   : sp.spmatrix,
        item_features       : sp.spmatrix,
        train_interactions  : Optional[sp.spmatrix] = None,
        user_features       : Optional[sp.spmatrix] = None,
        k                   : int = 10,
        batch_size          : int = 500,
        num_threads         : int = 4,
    ) -> np.ndarray:
    """
    Calculate the Intra-List Diversity at k (ILD@k) per user using cosine distance.
    ILD computes the average pairwise distance between all items recommended to 
    a user in their top-k list. A higher ILD score signifies a more diverse 
    set of recommendations, reducing popularity bias and monotony.
    ____________________________________________________________________
    The return is a 1D float32 array containing the ILD@k score for 
    each evaluated user.
    """
    assert num_threads >= 1, "num_threads must be >= 1"
    assert k >= 2, "k must be >= 2 to compute pairwise diversity"
    assert batch_size >= 1, "batch_size must be >= 1"
    assert item_features is not None, "item_features matrix is required to compute ILD"
    n_users, n_items = test_interactions.shape
    logger.debug("k = %d, users = %d, items = %d, batch_size = %d", 
                  k, n_users, n_items, batch_size)

    active_users   = np.where(test_interactions.getnnz(axis=1) > 0)[0]
    n_active_users = len(active_users)
    logger.debug("Filter out users with no test profile tracking")
    if n_active_users == 0:
        logger.warning("No active users found in test_interactions.")
        return np.zeros(0, dtype=np.float32)

    # Pre-normalize item to transforms downstream Cosine Similarity computations
    logger.debug("Pre-normalizing item features matrix")
    if sp.issparse(item_features):
        norm_features = safe_normalize(array = item_features, axis = 1)
    else:
        norms = np.linalg.norm(item_features, axis = 1, keepdims = True)
        norms[norms == 0] = 1.0
        norm_features = item_features / norms

    # Array allocation to store individual ILD score per active user
    ild_scores    = np.zeros(n_active_users, dtype=np.float32)
    all_item_ids  = np.arange(n_items, dtype=np.int32)
    total_batches = int(np.ceil(n_active_users / batch_size))
    triu_indices  = np.triu_indices(k, k=1)
    for batch_idx in tqdm(range(total_batches), 
                          desc        = f"Intra-List Diversity@{k}",
                          colour      = _cfg.get('tqdm', 'colour'),
                          ncols       = _cfg.getint('tqdm', 'ncols'),
                          bar_format  = _cfg.get('tqdm', 'BarFormats'),
                          unit        = 'batch',
                          mininterval = 0.1):
        start_u     = batch_idx * batch_size
        end_u       = min(start_u + batch_size, n_active_users)
        batch_users = active_users[start_u:end_u]
        current     = len(batch_users)
        user_ids_extended = np.repeat(batch_users, n_items)
        item_ids_extended = np.tile(all_item_ids, current)
        raw_scores  = model.predict(
                        user_ids_extended, 
                        item_ids_extended, 
                        user_features = user_features, 
                        item_features = item_features, 
                        num_threads   = num_threads)
        scores = raw_scores.reshape(current, n_items)

        # Invalidate training history items
        if train_interactions is not None:
            train_sliced = train_interactions[batch_users]
            if train_sliced.nnz > 0:
                row_indices = train_sliced.tocoo().row
                col_indices = train_sliced.tocoo().col
                scores[row_indices, col_indices] = -np.inf

        # Extract top-k item indices per user via O(N) partitioning
        partition = np.argpartition(scores, -k, axis=1)[:, -k:]
        for idx in range(current):
            top_k_items  = partition[idx]
            X_k          = norm_features[top_k_items]
            if sp.issparse(X_k):
                msimilar = (X_k @ X_k.T).toarray()
            else:
                msimilar = X_k @ X_k.T
            
            # Extract unique pairs using upper triangular matrix mask and store mean distance
            distance_matrix = 1.0 - msimilar
            global_idx      = start_u + idx
            ild_scores[global_idx] = distance_matrix[triu_indices].mean()
    gc.collect()
    logger.debug("Mean prediction is %.4f", float(ild_scores.mean()))
    return ild_scores


def Novelty_k(
        model             : object,
        test_interactions : sp.spmatrix,
        train_interactions: sp.spmatrix,
        user_features     : Optional[sp.spmatrix] = None,
        item_features     : Optional[sp.spmatrix] = None,
        k                 : int = 10,
        batch_size        : int = 500,
        num_threads       : int = 4,
    ) -> np.ndarray:
    """
    Calculate Novelty at k (Novelty@k) per user based 
    on item popularity self-information. Novelty measures 
    how unexpected or uncommon the recommended items are 
    relative to the global item popularity in the training 
    set. It is defined as the average self-information 
    (surprisal) of the top-k recommended items:
    Self-Information(i) = -log2(interactions(i) / total_users).
    ___________________________________________________________
    The Returns is a 1D float32 array containing the Novelty@k 
    score for each evaluated user.
    """
    assert k >= 1,           "k must be >= 1"
    assert num_threads >= 1, "num_threads must be >= 1"
    assert batch_size >= 1,  "batch_size must be >= 1"
    assert train_interactions is not None, 
        "train_interactions is strictly required to derive item popularity profiles"
    
    n_users, n_items = test_interactions.shape
    logger.debug("Check Parameter: k = %d, users = %d, items = %d, batch_size = %d", 
                  k, n_users, n_items, batch_size)
    active_users   = np.where(test_interactions.getnnz(axis=1) > 0)[0]
    n_active_users = len(active_users)
    if n_active_users == 0:
        logger.warning("No active users found in test_interactions.")
        return np.zeros(0, dtype=np.float32)

    logger.debug("Computing global item self"
                 "information values from training matrix")
    
    # Calculate the frequency of each item across all users in the training matrix
    item_counts           = np.squeeze(np.array(train_interactions.getnnz(axis=0)))
    item_counts           = np.maximum(item_counts, 1)
    item_probabilities    = item_counts / float(train_interactions.shape[0])
    item_self_information = -np.log2(item_probabilities)

    # Array allocation to store individual novelty score per active user
    novelty_scores = np.zeros(n_active_users, dtype=np.float32)
    all_item_ids   = np.arange(n_items, dtype=np.int32)
    total_batches  = int(np.ceil(n_active_users / batch_size))
    for batch_idx in tqdm(range(total_batches), 
                          desc        = f"Novelty@{k}",
                          colour      = _cfg.get('tqdm', 'colour'),
                          ncols       = _cfg.getint('tqdm', 'ncols'),
                          bar_format  = _cfg.get('tqdm', 'BarFormats'),
                          unit        = 'batch',
                          mininterval = 0.1):
        start_u     = batch_idx * batch_size
        end_u       = min(start_u + batch_size, n_active_users)
        batch_users = active_users[start_u:end_u]
        current     = len(batch_users)
        user_ids_extended = np.repeat(batch_users, n_items)
        item_ids_extended = np.tile(all_item_ids, current)
        raw_scores  = model.predict(
                        user_ids_extended, 
                        item_ids_extended, 
                        user_features = user_features, 
                        item_features = item_features, 
                        num_threads   = num_threads)
        scores = raw_scores.reshape(current, n_items)

        # Mask out training history to isolate unseen recommendation tracking
        train_sliced = train_interactions[batch_users]
        if train_sliced.nnz > 0:
            row_indices = train_sliced.tocoo().row
            col_indices = train_sliced.tocoo().col
            scores[row_indices, col_indices] = -np.inf

        # Extract top-k item indices per user via O(N) partitioning
        partition = np.argpartition(scores, -k, axis=1)[:, -k:]
        novelty   = item_self_information[partition]
        novelty_scores[start_u:end_u] = novelty.mean(axis=1)
    gc.collect()
    logger.debug("Novelty Mean Score = %.4f", float(novelty_scores.mean()))
    return novelty_scores


if __name__ == '__main__':
    pass