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


import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from typing import (Optional, List, Dict, Any, Callable, 
                    Union, Tuple)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from annoy import AnnoyIndex
    _ANNOY_AVAILABLE = True
except ImportError:
    _ANNOY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class FallbackError(Exception):
    """Base exception for fallback ranker errors."""

class DataNotLoadedError(FallbackError):
    """Raised when engine data is not available."""

class StrategyNotAvailableError(FallbackError):
    """Raised when the requested fallback strategy cannot be used."""


# ---------------------------------------------------------------------------
# ANN Index Wrapper
# ---------------------------------------------------------------------------
class ANNIndex:
    """
    Approximate Nearest Neighbour index for fast cosine similarity search.
    Uses FAISS if available, otherwise falls back to Annoy or brute force.
    """
    def __init__(self,
                 vectors     : np.ndarray,
                 use_gpu     : bool = False,
                 metric      : str  = 'cosine',
                 n_trees     : int  = 100,
                 force_brute : bool = False,
                ) -> None:
        self.vectors = vectors.astype(np.float32)
        self.dim     = vectors.shape[1]
        self.metric  = metric
        self.use_gpu = use_gpu
        self._index  = None
        if force_brute or (not _FAISS_AVAILABLE and not _ANNOY_AVAILABLE):
            self._method = 'brute'
            logger.debug("ANN: menggunakan brute‑force cosine similarity.")
            return
        if _FAISS_AVAILABLE and not force_brute:
            self._method = 'faiss'
            self._build_faiss()
        elif _ANNOY_AVAILABLE:
            self._method = 'annoy'
            self._build_annoy(n_trees)
        else:
            self._method = 'brute'


    def _build_faiss(self):
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors_normalized = self.vectors / norms
        index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(vectors_normalized)
        self._index = index
        logger.debug(f"ANN: FAISS index built with {self._index.ntotal} vectors.")

    def _build_annoy(self, n_trees: int):
        t = AnnoyIndex(self.dim, 'angular')
        for i, vec in enumerate(self.vectors):
            t.add_item(i, vec)
        t.build(n_trees)
        self._index = t
        logger.debug(f"ANN: Annoy index built with {n_trees} trees.")

    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = query_vectors.astype(np.float32)
        if self._method == 'brute':
            sim = cosine_similarity(query, self.vectors)
            idx = np.argsort(-sim, axis=1)[:, :k]
            dist = -np.take_along_axis(sim, idx, axis=1)
            return idx, dist

        if self._method == 'faiss':
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            query_norm = query / norms
            D, I = self._index.search(query_norm, k)
            dist = 1.0 - D
            return I, dist

        if self._method == 'annoy':
            n_queries = query.shape[0]
            indices = np.zeros((n_queries, k), dtype=int)
            distances = np.zeros((n_queries, k))
            for i, q in enumerate(query):
                idx, d = self._index.get_nns_by_vector(q, k, include_distances=True)
                indices[i, :len(idx)] = idx
                distances[i, :len(d)] = d
            return indices, distances

        # Fallback brute
        sim = cosine_similarity(query, self.vectors)
        idx = np.argsort(-sim, axis=1)[:, :k]
        dist = -np.take_along_axis(sim, idx, axis=1)
        return idx, dist


# ---------------------------------------------------------------------------
# Fallback Context
# ---------------------------------------------------------------------------
@dataclass
class FallbackContext:
    top_k                : int
    candidate_items      : pd.DataFrame
    candidate_vectors    : Optional[np.ndarray]
    current_item_ids     : List[Any]
    user_profile         : Optional[np.ndarray]
    popularity_scores    : Optional[pd.Series]
    collaborative_scores : Optional[Dict[Any, float]]
    random_state         : int = 4


# ---------------------------------------------------------------------------
# Base Strategy (ABC)
# ---------------------------------------------------------------------------
class BaseFallbackStrategy(ABC):
    @abstractmethod
    def select_items(self, context: FallbackContext) -> pd.DataFrame:
        '''Select fallback items from the candidate pool.'''
        raise NotImplementedError(
        "Subclasses of BaseFallbackStrategy must implement select_items()")


class ContentBasedStrategy(BaseFallbackStrategy):
    def __init__(self, 
        similarity_func = cosine_similarity, 
        min_similarity  : float = -1.0):
        self.similarity_func = similarity_func
        self.min_similarity = min_similarity

    def select_items(self, 
        context: FallbackContext) -> pd.DataFrame:
        if context.user_profile is None or context.candidate_vectors is None:
            raise StrategyNotAvailableError("Content strategy requires vectors.")
        sim = self.similarity_func(context.user_profile, context.candidate_vectors).flatten()
        valid = sim >= self.min_similarity
        if valid.sum() == 0:
            valid = np.ones_like(sim, dtype=bool)
        indices = np.where(valid)[0]
        top_idx = indices[np.argsort(sim[indices])[::-1][:context.top_k]]
        return context.candidate_items.iloc[top_idx]


class PopularityStrategy(BaseFallbackStrategy):
    def select_items(self, context: FallbackContext) -> pd.DataFrame:
        if context.popularity_scores is None:
            n = min(context.top_k, len(context.candidate_items))
            return context.candidate_items.sample(n, random_state=context.random_state)
        cand_scores = context.popularity_scores.reindex(
            context.candidate_items.index, fill_value = 0
        ).sort_values(ascending=False)
        return context.candidate_items.loc[cand_scores.head(context.top_k).index]


class CollaborativeStrategy(BaseFallbackStrategy):
    def select_items(self, context: FallbackContext) -> pd.DataFrame:
        if context.collaborative_scores is None:
            n = min(context.top_k, len(context.candidate_items))
            return context.candidate_items.sample(n, random_state=context.random_state)
        scores = pd.Series(context.collaborative_scores).reindex(
            context.candidate_items.index, fill_value=0.0)
        top_idx = scores.sort_values(ascending=False).head(context.top_k).index
        return context.candidate_items.loc[top_idx]


class HybridStrategy(BaseFallbackStrategy):
    def __init__(self, 
                 w_content = 0.4, 
                 w_pop     = 0.3, 
                 w_collab  = 0.3, 
                 min_similarity = -1.0):
        assert abs(w_content + w_pop + w_collab - 1.0) < 1e-6
        self.weights = (w_content, w_pop, w_collab)
        self.min_similarity = min_similarity

    def select_items(self, context: FallbackContext) -> pd.DataFrame:
        w_c, w_p, w_cl = self.weights
        # Content
        if context.candidate_vectors is not None and context.user_profile is not None:
            content_sim = cosine_similarity(context.user_profile, context.candidate_vectors).flatten()
        else:
            content_sim = np.zeros(len(context.candidate_items))

        # Popularity (0-1)
        if context.popularity_scores is not None:
            pop_raw = context.popularity_scores.reindex(
                context.candidate_items.index, fill_value=0
            ).values
            if pop_raw.max() > 0:
                pop_score = pop_raw / pop_raw.max()
            else:
                pop_score = pop_raw
        else:
            pop_score = np.zeros(len(context.candidate_items))

        # Collaborative (0-1)
        if context.collaborative_scores is not None:
            coll_dict = context.collaborative_scores
            coll_raw = np.array([coll_dict.get(idx, 0.0) for idx in context.candidate_items.index])
            if coll_raw.max() > 0:
                coll_score = coll_raw / coll_raw.max()
            else:
                coll_score = coll_raw
        else:
            coll_score = np.zeros(len(context.candidate_items))

        final = w_c * content_sim + w_p * pop_score + w_cl * coll_score
        final[content_sim < self.min_similarity] = -np.inf
        top_idx = np.argsort(-final)[:context.top_k]
        return context.candidate_items.iloc[top_idx]


# ---------------------------------------------------------------------------
# Utility: cache key for vectors
# ---------------------------------------------------------------------------
def _compute_cache_key(feature_cols: List[str], catalog_hash: str) -> str:
    return hashlib.md5(f"{sorted(feature_cols)}|{catalog_hash}".encode()).hexdigest()

íf __name__ == "__main__":
    pass