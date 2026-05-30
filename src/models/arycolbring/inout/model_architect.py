#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-30"


"""
trainer.py
----------
Training subclass of AryColBringBase.

Implements ``fit`` and ``fit_partial`` (and the internal ``_run_epoch``
dispatcher) while delegating all inference work to ``AryColBringPredictor``.
Calling ``predict`` or ``predict_rank`` on this class raises
``NotImplementedError`` with a clear redirect message.
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from ..CLproximity import (
    CSRMatrix,
    fit_logistic,
    fit_warp,
    fit_bpr,
    fit_warp_kos,
)
from .Scaffold import (
    AryColBringBase,
    CYTHON_DTYPE,
    TQDM_COLOUR,
    TQDM_NCOLS,
)
from .data_utils import validate_sparse_matrix

logger = logging.getLogger(__name__)

__all__ = ["AryColBringTrainer"]


class AryColBringTrainer(AryColBringBase):
    """
    Training interface for the AryColBring collaborative filtering model.

    Exposes ``fit`` and ``fit_partial`` for all four supported loss functions:
    ``"logistic"``, ``"warp"``, ``"bpr"``, and ``"warp-kos"``.

    After training, call ``get_item_representations`` / ``get_user_representations``
    (inherited from ``AryColBringBase``) to export embeddings, or hand the fitted
    instance off to ``AryColBringPredictor`` for serving.

    Raises ``NotImplementedError`` if ``predict`` or ``predict_rank`` are called —
    use :class:`AryColBringPredictor` for inference.

    Parameters mirror ``AryColBringBase``; see that class for full documentation.
    """

    # ── public training interface ─────────────────────────────────────────────

    def fit(
        self,
        interactions:  sp.spmatrix,
        user_features: Optional[sp.spmatrix] = None,
        item_features: Optional[sp.spmatrix] = None,
        sample_weight: Optional[sp.coo_matrix] = None,
        epochs:        int  = 1,
        num_threads:   int  = 1,
        verbose:       bool = False,
    ) -> "AryColBringTrainer":
        """
        Fit the model from scratch, discarding any previous embedding state.

        Parameters
        ----------
        interactions  : sparse matrix [n_users × n_items]
                        Non-zero entries are treated as positive interactions.
        user_features : optional CSR matrix [n_users × n_user_features]
        item_features : optional CSR matrix [n_items × n_item_features]
        sample_weight : optional COO matrix of the same shape as interactions
        epochs        : number of full passes over the data
        num_threads   : OpenMP thread count (≥ 1)
        verbose       : show tqdm progress bar

        Returns
        -------
        self
        """
        logger.debug(
            "AryColBringTrainer.fit: epochs=%d num_threads=%d",
            epochs, num_threads,
        )
        self._reset_state()
        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions:  sp.spmatrix,
        user_features: Optional[sp.spmatrix] = None,
        item_features: Optional[sp.spmatrix] = None,
        sample_weight: Optional[sp.coo_matrix] = None,
        epochs:        int  = 1,
        num_threads:   int  = 1,
        verbose:       bool = False,
    ) -> "AryColBringTrainer":
        """
        Incrementally fit the model, preserving previous embedding state.

        Suitable for online / warm-start learning.
        Parameters mirror ``fit``; see that docstring for full details.

        Returns
        -------
        self
        """
        logger.debug(
            "AryColBringTrainer.fit_partial: loss=%s epochs=%d num_threads=%d",
            self._loss, epochs, num_threads,
        )

        if num_threads < 1:
            raise ValueError("num_threads must be ≥ 1")
        if epochs < 1:
            raise ValueError("epochs must be ≥ 1")

        # Convert to COO and enforce dtype
        interactions = interactions.tocoo()
        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        validate_sparse_matrix(interactions, "interactions")

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        user_features, item_features = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        # Guard against NaN / Inf in all inputs
        for arr, label in [
            (user_features.data, "user_features"),
            (item_features.data, "item_features"),
            (interactions.data,  "interactions"),
            (sample_weight_data, "sample_weight"),
        ]:
            self._check_input_finite(arr, label)

        # Lazy initialisation on first call
        if self.item_embeddings is None:
            self._initialize(
                self._no_components,
                item_features.shape[1],
                user_features.shape[1],
            )

        # Dimension consistency checks
        if item_features.shape[1] != self.item_embeddings.shape[0]:
            raise ValueError(
                f"item_features has {item_features.shape[1]} columns but "
                f"embedding has {self.item_embeddings.shape[0]} rows."
            )
        if user_features.shape[1] != self.user_embeddings.shape[0]:
            raise ValueError(
                f"user_features has {user_features.shape[1]} columns but "
                f"embedding has {self.user_embeddings.shape[0]} rows."
            )

        # Pre-build positives lookup for ranking losses
        pairwise_losses = ("warp", "bpr", "warp-kos")
        positives_lookup = None
        if self._loss in pairwise_losses:
            positives_lookup = CSRMatrix(
                self._get_positives_lookup_matrix(interactions)
            )

        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)

        with tqdm(
            total=epochs,
            desc="Training",
            colour=TQDM_COLOUR,
            ncols=TQDM_NCOLS,
            disable=not verbose,
        ) as pbar:
            for epoch in self._epoch_iterator(epochs, verbose=False):
                self._random_state.shuffle(shuffle_indices)
                model_data = self._get_model_data()

                self._run_epoch(
                    item_features=item_features,
                    user_features=user_features,
                    interactions=interactions,
                    sample_weight=sample_weight_data,
                    shuffle_indices=shuffle_indices,
                    num_threads=num_threads,
                    loss=self._loss,
                    positives_lookup=positives_lookup,
                    model_data=model_data,
                )

                self._check_finite()
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch + 1, "loss": self._loss})
                logger.debug(
                    "fit_partial: epoch %d/%d done", epoch + 1, epochs
                )

        gc.collect()
        logger.debug("fit_partial: training complete")
        return self

    # ── epoch dispatch ────────────────────────────────────────────────────────

    def _run_epoch(
        self,
        item_features:   sp.csr_matrix,
        user_features:   sp.csr_matrix,
        interactions:    sp.coo_matrix,
        sample_weight:   np.ndarray,
        shuffle_indices: np.ndarray,
        num_threads:     int,
        loss:            str,
        positives_lookup,
        model_data,
    ) -> None:
        """
        Dispatch one training epoch to the appropriate Cython kernel.

        Parameters
        ----------
        item_features    : CSR item feature matrix
        user_features    : CSR user feature matrix
        interactions     : COO interaction matrix
        sample_weight    : float32 array aligned with interactions.data
        shuffle_indices  : int32 permutation array shuffled each epoch
        num_threads      : OpenMP thread count
        loss             : active loss function name
        positives_lookup : CSRMatrix of positives (None for logistic loss)
        model_data       : FastAryColBring Cython struct
        """
        logger.debug("_run_epoch: loss=%s", loss)

        cy_item = CSRMatrix(item_features)
        cy_user = CSRMatrix(user_features)

        if loss == "warp":
            fit_warp(
                cy_item, cy_user, positives_lookup,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads, self._random_state,
            )
        elif loss == "bpr":
            fit_bpr(
                cy_item, cy_user, positives_lookup,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads, self._random_state,
            )
        elif loss == "warp-kos":
            fit_warp_kos(
                cy_item, cy_user, positives_lookup,
                interactions.row, shuffle_indices,
                model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                self._k, self._n,
                num_threads, self._random_state,
            )
        else:  # logistic
            fit_logistic(
                cy_item, cy_user,
                interactions.row, interactions.col,
                interactions.data, sample_weight,
                shuffle_indices, model_data,
                self._learning_rate, self._item_alpha, self._user_alpha,
                num_threads,
            )

    # ── inference methods — not implemented on the trainer ───────────────────

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Not available on the trainer.  Use ``AryColBringPredictor``."""
        raise NotImplementedError(
            "AryColBringTrainer does not support inference. "
            "Instantiate AryColBringPredictor with the same parameters "
            "and fitted embeddings for scoring."
        )

    def predict_rank(self, *args, **kwargs) -> sp.csr_matrix:
        """Not available on the trainer.  Use ``AryColBringPredictor``."""
        raise NotImplementedError(
            "AryColBringTrainer does not support inference. "
            "Instantiate AryColBringPredictor with the same parameters "
            "and fitted embeddings for ranking."
        )

if __name__ == '__main__':
    pass
