#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-19"


"""
Production Pipeline Smoke Test Suite.
Validates the end-to-end integration of ostensible,
ersetz, and flex modules, ensuring stable tensor 
alignment, matrix densities, and valid splitting splits.
"""

import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import List

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from configs    import logger
from features   import load_data
from noisemaker import (ExchangeResult,
                        coo_ttsplit,
                        exnorex,
                        usertts)


def Noisemaker_Smoke_Test(Data      : pd.DataFrame,
                          UserFeat  : List,
                          IteamFeat : List,
                          test_ratio : float = 0.25,
                         ) -> bool:
    """
    Executes an isolated integration test simulating
    standard production workloads. Verifies index 
    tracking, sparse coordinate alignment, and 
    validation partitioning.
    """
    logger.debug("Executing system validation check.")
    user_features_target = UserFeat
    item_features_target = IteamFeat
    logger.debug("Synthetic workload instantiated with shape %s", data.shape)
    try:
        logger.debug("Step 1: Compiling sparse tensors "
            "from raw transactional sources.")
        payload = exnorex(data              = data,
                          user_col          = "CustomerID",
                          item_col          = "CategoryID",
                          rating_col        = None,
                          user_feature_cols = user_features_target,
                          item_feature_cols = item_features_target)
        assert isinstance(payload, ExchangeResult), \
            "Payload contract type mismatch."
        n_users = len(payload.user_ids)
        n_items = len(payload.item_ids)
        logger.debug("Extracted space configuration: Unique "
                     "Users = %d, Unique Items = %d", n_users, n_items)
        assert payload.interactions.shape == (n_users, n_items), \
            "Interaction dimensions corrupted."
        assert payload.user_features.shape[0] == n_users, \
            "User-feature space axis mismatch."
        assert payload.item_features.shape[0] == n_items, \
            "Item-feature space axis mismatch."
        assert payload.sample_weight.nnz == payload.interactions.nnz, \
            "Confidence weight pattern mismatch vs observation space matrix."
        logger.info("Step 1 Complete: Tensor spaces perfectly aligned.")

        # Evaluate random coordinate validation split via flex
        logger.debug("Evaluating global coordinate-space matrix partitioning.")
        assert 0.02 <= test_ratio <= 0.9, 'Test Ratio is out of range!'
        train_coo, test_coo = coo_ttsplit(payload.interactions, 
                                          tratio = test_ratio, 
                                          rstate = 4)
        assert train_coo.shape == payload.interactions.shape, \
            "Train matrix allocation modified global dimensions."
        assert test_coo.shape == payload.interactions.shape, \
            "Test matrix allocation modified global dimensions."
        logger.info("Standard COO splitting validated. Train "
            "NNZ: %d | Test NNZ: %d", train_coo.nnz, test_coo.nnz)

        # Evaluate stratified user validation split via flex
        logger.info("Evaluating cold-start safety using user-stratified splitting.")
        train_user, test_user = usertts(payload.interactions,
                                        tratio = test_ratio, 
                                        rstate = 4)
        assert train_user.shape == payload.interactions.shape, \
            "User split corrupted matrix tracking framework."
        assert test_user.shape == payload.interactions.shape, \
            "User split corrupted matrix tracking framework."
        logger.info("User-stratified split validated. "
            "Train NNZ: %d | Test NNZ: %d", train_user.nnz, test_user.nnz)

        logger.info("System integration integrity verified: Smoke Test Passed.")
        return True

    except Exception as error:
        logger.error("Pipeline failure detected during integration "
            "trace: %s", str(error), exc_info = True)
        return False


if __name__ == "__main__":
    pathdata  = LocDir.parent / 'data' / 'sampledata.parquet'
    data      = load_data(pathdata)
    UserFeat  = ["CityName"]
    IteamFeat = ["ProductPrice"]
    success = Noisemaker_Smoke_Test(data, UserFeat, IteamFeat, 0.2)
