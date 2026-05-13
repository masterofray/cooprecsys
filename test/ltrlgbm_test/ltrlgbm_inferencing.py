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


import os
import sys
import pandas as pd
from typing import List
from pathlib import Path
from copy import deepcopy
from datetime import datetime

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))
from prepare  import latest_found
from features import LabelEncoderManager, load_data
from configs  import LTRConfig, logger, FallbackConfig
from models.ltr_lgbm.inout import AdaptiveFallbackRanker, LTRModelInference

def LoadEncode(Data           : pd.DataFrame,
               feature_column : List,
               location       : Path = None,
               clean          : bool = True,
              ) -> pd.DataFrame:
    lct  = Path(location) if location else LocDir.parent/'artifact'
    LEM  = LabelEncoderManager(
            data        = deepcopy(Data), 
            Column      = feature_column,
            EncDir      = lct,
            Remove4Done = clean)
    LEM.load()
    LEM.transform()
    EncData = LEM.data
    logger.debug('Finished to Encoding the data')
    return EncData, LEM


def InferenceTest():
    logger.info("Initializing Inference Models")
    logger.info("_" * 60)

    # 1. Calling data
    data_path = LocDir.parents[0] / 'data' / 'sampledata.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {str(data_path)}.")
    logger.info(f"Loading data from {data_path}.")
    data = load_data(data_path)
    
    # 2. Define Schema & Features
    QueryID   = "CustomerID"
    LabelID   = "CategoryID"
    FeatureID = list(set(data.columns.tolist()) - {'QueryID', 'LabelID'})
    
    # 3. Initialize your config
    LTRcfg = LTRConfig.from_ini(ini_path = str(
             LocDir/'configs'/'configuration.ini'), 
             features = TheFeatures)
    LTRcfg.feature.label    = LabelID
    LTRcfg.feature.query_id = QueryID
    LTRcfg.feature.features = FeatureID
    LTRcfg.inference.top_k  = 25
    
    # 4. Encoding data
    DataEnc, LEM = LoadEncode(Data = data, feature_column = FeatureID)
    infr = LTRModelInference(LTRcfg)
    infr.data    = DataEnc
    infr.encman  = LEM
    
    # 5. Muat model LTR
    modelpath = latest_found(dir = LocDir.parent, 
                keyword = 'ltr_model', Not4Json = True)
    if modelpath.exists():
        LTRcfg.model_path = modelpath
        infr.model_path = modelpath
        infr.load_model()
        logger.info(f'Model is found and already load from {modelpath}')
    else:
        logger.error(f"Model is not found for {modelpath}.")
        raise FileNotFoundError()

    # 6. Initialized AdaptiveFallbackRanker
    fb_config = FallbackConfig.from_configparser(_cfg, section = 'FALLBACK')
    logger.debug("FallbackConfig loaded from _cfg")
    ranker = AdaptiveFallbackRanker(engine = infr, config = fb_config)
    
    # 7. Running ranking with fallback
    logger.info("Starting fallback ranking process.")
    TheResult = ranker()
    today = datetime.today()
    dstr  = today.strftime("%Y%m%d")
    odir  = LocDir.parent/'artifacts'/f'{dstr}_ranks.parquet'
    odir.parent.mkdir(exist_ok = True)
    ranker.save_rankings(str(odir), as_parquet = True)
    logger.info(f"Ranking completed. Result shape: {TheResult.shape}")
    logger.debug(f"Saved rankings to {odir}.")
    logger.debug(TheResult.sample(4))
    return TheResult


if __name__ == '__main__':
    TheResult = InferenceTest()
    print(TheResult.head())
    
