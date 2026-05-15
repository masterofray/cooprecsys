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
import re
import sys
import pandas as pd
import lightgbm as lgb
from typing import List
from pathlib import Path
from copy import deepcopy
from ipdb import set_trace
from datetime import datetime

LocDir = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(LocDir))

from prepare  import latest_found
from configs  import LTRConfig, logger, FallbackConfig, _cfg
from features import (LabelEncoderManager, load_data, TrueString,
                      DateProcessor, Inference_DataSplit, DataProcessor)
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
    rawdata               = load_data(data_path)
    
    # 2. Data Preprocessing
    cleanstrCL            = TrueString(rawdata)
    rawdata['TotalPrice'] = pd.to_numeric(rawdata['TotalPrice'], 
                            errors = 'coerce').astype('float64')
    
    # 3. Date feature engineer and Label Encoder
    dateproc = DateProcessor(rawdata)
    data02   = dateproc.fit()
    dropcolm = dateproc.candfeat
    data02.drop(dropcolm, axis = 1, inplace = True)
    DataEnc, LEM = LoadEncode(Data = data02, 
                   feature_column  = cleanstrCL)
    
    # 4. Define Schema & Features
    QueryID     = "CustomerID"
    LabelID     = "CategoryID"
    FeatureID   = list(set(DataEnc.columns.tolist()) - {QueryID, LabelID})

    # X, Y, Group = Inference_DataSplit(data     = DataEnc,
                                      # features = FeatureID,
                                      # label    = LabelID,
                                      # query_id = QueryID)
    # DataLgb = lgb.Dataset(X, label      = Y,
                          # group         = Group,
                          # free_raw_data = False)
    # logger.debug("LightGBM datasets built - Data Prediction: %d rows", len(Y))
    
    # 5. Initialize your config
    logger.debug("Lets try to intilized the main config")
    LTRcfg = LTRConfig.from_ini(ini_path = str(
             LocDir/'configs'/'configuration.ini'), 
             features = FeatureID)
    LTRcfg.feature.label    = LabelID
    LTRcfg.feature.query_id = QueryID
    LTRcfg.feature.features = FeatureID
    LTRcfg.inference.top_k  = 25
    
    # 6. Muat model LTR dan mulai process inferensi
    logger.debug("Begin the core inferencing!")
    infr = LTRModelInference(LTRcfg)
    infr.data   = DataEnc #DataLgb
    infr.encman = LEM
    modelpath   = latest_found(
                  dir       = str(LocDir.parent), 
                  keyword   = 'model', 
                  recursive = True, 
                  Not4Json  = True)
    if modelpath.exists():
        LTRcfg.model_path = modelpath
        infr.model_path   = modelpath
        infr.load_model()
        logger.info(f'Model is found and already load from {modelpath}.')
    else:
        logger.error(f'Model is not found for {modelpath}.')
        raise FileNotFoundError()

    # 7. Post Processing of Inferencing data
    logger.debug("Doing Post Processing using AdaptiveFallbackRanker\n"
                 "and have FallbackConfig loaded from _cfg")
    Fallcfg = FallbackConfig.from_configparser(_cfg, section = 'FALLBACK')
    ranker  = AdaptiveFallbackRanker(engine = infr, config = Fallcfg)
    
    # 8. Running ranking with fallback
    logger.info("Starting fallback ranking process.")
    TheResult = ranker()
    
    # 9. Saving Reusult
    today = datetime.today()
    dstr  = today.strftime("%Y%m%d")
    odir  = LocDir.parent/'artifacts'/f'{dstr}_ranks.parquet'
    odir.parent.mkdir(exist_ok = True)
    ranker.save_rankings(str(odir), as_parquet = True)
    
    # Finally
    logger.info(f"Ranking completed. Result shape: {TheResult.shape}.")
    logger.debug(f"Saved rankings to {odir}.")
    logger.debug(TheResult.sample(4))
    return TheResult


if __name__ == '__main__':
    TheResult = InferenceTest()
    print(TheResult.head())
    
