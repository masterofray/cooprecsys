#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-16"

import pandas as pd
#import lightgbm as lgb
from pathlib  import Path
from copy     import deepcopy
from datetime import datetime
from typing   import List, Tuple
from .inout   import AdaptiveFallbackRanker, LTRModelInference

LocDir = Path(__file__).resolve().parents[2]

from ...prepare  import latest_found
from ...configs  import LTRConfig, logger, FallbackConfig, _cfg
from ...features import (LabelEncoderManager, load_data, TrueString,
                      DateProcessor, Inference_DataSplit, DataProcessor)


def LoadEncode(Data           : pd.DataFrame,
               feature_column : List,
               location       : Path = None,
               clean          : bool = True,
              ) -> Tuple[pd.DataFrame, object]:
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


def InferenceTest(Datapath  : Path,
                  configpath: Path,
                  QueryID   : str  = 'CustomerID',
                  LabelID   : str  = 'CategoryID',
                  FilterDF  : List = None,
                  odir      : Path = None,
                 ) -> pd.DataFrame:
    logger.info("Initializing Inference Models")
    logger.info("_" * 60)

    # 1. Calling data
    if not Datapath.exists():
        raise FileNotFoundError(f"Could not find {str(Datapath)}.")
    logger.info(f"Loading data from {Datapath}.")
    rawdata = load_data(Datapath)
    
    # 2. Data Preprocessing
    cleanstrCL = TrueString(rawdata)
    for item in rawdata.columns:
        if 'price' in item.lower():
            rawdata[item] = pd.to_numeric(rawdata[item],
                            errors = 'coerce').astype('float32')
        else:
            continue
    
    # 3. Date feature engineer and Label Encoder
    dateproc = DateProcessor(rawdata)
    data02   = dateproc.fit()
    dropcolm = dateproc.candfeat
    data02.drop(dropcolm, axis = 1, inplace = True)
    DataEnc, LEM = LoadEncode(Data = data02, 
                   feature_column  = cleanstrCL)
    
    # 4. Define Schema & Features
    FeatureID   = list(set(DataEnc.columns.tolist()) - {QueryID, LabelID})
    #X, Y, Group = Inference_DataSplit(data     = DataEnc,
    #                                  features = FeatureID,
    #                                  label    = LabelID,
    #                                  query_id = QueryID)
    #DataLgb = lgb.Dataset(X, label      = Y,
    #                      group         = Group,
    #                      free_raw_data = False)
    #logger.debug("LightGBM datasets built - Data Prediction: %d rows", len(Y))
    
    # 5. Initialize your config
    logger.debug("Lets try to intilized the main config")
    LTRcfg = LTRConfig.from_ini(ini_path = str(configpath), 
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
    OutFinal  = LEM.inverse_transform(TheResult, True)
    if FilterDF is not None:
        OutFinal = OutFinal[FilterDF]
    
    # 9. Saving Reusult
    today = datetime.today()
    dstr  = today.strftime("%Y%m%d")
    if odir:
        odir = Path(odir)/f'{dstr}_ranks.parquet'
    else:
        odir  = LocDir.parent/'artifacts'/f'{dstr}_ranks.parquet'
    odir.parent.mkdir(parents = True, exist_ok = True)
    #ranker.save_rankings(str(odir), as_parquet = True)
    OutFinal.to_parquet(str(odir), 
             engine            = "pyarrow", 
             compression       = "gzip",
             compression_level = 9,
             index             = False)

    #10. Finally
    logger.info(f"Ranking completed. Result shape: {OutFinal.shape}.")
    logger.debug(f"Saved rankings to {odir}.")
    logger.debug(OutFinal.sample(4))
    return OutFinal

if __name__ == '__main__':
    pass