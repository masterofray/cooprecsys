#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-03"

'''
encdec.py
__________________________________________________________________
Label encoding and decoding utilities for LightGBM LTR inference.
Handles string-to-integer encoding for model features.
Author: MiniMax Agent
'''

import os
import sys
import json
import pandas as pd
import cloudpickle as cp
from pathlib   import Path
from numpy     import clip
from tqdm.auto import tqdm
from datetime  import datetime
from copy      import deepcopy
from typing    import Dict, List, Optional, Any, Union
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

LocDir = Path(__file__).resolve()
sys.path.append(str(LocDir.parents[1]))
from configs  import logger, _cfg
from prepare  import latest_found


class LabelEncoderManager(object):
    """Manages label encoders for string columns with persistence support."""
    def __init__(self, 
                 data   : pd.DataFrame,
                 Column : List[str],
                 EncDir : Optional[Path] = None,
                 Remove4Done : bool = None,
                ) -> None:
        """Initialize encoder manager.
        Args:
            data   : dataframe that will be encoding/decoding
            EncDir : Directory to save/load encoder files
        """
        logger.info('Begin to initialized the Directory.')
        self._data   = None
        self.data    = data
        self._Column = None
        self.Column  = Column
        tempenc      = _cfg.get('PATHS', 'labelcoder')
        self.EncDir  = Path(EncDir) if EncDir else Path(tempenc)
        self._proceedcl    = list()
        self._removecolumn = Remove4Done if Remove4Done is not None else _cfg.getboolean('FEATURES', 'remove_column')
        self.EncDir.mkdir(parents=True, exist_ok=True)
        self._savepath = Path.cwd() / 'LabelEncoderManager.pkl'
        self.encoders        : Dict[str, LabelEncoder] = dict()
        self.encoder_classes : Dict[str, List[Any]] = dict()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Data harus berupa pandas DataFrame")
        if value.empty:
            raise ValueError("Your dataframe is empty")
        if value.shape[0] < 2:
            raise ValueError("Your dataframe is not sufficient")
        logger.debug("Data updated successfully.")
        self._data = value

    @property
    def Column(self) -> List[str]:
        return self._Column

    @Column.setter
    def Column(self, value: List[str]) -> None:
        if not isinstance(value, list):
            raise TypeError("Column harus berupa List[str]")
        if self._data is not None:
            missing = [c for c in value if c not in self._data.columns]
            if missing:
                logger.warning(f"The Columns are not in our dataframe: {missing}")
        self._Column = value


    def fit(self) -> None:
        for item in tqdm(self.Column, 
                         desc        = 'Fit encoder Labels',
                         colour      = _cfg.get('tqdm', 'colour'),
                         ncols       = _cfg.getint('tqdm', 'ncols'),
                         bar_format  = _cfg.get('tqdm', 'BarFormats'),
                         unit        = 'Column',
                         mininterval = 0.1):
            Temp_enc = LabelEncoder()
            if item not in self.data.columns:
                logger.error(f"Column '{item}' not found in dataframe")
                continue
            values = self.data[item].fillna("__MISSING__").astype(str)
            Temp_enc.fit(values)
            self.encoders[item] = Temp_enc
            self.encoder_classes[item] = Temp_enc.classes_.tolist()
            logger.debug(f"Fitted encoder for '{item}': {len(Temp_enc.classes_)} unique values.")


    def transform(self) -> None:
        '''Transform columns using fitted encoders.'''
        for item in tqdm(self.Column,
                         desc        = 'Transform Labels in column',
                         colour      = _cfg.get('tqdm', 'colour'),
                         ncols       = _cfg.getint('tqdm', 'ncols'),
                         bar_format  = _cfg.get('tqdm', 'BarFormats'),
                         unit        = 'Column',
                         mininterval = 0.1):
            if item not in self.encoders:
                logger.error(f"No encoder found for {item}.")
                continue
            tempenc = self.encoders[item]
            values  = self.data[item].fillna("__MISSING__").astype(str)
            known_classes = set(tempenc.classes_)
            values = values.apply(
                lambda x: x if x in known_classes else tempenc.classes_[0])
            newID = f'{item}_enc'
            self.data[newID] = tempenc.transform(values)
            self._proceedcl.append(newID)
            if self._removecolumn:
                self.data.drop([item], axis = 1, inplace = True)

    def fit_transform(self) -> None:
        self.fit()
        self.transform()

    def inverse_transform(self,
            data      : pd.DataFrame = pd.DataFrame([]),
            removeEnc : bool = True,
        ) -> pd.DataFrame:
        if data.empty:
            data = deepcopy(self.data)
        for item in tqdm(self.Column,
                         desc        = 'Inverse the transformation Labels',
                         colour      = _cfg.get('tqdm', 'colour'),
                         ncols       = _cfg.getint('tqdm', 'ncols'),
                         bar_format  = _cfg.get('tqdm', 'BarFormats'),
                         unit        = 'Column',
                         mininterval = 0.1):
            ecol = f"{item}_enc"
            if ecol not in data.columns:
                if item in self.encoders:
                    logger.error(f"'{ecol}' not found in dataframe for '{item}'.")
                continue
            if item not in self.encoders:
                logger.error(f"No encoder found for '{item}'.")
                continue
            encoder         = self.encoders[item]
            encoded_values  = data[ecol].values
            max_idx         = len(encoder.classes_) - 1
            safe_values     = clip(encoded_values, 0, max_idx).astype(int)
            decoded_values  = encoder.inverse_transform(safe_values)
            data[item]      = decoded_values
            if removeEnc:
                data.drop([ecol], axis = 1, inplace = True)
        return data

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            dstr = datetime.now().strftime('%Y%m%d')
            path = self.EncDir / f"{dstr}_LabelEncoderManager.cloudpickle"
        encoder_data = {"encoders"        : self.encoders,
                        "encoder_classes" : self.encoder_classes}
        with open(path, 'wb') as f:
            cp.dump(encoder_data, f)
        logger.info(f"Encoders saved to: {path}")
        json_path = path.with_suffix('.json')
        self._savepath = deepcopy(path)
        logger.info(f'Save path is in {self._savepath}.')
        with open(json_path, 'w') as f:
            json.dump(encoder_data, f, indent = 2, default = str)
        logger.debug(f"Encoder JSON saved to: {json_path}")


    def load(self, path: Optional[Path] = './') -> None:
        """Loads the most recent encoder file if path is not provided."""
        if not Path(path).is_file():
            path = latest_found(
                        dir       = str(LocDir.parents[2]), 
                        keyword   = 'encode', 
                        recursive = True, 
                        Not4Json  = True)
        path      = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encoder file not found: {path}")
        logger.info(f"Loading encoders from: {path}")
        #set_trace()
        with open(path, 'rb') as file:
            encoder_data     = cp.load(file)
        self.encoders        = encoder_data["encoders"]
        self.encoder_classes = encoder_data["encoder_classes"]
        for item in self.encoders.keys():
            logger.debug(f"- {item}: {len(self.encoder_classes[item])} classes")


    def get_classes(self, column: str) -> List[Any]:
        '''Get all classes for a specific column encoder.'''
        logger.debug('run the get_classes function')
        if column not in self.encoder_classes:
            return list()
        return self.encoder_classes[column]


    def get_class_mapping(self, column: str) -> Dict[int, str]:
        '''Get mapping of encoded values to original classes.
        Dictionary mapping encoded int to original class string.'''
        logger.debug('run the get_class_mapping function')
        if column not in self.encoder_classes:
            return dict()
        return {i: cls for i, cls in enumerate(self.encoder_classes[column])}


# ============================================================================
# DECODER FOR INFERENCE OUTPUT
# ============================================================================
class InferenceDecoder:
    """Decode model predictions back to human-readable format."""
    def __init__(self,
                 Data : pd.DataFrame,
                 encoder_manager: LabelEncoderManager,
                ) -> None:
        logger.debug('initialized The InferenceDecoder!')
        self.encoder_manager = encoder_manager
        self.data            = Data
        self._data           = None

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Data harus berupa pandas DataFrame")
        if value.empty:
            raise ValueError("Your dataframe is empty")
        if value.shape[0] < 2:
            raise ValueError("Your dataframe is not sufficient")
        logger.debug("Data updated successfully.")
        self._data = value

    def decode_product_names(self) -> None:
        '''Decode encoded ProductName back to original strings'''
        if "ProductName" in self.encoder_manager.encoders:
            self.data = self.encoder_manager.inverse_transform(self.data, ["ProductName"])
            logger.debug('Done for decode_product_names!')

    def decode_all_strings(self, string_columns: List[str]) -> pd.DataFrame:
        '''Decode all encoded string columns back to original values.'''
        self.data = self.encoder_manager.inverse_transform(self.data, string_columns)
        logger.debug('Done for decode_all_strings!')

    def create_readable_output(
            self,
            string_columns : List[str],
            ) -> pd.DataFrame:
        """
        Create human-readable output dataframe.
        string_columns : List of string columns to decode
        The Returns is DataFrame with decoded strings and cleaned column names
        """
        self.decode_all_strings(string_columns)
        rename_map = {f"{col}_enc": col for col in string_columns
                      if f"{col}_enc" in data.columns}
        self.data = self.data.rename(columns = rename_map)
        logger.debug('Done for create_readable_output!')


if __name__ == "__main__":
    # Test loading and usage
    logger.info("Data Preprocessing Module")
    import re
    pathdata = LocDir.parents[2] / 'data' / 'sampledata.parquet'
    dt    = pd.read_parquet(str(pathdata))
    strCL = dt.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(strCL)
    cleanstrCL = list(filter(
        lambda x: not re.search(r'date|hour|minute|time', x, flags=re.IGNORECASE), strCL))
    logger.info(cleanstrCL)
    Enc = LabelEncoderManager(data = dt, Column = cleanstrCL)
    Enc.fit_transform()
    
    newdata = Enc.data
    logger.warning(newdata.sample(6))
    logger.info('\n\n')
    logger.warning(newdata.info())
    logger.info('\n\n')
    logger.warning(newdata.describe())
    logger.info('\n\n')
    for item in Enc.encoders.keys():
        logger.info(f"- {item}: {len(Enc.encoder_classes[item])} classes")
