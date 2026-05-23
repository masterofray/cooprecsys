#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-07"

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

LocDir = Path(__file__).resolve().parents[1]
sys.path.append(str(LocDir))
from configs import logger


def Dict2Json(data       : Dict[str, Any],
              filepath   : Optional[str] = None,
              pretty     : bool = True, 
              handle_nan : bool = True,
             ) -> Optional[str]:
    '''
    Convert dictionary to JSON format
    data       : Dictionary to convert
    filepath   : If provided, saves to file instead of returning string
    pretty     : If True, formats with indentation
    handle_nan : If True, converts NaN to None
    '''
    indent = 2 if pretty else None
    if handle_nan:
        import math
        data = {k: (None if isinstance(v, float) and math.isnan(v) else v) 
                for k, v in data.items()}
    try:
        if filepath:
            fp = Path(filepath).resolve().parent
            fp.mkdir(parents = True, exist_ok = True)
            with open(filepath, 'w', encoding='utf-8') as files:
                json.dump(data, files, indent = indent, ensure_ascii = False)
            logger.info(f"Successfully exported to {filepath}")
            return None
        else:
            return json.dumps(data, 
                              indent = indent, 
                              ensure_ascii = False)
    except TypeError as Arc:
        logger.error(f"Error: Cannot serialize - {Arc}, Will save as joblib.")
        newfilepath = Path(filepath).with_suffix(".joblib")
        return joblib.dump(data, str(newfilepath), compress=('lzma', 9))


if __name__ == '__main__':
    dataj = {"name"    : "Machine Learning Model",
             "params"  : {"lr": 0.01, "epochs": 100},
             "metrics" : {"accuracy": 0.95, "loss": 0.123}}
    Dict2Json(dataj, "./model_config.json")