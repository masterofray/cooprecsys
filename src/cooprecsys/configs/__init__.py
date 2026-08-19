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

import ast
from pathlib import Path
from configparser import ConfigParser

LocDir = Path(__file__).resolve().parents[0]
_cfg   = ConfigParser()
_cfg.read(LocDir / "configuration.ini")

from .fallback_config import FallbackConfig
from .lgbm_config     import LTRConfig
from .logged          import setup_logging
logger  = setup_logging()
verbose = True if (_cfg.get('logging', 'level')).upper() in ['DEBUG', 'INFO'] else False

def _cfglist(Config  : object, 
             section : str, 
             option  : str,
            ):
    try:
        raws   = Config.get(section, option)
        parsed = ast.literal_eval(raws)
        if isinstance(parsed, (list, tuple)):
            return parsed
        else:
            logger.error(f"Value from [{section}] -> '{option}'"
                          "is not a list or tuple as valid format.")
            raise ValueError()
    except configparser.NoSectionError:
        logger.error(f"Section [{section}] is not found.")
        return
    except configparser.NoOptionError:
        logger.error(f"Option '{option}' is not found at [{section}].")
        return
    except (ValueError, SyntaxError) as arc:
        logger.error(f"Became error when try to parse : {arc}")
        return
