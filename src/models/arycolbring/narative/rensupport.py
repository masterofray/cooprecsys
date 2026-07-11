#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-06"

import sys
import math
import numpy   as np
from pathlib   import Path
from tqdm.auto import tqdm
from typing    import Dict, List, Any
from jinja2    import Environment, FileSystemLoader, select_autoescape

LocDir     = Path(__file__).parent.resolve()
sys.path.append(str(LocDir.parents[2]))
from configs   import logger, _cfg
from assets    import VendorPath
from prepare   import FileCopier

Tplatedir  = LocDir / "templates"
advdir     = LocDir / "sttrain"
readir     = LocDir / "stinferc"
IMG_DIR    = LocDir.parents[3] / 'img'
OUTPUT_DIR = (LocDir.parents[3] / _cfg.get('PATHS', 'ACB_rpath')).resolve()

def get_env() -> Environment:
    """Initialize Jinja2 environment."""
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(
              loader        = FileSystemLoader(str(Tplatedir)),
              autoescape    = select_autoescape(["html", "xml"]),
              trim_blocks   = True,
              lstrip_blocks = True)
        logger.debug("Jinja environment initialized successfully.")
    except Exception as exc:
        env = Environment()
        logger.error("Failed initializing Jinja environment.", exc_info = True)
        raise RuntimeError() from exc
    finally:
        return env


def copymaps(mapdict : Dict, 
             goal    : Path,
             train   : bool = True,
             dirlist : List = ['css', 'js'],
            ) -> Dict:
    fixpath       = advdir if train else readir
    for fdr in tqdm(dirlist,
                    desc   = 'Directory Dest',
                    colour = _cfg.get('tqdm', 'colour'),
                    ncols  = _cfg.getint('tqdm', 'ncols'),
                    unit   = 'dir'):
        try:
            xdir  = fixpath / fdr
            if xdir.exists():
                for item in xdir.glob('*'):
                    if item.is_file():
                        mapdict[item] = goal/fdr
        except Exception as err:
            logger.error(err)
    return mapdict

def runcopy(advisor : bool = True, 
            dest    : Path = None,
           ) -> None:
    dirname  = 'advisor' if advisor else 'reason'
    dest     = OUTPUT_DIR / dirname / 'assets' if dest is None else Path(dest).resolve()
    copy_map = {VendorPath['vcss'] : dest / 'css',
                VendorPath['vjs']  : dest / 'js'}
    for ico in VendorPath['icon']:
        copy_map[ico]          = dest / 'icon'
    copy_map[IMG_DIR/'favicon.ico']   = dest
    copy_map[IMG_DIR/'logo_red.jpg']  = dest
    cpnow = copymaps(mapdict   = copy_map,
                     goal      = dest,
                     train     = advisor)
    for src, dst_dir in cpnow.items():
        _ = FileCopier(Scrpath = src,
                       Destdir = dst_dir)
    logger.debug('runcopy already finish.')


def bealabel(label: str) -> str:
    try:
        label = label.replace("_", " ")
        replacements = {"ndcg": "NDCG",
                        "map": "MAP",
                        "mrr": "MRR",
                        "auc": "AUC",
                        "ctr": "CTR",
                        "at": "@",
                        "precision": "Precision",
                        "recall": "Recall"}
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label.title()
    except Exception:
        return str(label)


def safe_float(value     : Any, 
               precision : int = 4,
              ) -> str:
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            return f"{value:.{precision}f}"
        return str(value)
    except Exception:
        return str(value)


def detect_gauge_metric(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    keywords    = ["ndcg", "map", "mrr",
                   "precision", "recall", 
                   "accuracy", "auc", "f1"]
    return any(k in metric_name for k in keywords)


def copier(dest: Path = None, advisor: bool = True) -> Dict[str, str]:
    """Copy static assets to output directory and return path mappings.
    
    Args:
        dest: Output destination directory
        advisor: If True, copy training assets (sttrain), else inference assets (stinferc)
    
    Returns:
        Dictionary with static path mappings for template rendering
    """
    dirname = 'advisor' if advisor else 'reason'
    if dest is None:
        dest = OUTPUT_DIR / dirname / 'assets'
    else:
        dest = Path(dest).resolve() / 'assets'
    
    # Run the copy operation
    runcopy(advisor=advisor, dest=dest)
    
    # Return path mappings for templates
    return {
        "static_css":    str(dest / 'css'),
        "static_js":     str(dest / 'js'),
        "static_vendor": str(dest / 'vendor'),
        "static_img":    str(dest),
    }


if __name__ == '__main__':
    logger.info('Rensupport test to be here!\n')
    runcopy(advisor = True)