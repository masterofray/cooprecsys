#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-10"


import re
from pathlib import Path
from shutil import copy2
from copy import deepcopy
from typing import Optional
from functools import lru_cache


def FileCopier(Scrpath  : Path, 
               Destdir  : Path,
              ) -> Path:
    srcp = Path(Scrpath)
    dstd = Path(Destdir)
    if not srcp.exists():
        raise FileNotFoundError(f'File tidak ditemukan: {srcp}')
    if not srcp.is_file():
        raise ValueError(f"Bukan file: {srcp}")
    dstd.mkdir(parents = True, exist_ok = True)
    DestPath = dstd / srcp.name
    copy2(srcp, DestPath)
    return DestPath

@lru_cache(maxsize = 128)
def _cachedlost(str_dir   : Path, 
                keyword   : str, 
                recursive : bool, 
                Not4Json  : bool,
               ) -> Optional[str]:
    """Internal cached function that works with string paths"""
    dir_path      = Path(str_dir).resolve()
    keyword_lower = keyword.lower()
    date_pattern  = re.compile(r"^(\d{8})")
    key_pattern   = re.compile(rf'{re.escape(keyword_lower)}', re.IGNORECASE)
    if not dir_path.is_dir():
        return None

    if not Not4Json:
        best_json  = None
        best_other = None
        paths      = dir_path.rglob("*") if recursive else dir_path.iterdir()
        for path in paths:
            if not path.is_file():
                continue
            if keyword_lower not in path.name.lower():
                continue
            if not key_pattern.search(path.name):
                continue
            date_match = date_pattern.match(path.name)
            if not date_match:
                continue

            date_int = int(date_match.group(1))
            is_json  = path.suffix.lower() == ".json"
            path_str = str(path)
            if is_json:
                if best_json is None or date_int > best_json[0]:
                    best_json = (date_int, path_str)
            else:
                if best_other is None or date_int > best_other[0]:
                    best_other = (date_int, path_str)
        if best_json:
            return best_json[1]
        if best_other:
            return best_other[1]
        return None
    else:
        paths = dir_path.rglob("*") if recursive else dir_path.iterdir()
        for path in paths:
            if not path.is_file():
                continue
            if not (key_pattern.search(path.name) or keyword_lower in path.name):
                continue
            if path.suffix.lower() == ".json":
                continue
            return deepcopy(path)
        return None


def latest_found(dir       : Path, 
                 keyword   : str  = "encoder",
                 recursive : bool = True,
                 Not4Json  : bool = False,
                ) -> Optional[Path]:
    """Wrapper function that converts Path to string for caching"""
    filepath = _cachedlost(dir, keyword, recursive, Not4Json)
    return Path(filepath) if filepath else str()


if __name__ == '__main__':
    test = Path.cwd().resolve().parents[2]
    modelpath = latest_found(dir = test, 
                keyword = 'encode', Not4Json = True)
    if modelpath.exists():
        print(modelpath)