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
from typing import Optional

def latest_found(dir       : Path, 
                 keyword   : str = "encoder",
                 recursive : bool = True,
                ) -> Optional[Path]:
    """
    Search dir for files containing 'keyword' (case‑insensitive).
    - Prioritize .json files.
    - Extract a date in YYYYMMDD format from the start of the filename.
    - Return the file with the most recent date.
    """
    if not dir.is_dir():
        return None
    date_pattern = re.compile(r"^(\d{8})")
    keyword_re   = re.compile(re.escape(keyword), re.IGNORECASE)
    candidates   = list()
    iterator     = directory.rglob("*") if recursive else directory.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        if not keyword_re.search(path.name):
            continue
        date_match = date_pattern.match(path.name)
        if not date_match:
            continue
        date_int = int(date_match.group(1))
        is_json  = path.suffix.lower() == ".json"
        candidates.append((date_int, is_json, path))
    if not candidates:
        return None

    # Sort by: is_json=True first, then by 
    # date_int descending (latest first)
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return candidates[0][2]