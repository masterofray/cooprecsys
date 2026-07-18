#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-06"


from .statsrender import Gen_MiniStats

from pathlib  import Path
assDir = Path(__file__).parent.resolve()

def GetPath(dirs: str) -> list[Path]:
    Dirs = Path(dirs)
    if not Dirs.exists() or not Dirs.is_dir():
        return list()
    return [item.resolve() for item in Dirs.iterdir()]

VendorPath = {'icon' : GetPath(assDir / 'icon'),
              'vcss' : assDir / 'vendors_css.zip',
              'vjs'  : assDir / 'vendors_js.zip'}
