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


import numpy   as np
from pathlib   import Path
from tqdm.auto import tqdm
from typing    import Dict, List, Optional
from jinja2    import Environment, FileSystemLoader, select_autoescape

#sys.path.append(str(LocDir.parents[2]))
from ....configs   import logger, _cfg
from ....assets    import (VendorPath, bealabel, safe_float,
                           detect_gauge_metric)
from ....prepare   import FileCopier

LocDir     = Path(__file__).parent.resolve()
Tplatedir  = LocDir / "templates"
advdir     = LocDir / "sttrain"
readir     = LocDir / "stinferc"
IMG_DIR    = LocDir.parents[2] / 'assets' / 'icon'
OUTPUT_DIR = (LocDir.parents[4] / _cfg.get('PATHS', 'ACB_rpath') / 'ACB_reports').resolve()

def get_env() -> Environment:
    """Initialize Jinja2 environment."""
    logger.debug("Initializing Jinja environment.")
    try:
        env = Environment(
              loader        = FileSystemLoader(str(Tplatedir)),
              autoescape    = select_autoescape(
                              enabled_extensions = ("html", "xml"),
                              default_for_string = True,
                              default            = False),
              trim_blocks   = True,
              lstrip_blocks = True)
        logger.debug("Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing Jinja environment.", exc_info = True)
        raise RuntimeError("Failed to initialize Jinja environment.") from exc


def copymaps(mapdict : Dict, 
             goal    : Path,
             train   : bool = True,
             dirlist : Optional[List] = None,
            ) -> Dict:
    dirlist       = dirlist if dirlist is not None else ['css', 'js']
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
           ) -> Dict[str, str]:
    """
    Copy vendor CSS/JS/icon assets (plus favicon/logo) into `dest` so the
    generated HTML report is self-contained.

    BUGFIX: this used to have no `return` statement at all (implicit
    `None`), even though its own type hint / every caller's variable name
    ("static_paths") assumed it handed back something. rearender.py did
    `context.update(static_paths)` on that `None` and crashed with
    `TypeError: 'NoneType' object is not iterable`. Copying itself was
    always working fine (side effect below) -- only the return value was
    missing. Now returns the copied assets' paths, relative to `dest`,
    so callers can safely merge it into a Jinja context.
    """
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

    # NOTE: the exact keys below ("static_css", "static_js") match what
    # templates/infrc_base.html.j2 (and train_base.html.j2) reference via
    # {{ static_css }}/{{ static_js }} -- confirmed by reading the actual
    # template source, not guessed.
    return {"static_css"      : "css",
            "static_js"       : "js",
            "static_icon_dir" : "icon",
            "favicon_path"    : "favicon.ico",
            "logo_path"       : "logo_red.jpg"}


if __name__ == '__main__':
    logger.info('Rensupport test to be here!\n')
    runcopy(advisor = True)