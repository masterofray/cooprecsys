#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.1.0"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
rensupport.py
_________________________________________
Shared Jinja2 environment + static-asset-copying utilities for
ary2tower's narative dashboard. Mirrors the role of
arycolbring/narative/rensupport.py, simplified for this module's
lighter structure: ONE shared static/{css,js}/ tree (not separate
per-mode trees), and a single template per mode (not arycolbring's
tabbed multi-template system) -- see narative/templates/.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from ....configs import logger, _cfg
except ImportError:  # pragma: no cover - fallback for standalone/test use
    logger = logging.getLogger(__name__)
    _cfg = None

LocDir     = Path(__file__).parent.resolve()
Tplatedir  = LocDir / "templates"
StaticDir  = LocDir / "static"


def _default_output_dir() -> Path:
    """Reuses the same [PATHS] output_dir config key arycolbring's own
    renderers read from, appending an ary2tower-specific subfolder so
    the two modules' reports never collide on disk."""
    base = "artifacts"
    if _cfg is not None:
        try:
            base = _cfg.get("PATHS", "output_dir", fallback="artifacts")
        except Exception:
            pass
    return (LocDir.parents[3] / base / "ary2tower_reports").resolve()


OUTPUT_DIR = _default_output_dir()


def get_env() -> Environment:
    """Initialize the Jinja2 environment for narative/templates/.

    NOTE: this function deliberately does NOT swallow exceptions in a
    `finally: return env` (that exact bug was found and fixed in
    arycolbring/narative/rensupport.py's get_env() during the repo-wide
    bug scan earlier this session -- see CHANGELOG.md). A failure here
    raises a RuntimeError, as it should.
    """
    logger.debug("Initializing ary2tower Jinja environment.")
    try:
        env = Environment(
              loader        = FileSystemLoader(str(Tplatedir)),
              autoescape    = select_autoescape(
                              enabled_extensions = ("html", "xml"),
                              default_for_string = True,
                              default            = False),
              trim_blocks   = True,
              lstrip_blocks = True)
        logger.debug("ary2tower Jinja environment initialized successfully.")
        return env
    except Exception as exc:
        logger.error("Failed initializing ary2tower Jinja environment.", exc_info=True)
        raise RuntimeError("Failed to initialize ary2tower Jinja environment.") from exc


def copy_static(goal: Path) -> Dict[str, Path]:
    """Copy narative/static/{css,js}/ into `goal` (the report's output
    directory), overwriting any existing copy there. Returns the
    {css: path, js: path} destination map."""
    goal = Path(goal)
    goal.mkdir(parents=True, exist_ok=True)
    destinations: Dict[str, Path] = dict()
    for subdir in ("css", "js"):
        src = StaticDir / subdir
        dst = goal / "static" / subdir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            destinations[subdir] = dst
            logger.debug("Copied %s -> %s", src, dst)
        else:
            logger.warning("Static asset dir missing, skipped: %s", src)
    return destinations
