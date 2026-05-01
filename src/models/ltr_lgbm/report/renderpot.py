#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-01"

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape


BASE_DIR     = Path(__file__).resolve()
TEMPLATE_DIR = BASE_DIR.parents[0] / "templates"
STATIC_DIR   = BASE_DIR.parents[0] / "static"
sys.path.append(str(BASE_DIR.parents[3]))
from configs import logger


# _____________________________________________________
# Build compact Helper function
# _____________________________________________________

def get_env() -> Environment:
    return Environment(
        loader        = FileSystemLoader(TEMPLATE_DIR),
        autoescape    = select_autoescape(["html", "xml"]),
        trim_blocks   = True,
        lstrip_blocks = True)

def _static_prefix(directory: Optional[str | Path] = None) -> Dict[str, str]:
    """Compute relative css/js paths from output HTML file.
    """
    logger.debug('Process in `_static_prefix`')
    if directory is None:
        rel_static = Path("static")
    else:
        output_dir = Path(directory).resolve().parent
        rel_static = Path(os.path.relpath(STATIC_DIR, output_dir))
    return {"static_css": (rel_static / "css").as_posix(),
            "static_js" : (rel_static / "js").as_posix()}


# _____________________________________________________
# Build main Function
# _____________________________________________________

def render_report(
        context     : Dict[str, Any],
        output_path : Optional[str | Path] = None,
    ) -> str:
    logger.debug('Begin to render Template Report.')
    env      = get_env()
    template = env.get_template("monitoring_report.html.j2")
    context  = dict(context)
    context.update(_static_prefix(output_path))
    logger.debug('Template report already done, sent to vizseason.py!')
    return template.render(**context)

if __name__ == '__main__':
    pass