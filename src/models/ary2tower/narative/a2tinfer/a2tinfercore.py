#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-08-01"

"""
a2tinfercore.py
_________________________________________
StaticInferenceDashboard: a thin class-based wrapper around
a2trearender.py's generate_inference_report(), for callers who prefer
constructing a reusable dashboard object (e.g. one bound to a fixed
output_dir) over calling the module-level function directly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from .....configs import logger
except ImportError:  # pragma: no cover - fallback for standalone/test use
    logger = logging.getLogger(__name__)

from ..a2trearender import generate_inference_report


class StaticInferenceDashboard:
    """Generates a static (pre-rendered, non-live) HTML inference
    dashboard for ary2tower. All the actual context-building/rendering
    logic lives in a2trearender.py -- this class only holds a default
    `output_dir` across multiple `.generate()` calls."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = output_dir

    def generate(self, context_data: Dict[str, Any],
                output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Render `context_data` to disk. `output_dir` overrides the
        instance default for this one call."""
        target = output_dir if output_dir is not None else self.output_dir
        logger.info("StaticInferenceDashboard.generate() -> %s", target)
        return generate_inference_report(context_data, output_dir=target)
