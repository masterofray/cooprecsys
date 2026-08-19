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
a2ttraincore.py
_________________________________________
StaticTrainingDashboard: a thin class-based wrapper around
a2tadvirender.py's generate_training_report(), mirroring
a2tinfer/a2tinfercore.py's structure.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from .....configs import logger
from ..a2tadvirender import generate_training_report


class StaticTrainingDashboard:
    """Generates a static HTML training dashboard for ary2tower. All
    the actual context-building/rendering logic lives in
    a2tadvirender.py -- this class only holds a default `output_dir`
    across multiple `.generate()` calls."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = output_dir

    def generate(self, context_data: Dict[str, Any],
                output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Render `context_data` to disk. `output_dir` overrides the
        instance default for this one call."""
        target = output_dir if output_dir is not None else self.output_dir
        logger.info("StaticTrainingDashboard.generate() -> %s", target)
        return generate_training_report(context_data, output_dir=target)
