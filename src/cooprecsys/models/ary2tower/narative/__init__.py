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

from .a2trearender import (build_inference_context, generate_inference_report,
                           render_inference_report)
from .a2tadvirender import (build_training_context, generate_training_report,
                            render_training_report)
from .a2tinfer import StaticInferenceDashboard
from .a2ttrain import StaticTrainingDashboard

__all__ = ['build_inference_context', 'generate_inference_report', 'render_inference_report',
           'build_training_context', 'generate_training_report', 'render_training_report',
           'StaticInferenceDashboard', 'StaticTrainingDashboard',
           ]
