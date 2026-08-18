#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-31"

"""
dashboard_components.py
_________________________________________
Reusable elements for embedding viztower figures into a report/notebook.

This is deliberately NOT a second dashboard theme/template system.
`arycolbring/narative/` already has one (light + orange, Jinja2 +
Plotly, see Task 1) -- ary2tower's *interactive* dashboard reuses that
one directly (see report.py, which calls
arycolbring.narative.rearender.generate_inference_report()). This
module only covers the simpler, static-image case: embedding a
matplotlib Figure as a `<img>` tag (in a notebook, an email, a
lightweight standalone HTML snippet, etc.) where pulling in the full
Jinja2/CSS/JS dashboard stack would be overkill.
"""

import base64
from io import BytesIO
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ....configs import logger
except ImportError:  # pragma: no cover - fallback for standalone/test use
    import logging
    logger = logging.getLogger(__name__)


def fig_to_base64_png(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string
    (no data: prefix -- see fig_to_data_uri for that)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    logger.debug("Encoded figure to base64 PNG (%d chars).", len(encoded))
    return encoded


def fig_to_data_uri(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib Figure to a `data:image/png;base64,...` URI,
    directly usable as an `<img src="...">` value."""
    return f"data:image/png;base64,{fig_to_base64_png(fig, dpi=dpi)}"


def figs_to_html_gallery(figs: Dict[str, plt.Figure], dpi: int = 120,
                          title: Optional[str] = None) -> str:
    """Wrap a {label: Figure} dict into a minimal, dependency-free HTML
    snippet (no external CSS/JS/CDN calls) -- one <img> per figure,
    labeled. For the full interactive dashboard experience, use
    report.py instead."""
    logger.debug("Building HTML gallery for %d figure(s).", len(figs))
    parts = ['<div style="font-family: -apple-system, Arial, sans-serif;">']
    if title:
        parts.append(f'<h2 style="color:#212529;">{title}</h2>')
    for label, fig in figs.items():
        uri = fig_to_data_uri(fig, dpi=dpi)
        parts.append(
            '<div style="margin-bottom:24px;">'
            f'<h3 style="color:#495057;font-size:0.95rem;">{label}</h3>'
            f'<img src="{uri}" style="max-width:100%;border:1px solid #DEE2E6;'
            'border-radius:8px;">'
            '</div>')
    parts.append("</div>")
    return "\n".join(parts)
