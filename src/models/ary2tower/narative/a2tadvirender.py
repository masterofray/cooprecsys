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
a2tadvirender.py
_________________________________________
ary2tower's training dashboard renderer. Same shape as
arycolbring/narative/advirender.py -- unlike a2trearender.py (the
inference report), ranking-quality metrics (precision/recall/ndcg/
auc/mrr) ARE expected and correctly displayed here as gauges, since
this is the training/evaluation report.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from ....configs import logger
from .rensupport import get_env, copy_static, OUTPUT_DIR

try:
    from ....assets import (generate_scorecards as _gen_scorecards,
                            generate_gauges, normalize_charts, overall_score_percent)
except ImportError:  # pragma: no cover - fallback for standalone/test use
    from src.assets import (generate_scorecards as _gen_scorecards,
                            generate_gauges, normalize_charts, overall_score_percent)


def generate_scorecards(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _gen_scorecards(metrics, sub_label="Training metric")


def build_training_context(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the full ary2tower training-report context."""
    logger.debug("Entering ary2tower build_training_context().")
    try:
        context = deepcopy(context_data)
        metrics = context.get("metrics", {})

        context["scorecards"] = generate_scorecards(metrics)
        context["gauges"] = generate_gauges(metrics)
        context["charts"] = normalize_charts(context.get("charts", []))
        context["overall_score_percent"] = overall_score_percent(metrics)

        context.setdefault("loss_history", [])
        context.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title", "ary2tower Training Dashboard")
        context.setdefault("subtitle", "Two-Tower Model — Training Run")
        context.setdefault("experiment_name", "Default Experiment")
        context.setdefault("theme_css", "supporttrain.css")
        context.setdefault("theme_js", "supporttrain.js")

        logger.info("ary2tower training context built: %d metric(s), %d gauge(s).",
                    len(metrics), len(context["gauges"]))
        return context
    except Exception as exc:
        logger.error("ary2tower training context building failed.", exc_info=True)
        raise RuntimeError("Failed building ary2tower training context.") from exc


def render_training_report(context: Dict[str, Any], output_path: Path) -> str:
    """Render + write the training HTML report to `output_path`."""
    env = get_env()
    template = env.get_template("a2t_training.html")
    html = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    copy_static(output_path.parent)
    return html


def generate_training_report(context_data: Dict[str, Any],
                             output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Build the context and render the training report to disk.
    Returns the written file's path."""
    context = build_training_context(context_data)
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_ary2tower_Training_Report.html"
    output_path = out_dir / filename
    render_training_report(context, output_path)
    logger.info("ary2tower training report written to %s", output_path)
    return output_path
