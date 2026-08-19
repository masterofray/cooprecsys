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


"""
Dynamic Recommendation Dashboard Renderer
Enterprise-grade HTML dashboard generator for:
- Recommendation Systems
- Learning-to-Rank
- Monitoring Reports
- ML Evaluation Dashboards
"""

import os
import re
import json
import math
import random
import pandas as pd
from tqdm.auto import tqdm
from pathlib  import Path
from copy     import deepcopy
from datetime import datetime
from typing   import Any, Dict, List, Optional
from jinja2   import (Environment, FileSystemLoader,
                     TemplateNotFound, select_autoescape)

from .renderutils import (get_env, static_prefix, load_context, 
    load_prediction_dataframe, generate_scorecards, generate_gauges, 
    generate_stat_minis, normalize_charts)

LocDir = Path(__file__).resolve()
from ....configs import logger, _cfg
from ....prepare import latest_found, FileCopier

rpath        = _cfg.get('PATHS', 'html_report_path')
STATIC_DIR   = LocDir.parent / "static"
IMG_DIR      = LocDir.parents[4] / 'img'
OUTPUT_DIR   = (LocDir.parents[4] / rpath).parents[0]
DEFAULT_CONTEXT_PATH = OUTPUT_DIR / "contextRecsys.json"


def copymaps(mapdict : Dict, 
             goal    : Path,
             dirlist : List = ['css', 'js', 'icon'],
            ) -> Dict:
    for fdr in tqdm(dirlist,
                    desc   = 'Directory Dest',
                    colour = _cfg.get('tqdm', 'colour'),
                    ncols  = _cfg.getint('tqdm', 'ncols'),
                    unit   = 'dir'):
        try:
            xdir  = STATIC_DIR / fdr
            if xdir.exists():
                for item in xdir.glob('*'):
                    if item.is_file():
                        mapdict[item] = goal/fdr
        except Exception as err:
            logger.error(err)
    return mapdict

def runcopy(dest: Path = None) -> None:
    dest     = OUTPUT_DIR / 'assets' if dest is None else Path(dest).resolve()
    copy_map = {STATIC_DIR / 'compute01.png': dest}
    copy_map[IMG_DIR/'favicon.ico']   = dest
    copy_map[IMG_DIR/'logo_red.jpg']  = dest
    cpnow = copymaps(copy_map, dest)
    for src, dst_dir in cpnow.items():
        _ = FileCopier(Scrpath = src,
                       Destdir = dst_dir)

# CONTEXT BUILDER
#__________________________________________________________
def ModifDF(content):
    RawDF            = load_prediction_dataframe(content)
    SubjectID        = _cfg.get('SHAP', 'SubjectID')
    MaxColumnFeature = _cfg.getint('SHAP', 'MaxFeatTabel')
    Columns = RawDF.columns.tolist()
    Columns.remove('relevance_score')
    Columns.remove('rank')
    if SubjectID in Columns:
        Columns.remove(SubjectID)
    NewColumn = random.sample(Columns, MaxColumnFeature - 2)
    NewColumn.insert(0, SubjectID)
    NewColumn.extend(['relevance_score', 'rank'])
    modifdata = deepcopy(RawDF[NewColumn])
    return modifdata, RawDF

def prettify(col : str) -> str:
    ACRONYMS = {"Id": "ID", "Enc": "ENC", "Sku": "SKU",
                "Url": "URL", "Api": "API", "Json": "JSON",
                "Html": "HTML",  "Xml": "XML"}
    col = col.replace("_", " ")
    col = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', col)
    col = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', col)
    col = re.sub(r'\s+', ' ', col).strip()
    col = col.title()
    words = [ACRONYMS.get(w, w)for w in col.split()]
    return " ".join(words)


def build_context(json_path: str | Path) -> Dict[str, Any]:
    """Build full dashboard rendering context."""
    logger.debug("Entering build_context().")
    try:
        if Path(json_path).exists():
            context = load_context(json_path)
        else:
            json_path = latest_found(dir  = LocDir.parents[4],
                                keyword   = 'Recsys',
                                recursive = True,
                                Not4Json  = False)
            context   = load_context(json_path)
        logger.info(f'The path is {str(json_path)}.')
        
        metrics = context.get("metrics", dict())
        logger.debug("Metrics count = %d", len(metrics))
        Pred2HTML, Raw = ModifDF(context)
        rawcolumn = Pred2HTML.columns.tolist()
        pretcolum = {c: prettify(c) for c in rawcolumn}
        
        context["scorecards"]     = generate_scorecards(metrics)
        context["gauges"]         = generate_gauges(metrics)
        context["stat_minis"]     = generate_stat_minis(Raw)
        context["charts"]         = normalize_charts(context.get("charts", []))
        context["thecolumn"]      = pretcolum
        context["rankings"]       = Pred2HTML.to_dict(orient="records")
        context["total_rankings"] = int(Pred2HTML.shape[0])
        context["bar_labels"]     = list(metrics.keys()) if metrics else []
        #context["bar_values"]    = list(metrics.values()) if metrics else []
        context["bar_data"]       = list(metrics.values()) if metrics else []

        context.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        context.setdefault("page_title", "LTR Dashboard")
        context.setdefault("subtitle", "Learning to Rank Monitoring")
        context.setdefault("experiment_name", "Default Experiment")
        context.setdefault("best_iteration", "N/A")
        
        # Calculate overall score percent for main gauge
        try:
            mtep = context["bar_data"][4:]
            osv  = round(sum(mtep)/len(mtep) * 100, 2)
            context["overall_score_percent"] = osv
        except (ValueError, TypeError):
            context["overall_score_percent"] = 0
        context['overall_score'] = f'{osv}%'
        
        context.setdefault("tuner_summary", None)
        context.setdefault("training_params", dict())
        logger.info("Dashboard context built successfully.")
        return context

    except Exception as exc:
        logger.error("Context building failed.", exc_info = True)
        raise RuntimeError("Failed building dashboard context.") from exc


# HTML RENDERER
#__________________________________________________________
def render_report(context: Dict[str, Any], output_path: str | Path) -> str:
    """Render dashboard HTML."""
    logger.debug("Entering render_report().")
    try:
        env = get_env()
        logger.debug("Loading template=base.html.j2")
        template = env.get_template("base.html.j2")
        ctx = dict(context)
        
        logger.debug("Injecting static asset paths.")
        ctx.update(static_prefix(output_path))
        
        logger.debug("Rendering HTML template.")
        html = template.render(**ctx)
        
        logger.info("Dashboard rendered successfully.")
        logger.debug("Rendered HTML size=%.2f KB", len(html.encode("utf-8")) / 1024)
        return html

    except TemplateNotFound as exc:
        logger.error("Template base.html.j2 missing.", exc_info=True)
        raise RuntimeError("Dashboard template missing.") from exc

    except Exception as exc:
        logger.error("Unexpected rendering failure.", exc_info=True)
        raise RuntimeError("Dashboard rendering failed.") from exc

def repot(ContPath : Path = './') -> str:
    """Main dashboard generation pipeline."""
    logger.info("Starting dashboard generation pipeline.")
    try:
        OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
        logger.debug("Output directory ensured=%s", OUTPUT_DIR)
        datepf      = datetime.now().strftime("%Y%m%d")
        output_path = OUTPUT_DIR / f"{datepf}_training_report.html"
        if Path(ContPath).is_file():
            context = build_context(ContPath)
        else:
            context = build_context(DEFAULT_CONTEXT_PATH)
        
        dlevel = True if _cfg.get('logging', 'level') in ['DEBUG', 'INFO'] else False
        if dlevel:
            with open(OUTPUT_DIR/'RunContext.json', 'w', encoding='utf-8') as fx:
                json.dump(context, fx, ensure_ascii = False, indent = 2)
            
        html        = render_report(context = context, output_path = output_path)
        logger.debug("Writing rendered HTML to disk.")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html)
        logger.debug("Output HTML path=%s .\n\n", output_path)

        runcopy()
        logger.info("Dashboard generated successfully.")
        logger.info("Dashboard path   = %s", output_path)
        logger.info("Total rankings   = %d", context.get("total_rankings", 0))
        logger.info("Total scorecards = %d", len(context.get("scorecards", [])))
        logger.info("Total charts     = %d", len(context.get("charts", [])))
        return output_path
    except Exception as exc:
        logger.error("Dashboard generation pipeline failed.", exc_info=True)
        raise RuntimeError("Dashboard generation failed.") from exc


if __name__ == "__main__":
    repot()