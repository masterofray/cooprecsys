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
vizseason.py
____________________________
Publication-quality visualisations and a self-contained HTML monitoring
report for the LTR pipeline. All plot artifacts (PNG) are saved under 
``config.path.output_dir``. The HTML report bundles every chart as a 
base64 data-URI so it requires no external assets and can be shared 
as a single file.
____________________________
Design notes
    * All state stored on ``self``; no public ``return`` from plot methods.
    * Figures are closed after saving to prevent memory leaks in long runs.
    * HTML report is rendered via an f-string template (zero Jinja2 dependency).
"""

import io
import re
import os
import sys
import shap
import base64
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from ipdb import set_trace

matplotlib.use("Agg")
LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))

from prepare import Dict2Json, FLmiss
from configs import LTRConfig, logger, _cfg
from models.ltr_lgbm.report import repot

# ---------------------------------------------------------------------------
# Seaborn global theme
# ---------------------------------------------------------------------------

sns.set_theme(
    style   = "whitegrid",
    palette = "muted",
    rc      = {"figure.dpi": 120, "axes.titlesize": 14, "axes.labelsize": 12},
)

vintages = [
    "#0f5403",  # Dark Green (Primary) - matches your text color
    "#1a7a0a",  # Medium Green (Secondary)
    "#28a745",  # Bright Green (Success)
    "#85c97d",  # Light Green (Accent)
]

MLPstyle = {
    # Background - Your specified light background
    "figure.facecolor"  : "#e6ebf2",
    "axes.facecolor"    : "#e6ebf2",
    "savefig.facecolor" : "#e6ebf2",
    
    # Grid - Subtle grid with green tint
    "axes.grid"         : True,
    "grid.color"        : "#c5d5c0",  # Light green-gray for grid
    "grid.linestyle"    : "-",
    "grid.linewidth"    : 0.8,
    
    # Typography - Your specified green text
    "text.color"        : "#0f5403",
    "axes.labelcolor"   : "#0f5403",
    "xtick.color"       : "#0f5403",
    "ytick.color"       : "#0f5403",
    "axes.titlecolor"   : "#0f5403",
    "axes.titlesize"    : 17,
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 14,
    "font.size"         : 10,
    
    # Spines - Clean L-frame with green borders
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : True,
    "axes.spines.bottom": True,
    "axes.edgecolor"    : "#0f5403",
    "axes.linewidth"    : 1.2,
    
    # Data Point Styling - Green theme throughout
    "axes.prop_cycle"   : plt.cycler(color = vintages),
    "lines.linewidth"   : 2.2,
    "lines.markersize"  : 8,
    "patch.edgecolor"   : "#0f5403",
    "patch.force_edgecolor": True,
}

plt.rcParams.update(MLPstyle)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode a Matplotlib figure to a base64 PNG string (data-URI ready)."""
    buf = io.BytesIO()
    fig.savefig(buf, format = "png", 
                bbox_inches = "tight", 
                dpi = 200)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def _save_fig(fig: plt.Figure, path: str = './output.png') -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    fig.savefig(path, 
                bbox_inches = "tight", 
                dpi = 300)
    plt.close(fig)
    logger.debug("Figure saved: %s", path)

def cfglist(section:str, option:str, numeric = True):
    raw_value = _cfg.get(section, option)
    clean     = raw_value.strip("[]")
    if numeric:
        return [int(x.strip()) for x in clean.split(",") if x.strip()]
    else:
        return [x.strip().strip("'\"") for x in clean.split(",") if x.strip()]

def safe_array(arr):
    return np.nan_to_num(
        np.asarray(arr),
        nan    = 0.0,
        posinf = 0.0,
        neginf = 0.0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class Visualizer:
    """Generate all visual artefacts for the LTR pipeline.
    config       : type `LTRConfig` master config.
    model        : Trained :class:`lgb.Booster`.
    evals_result : Training metric history dict (from :class:`LTRTrainer`).
    X_test       : Test feature matrix (for prediction distribution plots).
    metrics      : Flat summary metrics dict (from :class:`LTRTrainer`).
    """
    def __init__(self,
            config:       LTRConfig,
            model:        lgb.Booster,
            evals_result: Dict[str, Any],
            X_test:       np.ndarray,
            metrics:      Dict[str, float],
        ) -> None:
        self._config       = config
        self._model        = model
        self._evals_result = evals_result
        self._X_test       = X_test
        self._metrics      = metrics
        self._chartsdata   = list()
        self._predictdata  = pd.DataFrame([])
        self._SubjectID    = _cfg.get('SHAP', 'SubjectID')
        self._b64_images:  Dict[str, str]  = dict()
        self._saved_paths: Dict[str, str]  = dict()
        logger.debug("Visualizer initialised.")

    @property
    def output_dir(self) -> str:
        return self._config.path.output_dir

    @property
    def feature_names(self) -> List[str]:
        return self._config.feature.features

    def _record(self, 
                name: str, 
                fig: plt.Figure,
        ) -> None:
        """Save figure to PNG, encode to base64, store in state dicts."""
        path = os.path.join(self.output_dir, f"{name}.png")
        _save_fig(fig, path)
        self._saved_paths[name] = path
        #with open(path, "rb") as f:
        #    self._b64_images[name] = base64.b64encode(f.read()).decode("utf-8")
        logger.info("Chart saved: %s", path)
    
    def _get_SubjectID(self, sample_idx : int) -> str:
        value = self._predictdata.iloc[sample_idx][self._SubjectID]
        if isinstance(value, (int, float)):
            return str(int(value))
        return str(value)

    def _get_CustData(self, cust_id: Any) -> pd.DataFrame:
        """
        Retrieve filtered data for a specific customer with columns specified in config.
        Falls back to random column selection prioritized by numeric/relevant keywords
        (price, total, sum, count, etc.) if no requested columns are available.
        Always drops the SubjectID column from the result.
        """
        final_cols: List[str] = list()
        max_cols : int          = _cfg.getint('SHAP', 'MaxFeatTabel')
        mask     : pd.Series    = self._predictdata[self._SubjectID].astype(str) == str(cust_id)
        data_cust: pd.DataFrame = deepcopy(self._predictdata[mask])
        if data_cust.empty:
            raise ValueError(f"No data found for CustomerID '{cust_id}'.")
        data_cust = data_cust.drop(columns = [self._SubjectID], errors = 'ignore')

        try:
            desired_cols = cfglist(section = 'SHAP', 
                                   option  = 'FeatTableColumn', 
                                   numeric = False)
            assert isinstance(desired_cols, list), "cfglist did not return a list"
            available_desired = [item for item in desired_cols if item in data_cust.columns]
            final_cols        = available_desired[:max_cols]
        except Exception as err:
            raise RuntimeError(
            f"Failed to read columns from config ['SHAP'] 'FeatTableColumn': {err}")

        # Fallback: if no requested columns are available, pick randomly
        # with priority given to numeric/relevant keywords
        cfrandom = _cfg.getboolean('SHAP', 'UseRandomColumn')
        if not final_cols or cfrandom :
            all_cols: List[str]      = data_cust.columns.tolist()
            if not all_cols:
                raise ValueError("No feature columns remain after dropping SubjectID.")
            priority_pattern: str    = (
                r'price|total|sum|count|amount|qty|quantity|revenue|sales|'
                r'profit|discount|fee|tax|cost|value|weight')
            pattern: re.Pattern      = re.compile(priority_pattern, re.IGNORECASE)
            priority_cols: List[str] = [c for c in all_cols if pattern.search(c)]
            other_cols: List[str]    = [c for c in all_cols if not pattern.search(c)]
            random.shuffle(priority_cols)
            random.shuffle(other_cols)

            # Combine priority first, then others; cap at max_cols
            final_cols = (priority_cols + other_cols)[:max_cols]
            assert final_cols, "Fallback random selection produced no columns."
        dataCst = FLmiss(data = data_cust[final_cols])
        return dataCst


    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------
    def plot_feature_importance(
        self,
        importance_type : str = "gain",
        top_n           : int = 30,
    ) -> None:
        """Bar chart of LightGBM feature importances.
        importance_type : "gain" (default) or "split".
        top_n           : Maximum number of features to display.
        """
        logger.debug("Plotting feature importance (type=%s, top_n=%d).",
                     importance_type, top_n)
        imp      = self._model.feature_importance(importance_type = importance_type)
        names    = self.feature_names
        dataplot = pd.DataFrame({"Feature": names, "Importance": imp})\
                   .sort_values("Importance", ascending = False)\
                   .head(top_n)
        fig, ax  = plt.subplots(figsize=(11, max(5, top_n * 0.35)))
        sns.barplot(
            data    = dataplot,
            x       = "Importance",
            y       = "Feature",
            hue     = "Feature",
            palette = "Blues_r",
            legend  = False,
            ax      = ax)
        ax.set_title(f"Feature Importance - {importance_type.capitalize()} (Top {top_n})")
        ax.set_xlabel(f"Importance ({importance_type})")
        ax.set_ylabel("")
        
        topIM       = _cfg.getint('FEATURES', 'top_importances')
        datapltlist = np.nan_to_num(dataplot["Importance"], nan = 0.0,
                                    posinf = 0.0, neginf = 0.0).tolist()
        chart_info  = {'title'   : f'Top {topIM} of Feature Importance by {importance_type.upper()}',
                       'type'    : 'bar_chart_js',
                       'data'    : {'labels': dataplot["Feature"].tolist()[:topIM],
                                    'values': datapltlist[:topIM]},
                       'image'   : None,
                       'full'    : False}
        self._chartsdata.append(chart_info)
        self._record("feature_importance", fig)


    def plot_prediction_distribution(self) -> None:
        """Histogram + KDE of test-set relevance scores."""
        logger.debug("Plotting prediction distribution.")
        preds = self._model.predict(self._X_test,
                num_iteration = self._model.best_iteration or 0,)
        fig, ax = plt.subplots(figsize=(9, 5))
        edges_scott = np.histogram_bin_edges(preds, bins='scott')
        numbin = _cfg.getint('MODEL_LGBM', 'num_histbin')
        if numbin >= len(edges_scott):
            usebin = np.histogram_bin_edges(preds, bins = numbin)
        else:
            usebin = edges_scott
        sns.histplot(preds, 
                     bins  = usebin, 
                     kde   = True, 
                     color = "#4C72B0", 
                     ax    = ax)
        ax.set_title("Relevance Score Distribution (Test Set)")
        ax.set_xlabel("Predicted Relevance Score")
        ax.set_ylabel("Count")
        summary_stats = {
            'mean': float(np.mean(preds)),
            'std' : float(np.std(preds)),
            'min' : float(np.min(preds)),
            'max' : float(np.max(preds)),
            'n'   : int(len(preds))
        }
        stats_text = (
            f"mean = {summary_stats['mean']:.4f}\n"
            f"std = {summary_stats['std']:.4f}\n"
            f"min = {summary_stats['min']:.4f}\n"
            f"max = {summary_stats['max']:.4f}"
        )
        ax.text(0.97, 0.97, stats_text,
            transform   = ax.transAxes,
            va          = "top",
            ha          = "right",
            fontsize    = 10,
            family      = "monospace",
            bbox        = dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        histpredc   = np.nan_to_num(preds, nan = 0.0,
                                    posinf = 0.0, neginf = 0.0).tolist()
        bins        = usebin.tolist()
        chart_info  = {'title'  : 'Histogram of test-set relevance scores',
                       'type'   : 'histrelevance',
                       'data'   : {'values' : histpredc,
                                   'bins'   : bins,
                                   'axis'   : {'x_label': 'Predicted Relevance Score',
                                               'y_label': 'Count'},
                                   'stats'     : summary_stats,
                                   'stats_text': stats_text},
                        'image' : None,
                        'full'  : False}
        self._chartsdata.append(chart_info)
        self._record("prediction_distribution", fig)


    def plot_learning_curves(self) -> None:
        """Line chart of train/test NDCG over boosting rounds."""
        logger.debug("Preparing learning curves chart data.")
        lines = list()
        fig, axes = plt.subplots(1, 1, figsize = (10, 5))
        for split_name, metric_dict in self._evals_result.items():
            for metric_name, values in metric_dict.items():
                label     = f"{split_name} - {metric_name}"
                linestyle = "solid" if split_name == "train" else "dash"
                pltstyle  = "solid" if split_name == "train" else "dashed"
                y_values  = [float(v) for v in values]
                x_values  = list(range(len(y_values)))
                lines.append({
                    "label"      : label,
                    "x"          : x_values,
                    "y"          : y_values,
                    "line_style" : linestyle,
                    "metric"     : metric_name,
                    "split"      : split_name,})
                axes.plot(
                    values,
                    label     = label,
                    linestyle = pltstyle,
                    linewidth = 1.8)

        best_iter = getattr(self._model, "best_iteration", None)
        best_iteration_info = None
        if best_iter is not None:
            best_iteration_info = {
                "x"          : int(best_iter),
                "line_style" : "dot",
                "label"      : f"Best iteration ({best_iter})"}
            axes.axvline(
                x         = best_iter,
                color     = "red",
                linestyle = ":",
                linewidth = 1.5,
                label     = f"Best iteration ({best_iter})")
        axes.set_title("Learning Curves (NDCG per Round)")
        axes.set_xlabel("Boosting Round")
        axes.set_ylabel("NDCG")
        axes.legend(loc="lower right", fontsize=9)
    
        summary_stats = dict()
        for split_name, metric_dict in self._evals_result.items():
            for metric_name, values in metric_dict.items():
                arr = np.asarray(values, dtype=float)
                summary_stats[
                f"{split_name}_{metric_name}"] = {
                    "min"  : float(arr.min()),
                    "max"  : float(arr.max()),
                    "last" : float(arr[-1]),
                    "best" : float(arr.max())}
    
        chart_info = {
            "title" : "Learning Curves (NDCG per Round)",
            "type"  : "learning_curve",
            "data"  : {"lines" : lines,
                       "best_iteration"    : best_iteration_info,
                       "axis" : {"x_label" : "Boosting Round",
                                 "y_label" : "NDCG"},
                       "stats" : summary_stats},
            "image" : None,
            "full"  : False}
        self._chartsdata.append(chart_info)
        self._record("learning_curves", fig)


    def plot_metrics_summary(self) -> None:
        """Horizontal bar chart summarising key evaluation metrics."""
        logger.debug("Plotting metrics summary.")
        # Filter to NDCG and prediction stats only
        filtered = {k: v for k, v in self._metrics.items()
                    if "ndcg" in k.lower() or "pred" in k.lower()}
        if not filtered:
            logger.warning("No metrics found for summary chart.")
            return None
        tempdata = pd.DataFrame(list(filtered.items()),
                    columns = ["Metric", "Value"],
                    ).sort_values("Value", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(3, len(tempdata) * 0.55)))
        bars = ax.barh(
            tempdata["Metric"], tempdata["Value"],
            color     = "#55A868",
            edgecolor = "white",
            height    = 0.6)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_title("Evaluation Metrics Summary")
        ax.set_xlabel("Value")
        ax.set_xlim(0, max(tempdata["Value"]) * 1.18)
        self._record("metrics_summary", fig)


    def plot_feature_correlation(self, sample_n: int = 5_000) -> None:
        """Heatmap of pairwise feature correlations on a random sample.
        sample_n: Maximum rows to sample before computing the correlation matrix
                  (guards against OOM on huge test sets).
        """
        logger.debug("Plotting feature correlation heatmap.")
        n       = min(sample_n, self._X_test.shape[0])
        idx     = np.random.choice(self._X_test.shape[0], size=n, replace=False)
        sample  = pd.DataFrame(self._X_test[idx], columns=self.feature_names)
        corr    = sample.corr()
        size    = max(8, len(self.feature_names) * 0.55)
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            corr,
            ax          = ax,
            cmap        = "RdBu_r",
            center      = 0,
            linewidths  = 0.3,
            square      = True,
            annot       = len(self.feature_names) <= 20,
            fmt         = ".2f",
            annot_kws   = {"fontsize": 7},
        )
        ax.set_title("Feature Correlation Matrix")
        self._record("feature_correlation", fig)
        corr_js     = np.nan_to_num(corr.values, nan = 0.0,
                                    posinf = 0.0, neginf = 0.0).tolist()
        chart_info  = {'title'  : 'Correlation Matrix',
                       'type'   : 'heatmap_plotly',
                       'data'   : {'z': corr_js,
                                   'x': list(map(str, self.feature_names)),
                                   'y': list(map(str, self.feature_names))},
                        'image' : None,
                        'full'  : False}
        self._chartsdata.append(chart_info)


    def shap_waterfall_chart(self):
        """
        Menambahkan beberapa SHAP waterfall chart
        ke self._chartsdata agar dapat dirender
        di HTML/Jinja2 + JavaScript.
        """
        sample_indices = cfglist(section = 'SHAP', option = 'rowID')
        max_display    = min(_cfg.getint('SHAP', 'numfeats'), 15)
        Columns        = deepcopy(self.feature_names)
        explainer      = shap.TreeExplainer(
            model                 = self._model,
            data                  = self._X_test,
            feature_perturbation  = "interventional",
            model_output          = "raw",
            approximate           = False)
        shap_values = explainer(self._X_test, check_additivity = False)

        for sample_idx in sample_indices:
            if sample_idx < 0:
                continue
            if sample_idx >= len(self._X_test):
                continue

            sample = shap_values[sample_idx]
            if len(sample.values.shape) > 1:
                sample_values = sample.values[:, 1]
            else:
                sample_values = sample.values
            shap_contribs = safe_array(sample_values).astype(float).tolist()

            if isinstance(self._X_test, pd.DataFrame):
                raw_feature_values = (self._X_test.iloc[sample_idx].values)
            else:
                raw_feature_values = self._X_test[sample_idx]
            raw_feature_values = safe_array(raw_feature_values).tolist()
            
            feature_values = list()
            for v in raw_feature_values:
                if isinstance(v, (np.integer, int)):
                    feature_values.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    feature_values.append(float(v))
                else:
                    feature_values.append(str(v))

            if np.isscalar(sample.base_values):
                base_value = float(safe_array(sample.base_values))
            else:
                base_value = float(safe_array(sample.base_values[1]))
            prediction = float(safe_array(base_value + np.sum(shap_contribs)))

            rows = list()
            for fname, fvalue, sval in zip(
                Columns,
                feature_values,
                shap_contribs):
                rows.append({
                    "feature"       : str(fname),
                    "feature_value" : fvalue,
                    "shap_value"    : round(float(sval), 8),
                    "abs_shap"      : round(float(abs(sval)), 8)
                    })
            rows        = sorted(rows, key = lambda x: x["abs_shap"], reverse = True)
            rows        = rows[:max_display]
            CustID      = self._get_SubjectID(sample_idx)
            CustData    = self._get_CustData(CustID)
            chart_info  = {
                "title" : f"SHAP Waterfall (Row {sample_idx})",
                "type"  : "shap_waterfall_js",
                "data"  : {"sample_idx"  : int(sample_idx),
                           'SubName'     : str(self._SubjectID),
                           'SubValue'    : CustID,
                           'SubData'     : CustData.to_dict(orient = 'records'),
                           "base_value"  : round(base_value, 8),
                           "prediction"  : round(prediction, 8),
                           "max_display" : int(max_display),
                           "features"    : rows,
                          },
                "image" : None,
                "full"  : False}
            self._chartsdata.append(chart_info)


    def __call__(self) -> None:
        logger.info("Generating all visualisations.")
        self.plot_feature_importance()
        self.plot_prediction_distribution()
        self.plot_learning_curves()
        self.plot_metrics_summary()
        self.plot_feature_correlation()
        logger.info("All visualisations complete.")


    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    def genreport(
            self,
            preddata     : Optional[str | Path] = None,
            tuner_summary: Optional[Dict[str, Any]] = None,
            model_metric : Dict = dict(),
        ) -> None:
        """
        Render a self-contained HTML monitoring report.
        The report bundles every chart as a base64 data-URI so it can
        be shared as a single file with no external dependencies.
        ________________________________
        Parameters
        preddata     : Result of Prediction dataframe's locatated
        tuner_summary: Optional dict returned by :meth:`BayesianTuner.summary`
                       (adds a tuning section to the report).
        ________________________________
        Behavior
        - If charts already exist in self._b64_images, use them.
        - If some / all charts are missing, fallback gracefully with filtered defaults.
        - Never fail report generation only because charts are absent.
        """
        logger.info("Starting genreport().")

        # =====================================================
        # Prepare output path
        # =====================================================
        try:
            output_path = Path(self._config.path.html_report_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Output path prepared: %s",output_path)
        except Exception as arc:
            logger.exception("Failed preparing output path.")
            raise RuntimeError("Cannot prepare HTML output path.") from arc

        # =====================================================
        # Chart Defaults
        # =====================================================
        chart_specs = [
            ("feature_importance", "Feature Importance", False),
            ("learning_curves", "Learning Curves", False),
            ("prediction_distribution",
             "Prediction Score Distribution", False),
            ("metrics_summary", "Metrics Summary", False)]

        # =====================================================
        # Open the prediction data
        # =====================================================
        pred = Path(preddata)
        parq = _cfg.getboolean('INFERENCE', 'parquet')
        ext  = '.csv' if not parq else '.parquet'
        msg  = 'Prediction data is not floud in {str(pred)}' \
               'We can not continue the progress to write HTML file.'
        assert pred.exists(), msg
        if 'csv' in ext:
            predictdata = pd.read_csv(str(pred))
        else:
            predictdata = pd.read_parquet(str(pred), engine = 'pyarrow')
        self._predictdata = deepcopy(predictdata)
        self.shap_waterfall_chart()

        # =====================================================
        # Build charts safely
        # =====================================================
        # try:
        #     logger.debug("Attempting to load charts from self._b64_images.")
        #     images = getattr(self, "_b64_images", None)
        #     if not isinstance(images, dict):
        #         raise TypeError("_b64_images missing or not dict.")
        #     for key, title, full in chart_specs:
        #         try:
        #             img = images[key]
        #             if img:
        #                 self._chartsdata.append(
        #                     {"title": title,
        #                      'type' : image_chart,
        #                      "image": img,
        #                      "full" : full})
        #                 logger.debug("Chart loaded: %s",key)
        #             else:
        #                 logger.debug("Chart empty, skipped: %s",key)
        #         except KeyError:
        #             logger.error("Chart key missing, skipped: %s",key)
        # except Exception as exc:
        #     logger.error("Chart loading fallback activated: %s",str(exc))
        #     for key, title, full in chart_specs:
        #         self._chartsdata.append({"title": title,
        #                                  "image": None,
        #                                  "full" : full})
        # logger.debug("Final chart count prepared: %d",len(self._chartsdata))

        # =====================================================
        # Build context Recsys
        # =====================================================
        try:
            contextRecsys = {
            "page_title"        : "LightGBM LTR Monitoring Report",
            "title"             : "LightGBM LTR Monitoring Dashboard",
            "subtitle"          : "Production Model Evaluation & Diagnostics",
            "experiment_name"   : getattr(self._config.model,
                                          "experiment_name",
                                          "unknown_experiment"),
            "model_path"        : getattr(self._config.model,
                                          "model_path",
                                          "unknown_model"),
            "best_iteration"    : getattr(self._model,
                                          "best_iteration",
                                          None),
            "generated_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics"           : dict(sorted(getattr(self,"_metrics",{}).items())),
            "training_params"   : {**getattr(self._config.training,
                                        "params",{}),
                                    "num_boost_round": getattr(
                                        self._config.training,
                                        "num_boost_round",
                                        None),
                                    "early_stopping_rounds": getattr(
                                        self._config.training,
                                        "early_stopping_rounds",
                                        None)},
            "tuner_summary"     : tuner_summary,
            "charts"            : self._chartsdata,
            "modelmetric"       : model_metric,
            "username"          : _cfg.get('DEFAULT', 'Username'),
            "predictiondata"    : predictdata.to_dict(orient = 'records'),
            }
            logger.debug("contextRecsys built for template "
            "report`s requirement successfully.")
        except Exception as arc:
            logger.exception("Failed building report contextRecsys.")
            raise RuntimeError("Unable to build report contextRecsys.") from arc


        # =====================================================
        # Render HTML
        # =====================================================
        try:
            logger.debug("Rendering report HTML.")
            Dict2Json(contextRecsys, str(output_path.parent / 'contextRecsys.json'))
            htmlpath = repot()
            if not htmlpath.exists():
                raise ValueError("Rendered HTML is empty.")
            logger.debug("HTML rendered successfully ""(%s).",str(htmlpath))
        except Exception as arc:
            logger.error("Failed rendering HTML.")
            raise RuntimeError("Report rendering failed.") from arc


if __name__ == '__main__':
    pass