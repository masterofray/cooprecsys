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
import os
import sys
import base64
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from ipdb import set_trace


matplotlib.use("Agg")
LocDir = Path(__file__).resolve().parents[3]
sys.path.append(str(LocDir))

from prepare import Dict2Json
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
        with open(path, "rb") as f:
            self._b64_images[name] = base64.b64encode(f.read()).decode("utf-8")
        logger.info("Chart saved: %s", path)


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
        names    = self.feature_names #self._model.feature_name()
        dataplot = pd.DataFrame({"Feature": names, "Importance": imp})\
                   .sort_values("Importance", ascending = False)\
                   .head(top_n)
        #set_trace()
        fig, ax  = plt.subplots(figsize=(11, max(5, top_n * 0.35)))
        sns.barplot(
            data    = dataplot,
            x       = "Importance",
            y       = "Feature",
            hue     = "Feature",
            palette = "Blues_r",
            legend  = False,
            ax      = ax)
        ax.set_title(f"Feature Importance — {importance_type.capitalize()} (Top {top_n})")
        ax.set_xlabel(f"Importance ({importance_type})")
        ax.set_ylabel("")
        
        datapltlist = np.nan_to_num(dataplot["Importance"], nan = 0.0,
                                    posinf = 0.0, neginf = 0.0).tolist()
        chart_info  = {'title'   : f'Feature Importance ({importance_type})',
                       'type'    : 'bar_chart_js',
                       'data'    : {'labels': dataplot["Feature"].tolist(),
                                    'values': datapltlist},
                       'image'   : None,
                       'full'    : False}
        self._chartsdata.append(chart_info)
        #self._record("feature_importance", fig)


    def plot_prediction_distribution(self) -> None:
        """Histogram + KDE of test-set relevance scores."""
        logger.debug("Plotting prediction distribution.")
        preds = self._model.predict(self._X_test,
                num_iteration = self._model.best_iteration or 0,)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(preds, bins=60, kde=True, color="#4C72B0", ax=ax)
        ax.set_title("Relevance Score Distribution (Test Set)")
        ax.set_xlabel("Predicted Relevance Score")
        ax.set_ylabel("Count")

        # Annotate summary stats
        stats_text = (
            f"mean={np.mean(preds):.4f}\n"
            f"std={np.std(preds):.4f}\n"
            f"min={np.min(preds):.4f}\n"
            f"max={np.max(preds):.4f}")
        ax.text(
            0.97, 0.97, stats_text,
            transform   = ax.transAxes,
            va          = "top",
            ha          = "right",
            fontsize    = 10,
            family      = "monospace",
            bbox        = dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        self._record("prediction_distribution", fig)

    def plot_learning_curves(self) -> None:
        """Line chart of train/test NDCG over boosting rounds."""
        logger.debug("Plotting learning curves.")
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        for split_name, metric_dict in self._evals_result.items():
            for metric_name, values in metric_dict.items():
                label = f"{split_name} — {metric_name}"
                linestyle = "-" if split_name == "train" else "--"
                axes.plot(
                    values,
                    label     = label,
                    linestyle = linestyle,
                    linewidth = 1.8)
        best_iter = self._model.best_iteration
        if best_iter:
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
                    columns=["Metric", "Value"],
                    ).sort_values("Value", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(3, len(tempdata) * 0.55)))
        bars = ax.barh(
            tempdata["Metric"], tempdata["Value"],
            color   = "#55A868",
            edgecolor = "white",
            height  = 0.6)
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
        if pred.exists():
            if 'csv' in ext:
                predictdata = pd.read_csv(str(pred))
            else:
                predictdata = pd.read_parquet(str(pred), engine = 'pyarrow')
        else:
            logger.error('Prediction data is not floud in {str(pred)}'
                         'We can not continue the progress to write HTML file.')
            predictdata = pd.DataFrame(list())

        # =====================================================
        # Build charts safely
        # =====================================================
        try:
            logger.debug("Attempting to load charts from self._b64_images.")
            images = getattr(self, "_b64_images", None)
            if not isinstance(images, dict):
                raise TypeError("_b64_images missing or not dict.")
            for key, title, full in chart_specs:
                try:
                    img = images[key]
                    if img:
                        self._chartsdata.append(
                            {"title": title,
                             'type' : image_chart,
                             "image": img,
                             "full" : full})
                        logger.debug("Chart loaded: %s",key)
                    else:
                        logger.debug("Chart empty, skipped: %s",key)
                except KeyError:
                    logger.error("Chart key missing, skipped: %s",key)
        except Exception as exc:
            logger.error("Chart loading fallback activated: %s",str(exc))
            for key, title, full in chart_specs:
                self._chartsdata.append({"title": title,
                                         "image": None,
                                         "full" : full})
        logger.debug("Final chart count prepared: %d",len(self._chartsdata))

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