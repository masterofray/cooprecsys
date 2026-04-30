#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-04-30"


"""
visualization.py
================
Publication-quality visualisations and a self-contained HTML monitoring
report for the LTR pipeline.

All plot artifacts (PNG) are saved under ``config.path.output_dir``.
The HTML report bundles every chart as a base64 data-URI so it requires
no external assets and can be shared as a single file.

Classes
-------
Visualizer — generates and saves all visual artefacts.

Design notes
------------
* All state stored on ``self``; no public ``return`` from plot methods.
* Figures are closed after saving to prevent memory leaks in long runs.
* HTML report is rendered via an f-string template (zero Jinja2 dependency).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend — safe for servers/workers

from ltr_framework.config import LTRConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seaborn global theme
# ---------------------------------------------------------------------------

sns.set_theme(
    style   = "whitegrid",
    palette = "muted",
    rc      = {"figure.dpi": 120, "axes.titlesize": 14, "axes.labelsize": 12},
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode a Matplotlib figure to a base64 PNG string (data-URI ready)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _save_fig(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* and close it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.debug("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Visualizer:
    """Generate all visual artefacts for the LTR pipeline.

    Parameters
    ----------
    config:
        :class:`~ltr_framework.config.LTRConfig` master config.
    model:
        Trained :class:`lgb.Booster`.
    evals_result:
        Training metric history dict (from :class:`LTRTrainer`).
    X_test:
        Test feature matrix (for prediction distribution plots).
    metrics:
        Flat summary metrics dict (from :class:`LTRTrainer`).

    Attributes (populated after each plot method)
    --------------------------------------------
    _b64_images : Dict[str, str]
        Map from chart name → base64 PNG (used by :meth:`generate_html_report`).
    """

    def __init__(
        self,
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

        self._b64_images:  Dict[str, str]  = {}
        self._saved_paths: Dict[str, str]  = {}

        logger.debug("Visualizer initialised.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dir(self) -> str:
        return self._config.path.output_dir

    @property
    def feature_names(self) -> List[str]:
        return self._config.feature.features

    # ------------------------------------------------------------------
    # Internal save helper
    # ------------------------------------------------------------------

    def _record(self, name: str, fig: plt.Figure) -> None:
        """Save figure to PNG, encode to base64, store in state dicts."""
        path = os.path.join(self.output_dir, f"{name}.png")
        _save_fig(fig, path)
        self._saved_paths[name] = path

        # Re-read PNG → base64 for the HTML report
        with open(path, "rb") as f:
            self._b64_images[name] = base64.b64encode(f.read()).decode("utf-8")

        logger.info("Chart saved: %s", path)

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 30,
    ) -> None:
        """Bar chart of LightGBM feature importances.

        Parameters
        ----------
        importance_type:
            ``"gain"`` (default) or ``"split"``.
        top_n:
            Maximum number of features to display.
        """
        logger.info(
            "Plotting feature importance (type=%s, top_n=%d).",
            importance_type, top_n,
        )

        imp     = self._model.feature_importance(importance_type=importance_type)
        names   = self._model.feature_name()

        df = (
            pd.DataFrame({"Feature": names, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.35)))
        sns.barplot(
            data    = df,
            x       = "Importance",
            y       = "Feature",
            palette = "Blues_r",
            ax      = ax,
        )
        ax.set_title(
            f"Feature Importance — {importance_type.capitalize()} (Top {top_n})"
        )
        ax.set_xlabel(f"Importance ({importance_type})")
        ax.set_ylabel("")

        self._record("feature_importance", fig)

    # ------------------------------------------------------------------

    def plot_prediction_distribution(self) -> None:
        """Histogram + KDE of test-set relevance scores."""
        logger.info("Plotting prediction distribution.")

        preds = self._model.predict(
            self._X_test,
            num_iteration = self._model.best_iteration or 0,
        )

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
            f"max={np.max(preds):.4f}"
        )
        ax.text(
            0.97, 0.97, stats_text,
            transform   = ax.transAxes,
            va          = "top",
            ha          = "right",
            fontsize    = 10,
            family      = "monospace",
            bbox        = dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
        )

        self._record("prediction_distribution", fig)

    # ------------------------------------------------------------------

    def plot_learning_curves(self) -> None:
        """Line chart of train/test NDCG over boosting rounds."""
        logger.info("Plotting learning curves.")

        fig, axes = plt.subplots(1, 1, figsize=(10, 5))

        for split_name, metric_dict in self._evals_result.items():
            for metric_name, values in metric_dict.items():
                label = f"{split_name} — {metric_name}"
                linestyle = "-" if split_name == "train" else "--"
                axes.plot(
                    values,
                    label     = label,
                    linestyle = linestyle,
                    linewidth = 1.8,
                )

        best_iter = self._model.best_iteration
        if best_iter:
            axes.axvline(
                x         = best_iter,
                color     = "red",
                linestyle = ":",
                linewidth = 1.5,
                label     = f"Best iteration ({best_iter})",
            )

        axes.set_title("Learning Curves (NDCG per Round)")
        axes.set_xlabel("Boosting Round")
        axes.set_ylabel("NDCG")
        axes.legend(loc="lower right", fontsize=9)

        self._record("learning_curves", fig)

    # ------------------------------------------------------------------

    def plot_metrics_summary(self) -> None:
        """Horizontal bar chart summarising key evaluation metrics."""
        logger.info("Plotting metrics summary.")

        # Filter to NDCG and prediction stats only
        filtered = {
            k: v for k, v in self._metrics.items()
            if "ndcg" in k.lower() or "pred" in k.lower()
        }
        if not filtered:
            logger.warning("No metrics found for summary chart.")
            return

        df = pd.DataFrame(
            list(filtered.items()),
            columns=["Metric", "Value"],
        ).sort_values("Value", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.55)))
        bars = ax.barh(
            df["Metric"], df["Value"],
            color   = "#55A868",
            edgecolor = "white",
            height  = 0.6,
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_title("Evaluation Metrics Summary")
        ax.set_xlabel("Value")
        ax.set_xlim(0, max(df["Value"]) * 1.18)

        self._record("metrics_summary", fig)

    # ------------------------------------------------------------------

    def plot_feature_correlation(self, sample_n: int = 5_000) -> None:
        """Heatmap of pairwise feature correlations on a random sample.

        Parameters
        ----------
        sample_n:
            Maximum rows to sample before computing the correlation matrix
            (guards against OOM on huge test sets).
        """
        logger.info("Plotting feature correlation heatmap.")

        n = min(sample_n, self._X_test.shape[0])
        idx = np.random.choice(self._X_test.shape[0], size=n, replace=False)
        sample = pd.DataFrame(self._X_test[idx], columns=self.feature_names)

        corr = sample.corr()

        size = max(8, len(self.feature_names) * 0.55)
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

    # ------------------------------------------------------------------

    def generate_all(self) -> None:
        """Generate every standard chart in sequence."""
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

    def generate_html_report(
        self,
        tuner_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Render a self-contained HTML monitoring report.

        The report bundles every chart as a base64 data-URI so it can
        be shared as a single file with no external dependencies.

        Parameters
        ----------
        tuner_summary:
            Optional dict returned by :meth:`BayesianTuner.summary`
            (adds a tuning section to the report).
        """
        logger.info("Generating HTML report.")

        path = self._config.path.html_report_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # ------------------------------------------------------------------
        # Helper sub-renderers
        # ------------------------------------------------------------------

        def _img_tag(name: str) -> str:
            b64 = self._b64_images.get(name, "")
            if not b64:
                return f"<p><em>Chart '{name}' not available.</em></p>"
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{name}" style="max-width:100%;border-radius:6px;'
                f'box-shadow:0 2px 8px rgba(0,0,0,.15);">'
            )

        def _metric_cards(metrics: Dict[str, float]) -> str:
            cards = ""
            for k, v in sorted(metrics.items()):
                cards += (
                    f'<div class="card">'
                    f'  <div class="card-label">{k}</div>'
                    f'  <div class="card-value">{v:.6f}</div>'
                    f'</div>'
                )
            return cards

        def _tuning_section(ts: Optional[Dict[str, Any]]) -> str:
            if not ts:
                return "<p><em>Bayesian tuning was not run in this session.</em></p>"
            rows = "".join(
                f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
                for k, v in ts.get("best_params", {}).items()
            )
            return f"""
            <table>
              <tr><th>Metric</th><th>Value</th></tr>
              <tr><td>Trials completed</td><td>{ts.get('n_trials', '—')}</td></tr>
              <tr><td>Trials pruned</td><td>{ts.get('n_pruned', '—')}</td></tr>
              <tr>
                <td>Best NDCG</td>
                <td><strong>{ts.get('best_value', 0.0):.6f}</strong></td>
              </tr>
            </table>
            <h3>Best Hyper-parameters</h3>
            <table><tr><th>Parameter</th><th>Value</th></tr>{rows}</table>
            """

        # ------------------------------------------------------------------
        # Full HTML
        # ------------------------------------------------------------------

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LightGBM LTR — Monitoring Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #f4f6fb;
      color: #2d3748;
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(135deg, #1a202c 0%, #2d3a5e 100%);
      color: #fff;
      padding: 2.4rem 3rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }}
    header .logo {{
      font-size: 2.6rem;
    }}
    header h1 {{ font-size: 1.8rem; font-weight: 700; }}
    header p  {{ opacity: .75; font-size: .95rem; margin-top: .25rem; }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 2rem 2.5rem;
    }}
    section {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      padding: 2rem;
      margin-bottom: 2rem;
    }}
    section h2 {{
      font-size: 1.25rem;
      font-weight: 700;
      border-bottom: 3px solid #4C72B0;
      padding-bottom: .5rem;
      margin-bottom: 1.5rem;
      color: #1a202c;
    }}
    section h3 {{
      font-size: 1rem;
      font-weight: 600;
      margin: 1.25rem 0 .75rem;
      color: #2d3a5e;
    }}
    .cards {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    .card {{
      background: #f7f9fc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 1rem 1.4rem;
      min-width: 180px;
      flex: 1;
    }}
    .card-label {{
      font-size: .78rem;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: .35rem;
    }}
    .card-value {{
      font-size: 1.45rem;
      font-weight: 700;
      color: #4C72B0;
      font-variant-numeric: tabular-nums;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 1.5rem;
    }}
    .chart-wrap {{
      background: #f9fafb;
      border: 1px solid #e8ecf2;
      border-radius: 8px;
      padding: 1rem;
      overflow: hidden;
    }}
    .chart-wrap h3 {{
      margin-top: 0;
      margin-bottom: .6rem;
      font-size: .9rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: .88rem;
      margin-top: .5rem;
    }}
    th, td {{
      text-align: left;
      padding: .55rem .8rem;
      border-bottom: 1px solid #e2e8f0;
    }}
    th {{
      background: #edf2f7;
      font-weight: 600;
      font-size: .8rem;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    code {{
      background: #edf2f7;
      padding: .15rem .4rem;
      border-radius: 4px;
      font-size: .85em;
    }}
    footer {{
      text-align: center;
      padding: 1.5rem;
      color: #a0aec0;
      font-size: .82rem;
    }}
    @media (max-width: 680px) {{
      header {{ flex-direction: column; align-items: flex-start; }}
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="logo">📊</div>
    <div>
      <h1>LightGBM LTR &mdash; Monitoring Report</h1>
      <p>Experiment: <strong>{self._config.model.experiment_name}</strong>
         &nbsp;|&nbsp; Model: {self._config.model.model_path}</p>
    </div>
  </header>

  <div class="container">

    <!-- ── Evaluation Metrics ── -->
    <section>
      <h2>📈 Evaluation Metrics</h2>
      <div class="cards">
        {_metric_cards(self._metrics)}
      </div>
    </section>

    <!-- ── Bayesian Tuning ── -->
    <section>
      <h2>🔬 Bayesian Hyper-parameter Tuning</h2>
      {_tuning_section(tuner_summary)}
    </section>

    <!-- ── Training Config ── -->
    <section>
      <h2>⚙️ Training Configuration</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        {''.join(
            f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
            for k, v in sorted(self._config.training.params.items())
        )}
        <tr>
          <td>num_boost_round</td>
          <td><code>{self._config.training.num_boost_round}</code></td>
        </tr>
        <tr>
          <td>early_stopping_rounds</td>
          <td><code>{self._config.training.early_stopping_rounds}</code></td>
        </tr>
        <tr>
          <td>best_iteration</td>
          <td><code>{self._model.best_iteration}</code></td>
        </tr>
      </table>
    </section>

    <!-- ── Charts ── -->
    <section>
      <h2>🎨 Visualisations</h2>
      <div class="charts-grid">

        <div class="chart-wrap">
          <h3>Feature Importance (Gain)</h3>
          {_img_tag("feature_importance")}
        </div>

        <div class="chart-wrap">
          <h3>Learning Curves</h3>
          {_img_tag("learning_curves")}
        </div>

        <div class="chart-wrap">
          <h3>Prediction Score Distribution</h3>
          {_img_tag("prediction_distribution")}
        </div>

        <div class="chart-wrap">
          <h3>Metrics Summary</h3>
          {_img_tag("metrics_summary")}
        </div>

        <div class="chart-wrap" style="grid-column: 1 / -1;">
          <h3>Feature Correlation Heatmap</h3>
          {_img_tag("feature_correlation")}
        </div>

      </div>
    </section>

  </div>

  <footer>
    Generated by <strong>ltr_framework</strong> v1.0.0 &mdash;
    Contact: aryanto.dandan@gmail.com
  </footer>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report written to: %s", path)
