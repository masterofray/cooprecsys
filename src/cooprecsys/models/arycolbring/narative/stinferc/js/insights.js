/*
* author     = "Aryanto"
* copyright  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
* license    = "GNUPublic"
* version    = "0.1.0"
* email      = "aryanto.dandan@gmail.com"
* status     = "Development"
* created    = "2026-07-29"
* About
  - insights.js
  - Renders the Insights tab: prediction-score histogram, item embedding
    scatter (2D PCA), and item-item similarity heatmap. Reuses the
    `.js-heatmap-render` data-attribute convention already established in
    ltr_lgbm/report/static/js/heatmaps.js (data-z/data-x/data-y as JSON
    strings), plus two sibling conventions for the scatter and histogram.
 */

(function () {
    var ACCENT = '#FF6B35';
    var ACCENT_TEAL = '#4ECDC4';

    function plotlyReady() {
        return typeof Plotly !== 'undefined';
    }

    function showUnavailable(el, label) {
        el.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:24px 0;">'
            + label + ' unavailable: Plotly failed to load.</p>';
    }

    function renderHistograms(container) {
        var elements = container.querySelectorAll('.js-histogram-render');
        elements.forEach(function (el) {
            if (!plotlyReady()) { showUnavailable(el, 'Score distribution'); return; }
            try {
                var counts = JSON.parse(el.getAttribute('data-counts') || '[]');
                var binEdges = JSON.parse(el.getAttribute('data-bin-edges') || '[]');
                var mean = parseFloat(el.getAttribute('data-mean'));
                var median = parseFloat(el.getAttribute('data-median'));
                if (!counts.length || !binEdges.length) return;

                var centers = [];
                for (var i = 0; i < counts.length; i++) {
                    centers.push((binEdges[i] + binEdges[i + 1]) / 2);
                }

                var trace = {
                    x: centers, y: counts, type: 'bar',
                    marker: { color: ACCENT, line: { color: '#ffffff', width: 1 } },
                    name: 'Predictions',
                };
                var shapes = [];
                if (!isNaN(mean)) {
                    shapes.push({ type: 'line', x0: mean, x1: mean, y0: 0, y1: 1,
                                 yref: 'paper', line: { color: ACCENT_TEAL, width: 2, dash: 'dash' } });
                }
                var layout = {
                    margin: { t: 10, r: 10, b: 40, l: 40 },
                    xaxis: { title: 'Prediction score' },
                    yaxis: { title: 'Count' },
                    shapes: shapes,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: 'var(--text)' },
                };
                Plotly.newPlot(el, [trace], layout, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[Insights] Histogram render failed:', err);
            }
        });
    }

    function renderScatters(container) {
        var elements = container.querySelectorAll('.js-scatter-render');
        elements.forEach(function (el) {
            if (!plotlyReady()) { showUnavailable(el, 'Embedding projection'); return; }
            try {
                var x = JSON.parse(el.getAttribute('data-x') || '[]');
                var y = JSON.parse(el.getAttribute('data-y') || '[]');
                var ids = JSON.parse(el.getAttribute('data-ids') || '[]');
                var highlightRaw = el.getAttribute('data-highlight-index');
                var highlightIndex = highlightRaw !== '' ? parseInt(highlightRaw, 10) : null;
                if (!x.length) return;

                var colors = x.map(function (_, i) {
                    return (highlightIndex !== null && i === highlightIndex) ? ACCENT : ACCENT_TEAL;
                });
                var sizes = x.map(function (_, i) {
                    return (highlightIndex !== null && i === highlightIndex) ? 14 : 7;
                });

                var trace = {
                    x: x, y: y, text: ids, mode: 'markers', type: 'scattergl',
                    marker: { color: colors, size: sizes, opacity: 0.85,
                             line: { color: '#ffffff', width: 1 } },
                    hovertemplate: 'id=%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>',
                };
                var layout = {
                    margin: { t: 10, r: 10, b: 40, l: 40 },
                    xaxis: { title: 'PC1', zeroline: false },
                    yaxis: { title: 'PC2', zeroline: false },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: 'var(--text)' },
                };
                Plotly.newPlot(el, [trace], layout, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[Insights] Scatter render failed:', err);
            }
        });
    }

    function renderHeatmaps(container) {
        var elements = container.querySelectorAll('.js-heatmap-render');
        elements.forEach(function (el) {
            if (!plotlyReady()) { showUnavailable(el, 'Similarity heatmap'); return; }
            try {
                var z = JSON.parse(el.getAttribute('data-z') || '[]');
                var x = JSON.parse(el.getAttribute('data-x') || '[]');
                var y = JSON.parse(el.getAttribute('data-y') || '[]');
                if (!z.length) return;

                var trace = {
                    z: z, x: x, y: y, type: 'heatmap',
                    colorscale: [[0, '#F8F9FA'], [0.5, '#FFC299'], [1, ACCENT]],
                    hovertemplate: '%{x} vs %{y}: %{z:.3f}<extra></extra>',
                };
                var layout = {
                    margin: { t: 10, r: 10, b: 60, l: 60 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: 'var(--text)' },
                };
                Plotly.newPlot(el, [trace], layout, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[Insights] Heatmap render failed:', err);
            }
        });
    }

    function initInsights(container) {
        if (!container) return;
        renderHistograms(container);
        renderScatters(container);
        renderHeatmaps(container);
        console.log('[Insights] Initialized.');
    }

    window.initInsights = initInsights;
})();
