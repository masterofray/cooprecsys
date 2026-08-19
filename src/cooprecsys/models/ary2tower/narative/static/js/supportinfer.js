/*
* author  = "Aryanto"
* about   = supportinfer.js -- renders the ary2tower inference dashboard's
*           score histogram, embedding scatter, and similarity heatmap.
*           Same data-attribute convention as arycolbring's insights.js
*           (data-z/data-x/data-y etc. as JSON strings).
*/

(function () {
    var ACCENT = '#FF6B35';
    var ACCENT_TEAL = '#4ECDC4';

    function plotlyReady() {
        return typeof Plotly !== 'undefined';
    }

    function showUnavailable(el, label) {
        el.innerHTML = '<p style="color:#868e96;text-align:center;padding:24px 0;">'
            + label + ' unavailable: Plotly failed to load.</p>';
    }

    function renderHistograms() {
        document.querySelectorAll('.js-histogram-render').forEach(function (el) {
            if (!plotlyReady()) { showUnavailable(el, 'Score distribution'); return; }
            try {
                var counts = JSON.parse(el.getAttribute('data-counts') || '[]');
                var binEdges = JSON.parse(el.getAttribute('data-bin-edges') || '[]');
                var mean = parseFloat(el.getAttribute('data-mean'));
                if (!counts.length || !binEdges.length) return;

                var centers = [];
                for (var i = 0; i < counts.length; i++) {
                    centers.push((binEdges[i] + binEdges[i + 1]) / 2);
                }

                var trace = { x: centers, y: counts, type: 'bar', marker: { color: ACCENT } };
                var shapes = [];
                if (!isNaN(mean)) {
                    shapes.push({ type: 'line', x0: mean, x1: mean, y0: 0, y1: 1, yref: 'paper',
                                 line: { color: ACCENT_TEAL, width: 2, dash: 'dash' } });
                }
                Plotly.newPlot(el, [trace], {
                    margin: { t: 10, r: 10, b: 40, l: 40 },
                    xaxis: { title: 'Prediction score' }, yaxis: { title: 'Count' },
                    shapes: shapes, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                }, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[ary2tower] Histogram render failed:', err);
            }
        });
    }

    function renderScatters() {
        document.querySelectorAll('.js-scatter-render').forEach(function (el) {
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

                var trace = { x: x, y: y, text: ids, mode: 'markers', type: 'scattergl',
                             marker: { color: colors, size: sizes, opacity: 0.85 },
                             hovertemplate: 'id=%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>' };
                Plotly.newPlot(el, [trace], {
                    margin: { t: 10, r: 10, b: 40, l: 40 },
                    xaxis: { title: 'PC1' }, yaxis: { title: 'PC2' },
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                }, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[ary2tower] Scatter render failed:', err);
            }
        });
    }

    function renderHeatmaps() {
        document.querySelectorAll('.js-heatmap-render').forEach(function (el) {
            if (!plotlyReady()) { showUnavailable(el, 'Similarity heatmap'); return; }
            try {
                var z = JSON.parse(el.getAttribute('data-z') || '[]');
                var x = JSON.parse(el.getAttribute('data-x') || '[]');
                var y = JSON.parse(el.getAttribute('data-y') || '[]');
                if (!z.length) return;

                var trace = { z: z, x: x, y: y, type: 'heatmap',
                             colorscale: [[0, '#F8F9FA'], [0.5, '#FFC299'], [1, ACCENT]] };
                Plotly.newPlot(el, [trace], {
                    margin: { t: 10, r: 10, b: 60, l: 60 },
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                }, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[ary2tower] Heatmap render failed:', err);
            }
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        renderHistograms();
        renderScatters();
        renderHeatmaps();
        console.log('[ary2tower] Inference dashboard initialized.');
    });
})();
