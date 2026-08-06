/*
* author  = "Aryanto"
* about   = supporttrain.js -- renders the ary2tower training dashboard's
*           loss curve.
*/

(function () {
    var ACCENT = '#FF6B35';

    function plotlyReady() {
        return typeof Plotly !== 'undefined';
    }

    function renderLossCurve() {
        document.querySelectorAll('.js-loss-render').forEach(function (el) {
            if (!plotlyReady()) {
                el.innerHTML = '<p style="color:#868e96;text-align:center;padding:24px 0;">'
                    + 'Loss curve unavailable: Plotly failed to load.</p>';
                return;
            }
            try {
                var loss = JSON.parse(el.getAttribute('data-loss') || '[]');
                if (!loss.length) return;

                var epochs = [];
                var values = [];
                for (var i = 0; i < loss.length; i++) {
                    if (loss[i] === null || isNaN(loss[i])) continue;
                    epochs.push(i + 1);
                    values.push(loss[i]);
                }
                if (!values.length) {
                    el.innerHTML = '<p style="color:#868e96;text-align:center;padding:24px 0;">'
                        + 'No per-epoch loss available (Cython backend doesn\'t return one).</p>';
                    return;
                }

                var trace = { x: epochs, y: values, type: 'scatter', mode: 'lines+markers',
                             line: { color: ACCENT, width: 2 }, marker: { size: 5 } };
                Plotly.newPlot(el, [trace], {
                    margin: { t: 10, r: 10, b: 40, l: 50 },
                    xaxis: { title: 'Epoch' }, yaxis: { title: 'Loss' },
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                }, { responsive: true, displaylogo: false });
            } catch (err) {
                console.error('[ary2tower] Loss curve render failed:', err);
            }
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        renderLossCurve();
        console.log('[ary2tower] Training dashboard initialized.');
    });
})();
