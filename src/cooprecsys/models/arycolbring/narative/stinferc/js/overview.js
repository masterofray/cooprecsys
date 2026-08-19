/**
 * overview.js — Inference overview charts (line, donut, gauge)
 */

(function () {
    'use strict';

    function safeParse(v, f) { if (f === undefined) f = []; try { return JSON.parse(v); } catch(e) { return f; } }
    function calibrateCanvas(c) { var p = c.parentElement; if (!p) return; var w = p.clientWidth, h = p.clientHeight; if (w > 0) c.setAttribute('width', w); if (h > 0) c.setAttribute('height', h); }
    function alpha(hex, a) { var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16); return 'rgba('+r+','+g+','+b+','+a+')'; }

    function initLineCharts(container) {
        container.querySelectorAll('.js-line-chart').forEach(function (canvas) {
            var labels = safeParse(canvas.dataset.labels), values = safeParse(canvas.dataset.values);
            if (!labels.length || !values.length) return;
            calibrateCanvas(canvas);
            var ctx = canvas.getContext('2d');
            var h = canvas.parentElement.clientHeight || 280;
            var grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, alpha('#FF6B35', 0.18));
            grad.addColorStop(1, alpha('#FF6B35', 0.0));

            new Chart(ctx, {
                type: 'line',
                data: { labels: labels, datasets: [{
                    label: 'Value', data: values,
                    borderColor: '#38bdf8', backgroundColor: grad,
                    borderWidth: 2.5, tension: 0.4, fill: true,
                    pointBackgroundColor: '#38bdf8', pointBorderColor: '#fff',
                    pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7
                }]},
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false },
                        tooltip: { backgroundColor: '#fff', titleColor: '#212529', bodyColor: '#495057', borderColor: 'rgba(33,37,41,0.15)', borderWidth: 1, padding: 12, cornerRadius: 8,
                            callbacks: { label: function(c) { return c.parsed.y.toFixed(4); } }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(33,37,41,0.08)' }, ticks: { color: '#868e96', font: { size: 10 } } },
                        y: { grid: { color: 'rgba(33,37,41,0.08)' }, ticks: { color: '#868e96', font: { size: 10 } } }
                    }
                }
            });
        });
    }

    function initDonutCharts(container) {
        var colors = ['#FF6B35', '#4ECDC4', '#f9a825', '#6a1b9a'];
        container.querySelectorAll('.js-donut-chart').forEach(function (canvas) {
            var pct = Number(canvas.dataset.percent || 0), ci = Number(canvas.dataset.colorIndex || 0);
            calibrateCanvas(canvas);
            new Chart(canvas, {
                type: 'doughnut',
                data: { datasets: [{ data: [pct, 100 - pct], backgroundColor: [colors[ci % colors.length], 'rgba(33,37,41,0.08)'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { display: false } } }
            });
        });
    }

    function initGaugeCharts(container) {
        var gc = ['#FF6B35', '#4ECDC4', '#f9a825', '#c62828', '#6a1b9a', '#00838f', '#e65100', '#00695c', '#ad1457', '#4ECDC4', '#FF6B35'];
        container.querySelectorAll('.js-gauge-chart').forEach(function (canvas) {
            var pct = Number(canvas.dataset.percent || 0), ci = Number(canvas.dataset.colorIndex || 0);
            calibrateCanvas(canvas);
            new Chart(canvas, {
                type: 'doughnut',
                data: { datasets: [{ data: [pct, 100 - pct], backgroundColor: [gc[ci % gc.length], 'rgba(33,37,41,0.06)'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', rotation: -90, circumference: 180, plugins: { legend: { display: false } } }
            });
        });
    }

    function initScoreCards(container) {
        container.querySelectorAll('.scorecard').forEach(function (card) {
            card.addEventListener('click', function () {
                var val = card.querySelector('.scorecard-value');
                if (!val) return;
                navigator.clipboard.writeText(val.textContent.trim()).then(function () {
                    card.classList.add('copied');
                    setTimeout(function () { card.classList.remove('copied'); }, 600);
                });
            });
        });
    }

    function initOverview(c) {
        if (!c) return;
        initLineCharts(c); initDonutCharts(c); initGaugeCharts(c); initScoreCards(c);
        console.log('[Overview] Initialized.');
    }

    window.initOverview = initOverview;
})();
