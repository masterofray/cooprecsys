/**
 * diagnostics.js — KPI charts, grouped bar charts
 */

(function () {
    'use strict';

    function alpha(hex, a) { var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16); return 'rgba('+r+','+g+','+b+','+a+')'; }
    function safeParse(v, f) { if (f === undefined) f = []; try { return JSON.parse(v); } catch(e) { return f; } }

    function initBarCharts(container) {
        container.querySelectorAll('.js-bar-chart').forEach(function (canvas) {
            var labels = safeParse(canvas.dataset.labels), values = safeParse(canvas.dataset.values);
            if (!labels.length || !values.length) return;
            var barHeight = 22, barSpacing = 16, padding = 60;
            var h = (labels.length * (barHeight + barSpacing)) + padding;
            var wrapper = canvas.parentElement;
            if (wrapper) { wrapper.style.height = h + 'px'; wrapper.style.minHeight = h + 'px'; }

            var bg = values.map(function (_, i) { return i === 0 ? '#c62828' : 'rgba(21,101,192,0.75)'; });
            var bd = values.map(function (_, i) { return i === 0 ? '#ef5350' : 'rgba(33,150,243,1)'; });

            new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: { labels: labels, datasets: [{ label: 'Value', data: values, backgroundColor: bg, borderColor: bd, borderWidth: 1, borderRadius: 4, barThickness: barHeight }] },
                options: {
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: false,
                    plugins: { legend: { display: false } },
                    layout: { padding: { left: 10, right: 25, top: 5, bottom: 5 } },
                    scales: {
                        x: { beginAtZero: true, ticks: { color: '#3d6b3d' }, grid: { color: 'rgba(45,90,45,0.08)' } },
                        y: { ticks: { color: '#3d6b3d', autoSkip: false }, grid: { display: false } }
                    }
                }
            });
        });
    }

    function initGroupedBarCharts(container) {
        container.querySelectorAll('.js-grouped-bar-chart').forEach(function (canvas) {
            var labels = safeParse(canvas.dataset.labels), datasetsRaw = safeParse(canvas.dataset.datasets);
            if (!labels.length || !datasetsRaw.length) return;
            var palette = ['#1565c0', '#2e7d32', '#6a1b9a'];
            var datasets = datasetsRaw.map(function (ds, i) {
                return { label: ds.label, data: ds.data, backgroundColor: alpha(palette[i % palette.length], 0.7), borderColor: palette[i % palette.length], borderWidth: 1, borderRadius: 4 };
            });
            new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: { labels: labels, datasets: datasets },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'top', labels: { padding: 20 } },
                        tooltip: { backgroundColor: '#fff', titleColor: '#1a3a1a', bodyColor: '#3d6b3d', borderColor: 'rgba(45,90,45,0.15)', borderWidth: 1, padding: 12, cornerRadius: 8 }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(45,90,45,0.08)' } },
                        y: { grid: { color: 'rgba(45,90,45,0.08)' }, beginAtZero: true }
                    }
                }
            });
        });
    }

    function initDiagnostics(container) {
        if (!container) return;
        initBarCharts(container);
        initGroupedBarCharts(container);
        console.log('[Diagnostics] Initialized.');
    }

    window.initDiagnostics = initDiagnostics;
})();
