/**
 * overview.js — Overview page charts (line, donut, gauge)
 * Based on reference overview.js patterns
 * Reads data from data-* attributes on HTML elements
 */

(function () {
    'use strict';

    // ── Helpers ──
    function safeParse(value, fallback) {
        if (fallback === undefined) fallback = [];
        try { return JSON.parse(value); }
        catch (e) { console.warn('[Overview] Parse fail:', e); return fallback; }
    }

    function calibrateCanvas(canvas) {
        var parent = canvas.parentElement;
        if (!parent) return;
        var w = parent.clientWidth;
        var h = parent.clientHeight;
        if (w > 0) canvas.setAttribute('width', w);
        if (h > 0) canvas.setAttribute('height', h);
    }

    function alpha(hex, a) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
    }

    // ── Line Chart ──
    function initLineCharts(container) {
        container.querySelectorAll('.js-line-chart').forEach(function (canvas) {
            var labels = safeParse(canvas.dataset.labels);
            var values = safeParse(canvas.dataset.values);
            if (!labels.length || !values.length) return;

            calibrateCanvas(canvas);
            var ctx = canvas.getContext('2d');
            var h = canvas.parentElement.clientHeight || 280;
            var gradient = ctx.createLinearGradient(0, 0, 0, h);
            gradient.addColorStop(0, alpha('#FF6B35', 0.18));
            gradient.addColorStop(1, alpha('#FF6B35', 0.0));

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Metric Value',
                        data: values,
                        borderColor: '#38bdf8',
                        backgroundColor: gradient,
                        borderWidth: 2.5,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#38bdf8',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#ffffff',
                        pointHoverBorderColor: '#38bdf8'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#ffffff',
                            titleColor: '#212529',
                            bodyColor: '#495057',
                            borderColor: 'rgba(33,37,41,0.15)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            callbacks: { label: function (c) { return c.parsed.y.toFixed(4); } }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(33,37,41,0.08)' }, ticks: { color: '#868e96', font: { size: 10 } } },
                        y: { grid: { color: 'rgba(33,37,41,0.08)' }, ticks: { color: '#868e96', font: { size: 10 } } }
                    }
                }
            });
            console.log('[Overview] LineChart rendered:', canvas.id);
        });
    }

    // ── Donut Chart ──
    function initDonutCharts(container) {
        var colors = ['#FF6B35', '#4ECDC4', '#f9a825', '#6a1b9a'];
        container.querySelectorAll('.js-donut-chart').forEach(function (canvas) {
            var percent = Number(canvas.dataset.percent || 0);
            var ci = Number(canvas.dataset.colorIndex || 0);
            calibrateCanvas(canvas);

            new Chart(canvas, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [percent, 100 - percent],
                        backgroundColor: [colors[ci % colors.length], 'rgba(33,37,41,0.08)'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '72%',
                    plugins: { legend: { display: false } }
                }
            });
        });
    }

    // ── Gauge Chart (half doughnut) ──
    function initGaugeCharts(container) {
        var gaugeColors = ['#FF6B35', '#4ECDC4', '#f9a825', '#c62828', '#6a1b9a',
                           '#00838f', '#e65100', '#00695c', '#ad1457', '#4ECDC4', '#FF6B35'];
        container.querySelectorAll('.js-gauge-chart').forEach(function (canvas) {
            var percent = Number(canvas.dataset.percent || 0);
            var ci = Number(canvas.dataset.colorIndex || 0);
            calibrateCanvas(canvas);

            new Chart(canvas, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [percent, 100 - percent],
                        backgroundColor: [gaugeColors[ci % gaugeColors.length], 'rgba(33,37,41,0.06)'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '75%',
                    rotation: -90,
                    circumference: 180,
                    plugins: { legend: { display: false } }
                }
            });
        });
    }

    // ── Score Card Click-to-Copy ──
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

    // ── Main Init ──
    function initOverview(container) {
        if (!container) return;
        initLineCharts(container);
        initDonutCharts(container);
        initGaugeCharts(container);
        initScoreCards(container);
        console.log('[Overview] All charts initialized.');
    }

    window.initOverview = initOverview;
})();
