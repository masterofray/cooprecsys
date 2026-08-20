/**
 * dashboard.js — Main orchestrator
 * Sets Chart.js defaults and calls per-page init functions
 */

(function () {
    'use strict';

    function setChartDefaults() {
        if (typeof Chart === 'undefined') return;
        Chart.defaults.color = '#495057';
        Chart.defaults.borderColor = 'rgba(33, 37, 41, 0.12)';
        Chart.defaults.font.family = "'Inter','Segoe UI','Roboto',sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.plugins.legend.labels.boxWidth = 14;
        Chart.defaults.plugins.legend.labels.padding = 16;
    }

    document.addEventListener('DOMContentLoaded', function () {
        setChartDefaults();

        // Init all page modules
        var overview = document.getElementById('page-overview');
        if (overview && window.initOverview) window.initOverview(overview);

        var rankings = document.getElementById('page-rankings');
        if (rankings && window.initRankings) window.initRankings(rankings);

        var diagnostics = document.getElementById('page-diagnostics');
        if (diagnostics && window.initDiagnostics) window.initDiagnostics(diagnostics);

        var config = document.getElementById('page-config');
        if (config && window.initConfig) window.initConfig(config);

        console.log('[Dashboard] All modules loaded.');
    });
})();
