/**
 * rankings.js — Predictions table filtering, hover, confidence badges
 */

(function () {
    'use strict';

    function initHoverEffects(table) {
        if (!table) return;
        table.querySelectorAll('tbody tr').forEach(function (row) {
            row.addEventListener('mouseenter', function () { row.classList.add('row-hover'); });
            row.addEventListener('mouseleave', function () { row.classList.remove('row-hover'); });
        });
    }

    function evaluateNumericExpression(expr, x) {
        expr = expr.replace(/\s+/g, ' ').trim().toLowerCase();
        var rm = expr.match(/^(-?\d+(\.\d+)?)\s*(<=|<)\s*x\s*(<=|<)\s*(-?\d+(\.\d+)?)$/);
        if (rm) {
            var lv = parseFloat(rm[1]), lo = rm[3], ro = rm[4], rv = parseFloat(rm[5]);
            return (lo === '<=' ? lv <= x : lv < x) && (ro === '<=' ? x <= rv : x < rv);
        }
        if (expr.indexOf(' and ') > -1) return expr.split(/\sand\s/).every(function (c) { return evalSimple(c, x); });
        if (expr.indexOf(' or ') > -1) return expr.split(/\sor\s/).some(function (c) { return evalSimple(c, x); });
        return evalSimple(expr, x);
    }

    function evalSimple(cond, x) {
        cond = cond.trim().toLowerCase().replace(/^x\s*/, '');
        var m = cond.match(/^(<=|>=|!=|=|<|>)(.*)$/);
        if (!m) return String(x).indexOf(cond) > -1;
        var op = m[1], cv = parseFloat(m[2]);
        if (isNaN(cv)) return false;
        switch (op) {
            case '>': return x > cv; case '>=': return x >= cv;
            case '<': return x < cv; case '<=': return x <= cv;
            case '=': return x === cv; case '!=': return x !== cv;
            default: return false;
        }
    }

    function initTableFiltering(table) {
        if (!table) return;
        var filters = table.querySelectorAll('.table-filter');
        var rows = table.querySelectorAll('tbody tr');

        filters.forEach(function (filter) {
            filter.addEventListener('input', function () {
                rows.forEach(function (row) {
                    var visible = true;
                    filters.forEach(function (f) {
                        var q = f.value.trim();
                        if (!q) return;
                        var col = f.dataset.column;
                        var cell = row.querySelector('td[data-column="' + col + '"]');
                        if (!cell) { visible = false; return; }
                        var text = cell.textContent.trim();
                        var num = parseFloat(text.replace(/,/g, ''));
                        if (!isNaN(num)) {
                            if (!evaluateNumericExpression(q, num)) visible = false;
                        } else {
                            if (text.toLowerCase().indexOf(q.toLowerCase()) === -1) visible = false;
                        }
                    });
                    row.style.display = visible ? '' : 'none';
                });
            });
        });
    }

    function initKeyboardShortcuts(container) {
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                container.querySelectorAll('.table-filter').forEach(function (input) {
                    input.value = '';
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                });
                container.classList.add('filters-cleared');
                setTimeout(function () { container.classList.remove('filters-cleared'); }, 300);
                var scrollBox = container.querySelector('.table-scroll');
                if (scrollBox) scrollBox.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
            }
        });
    }

    window.exportTableCSV = function () {
        var table = document.getElementById('predictionsTable');
        if (!table) return;
        var rows = table.querySelectorAll('tr');
        var csv = [];
        rows.forEach(function (row) {
            var cols = row.querySelectorAll('th, td');
            var rd = [];
            cols.forEach(function (col) {
                var input = col.querySelector('input');
                var text = input ? '' : col.textContent.replace(/,/g, '').trim();
                rd.push('"' + text + '"');
            });
            csv.push(rd.join(','));
        });
        var blob = new Blob([csv.join('\n')], { type: 'text/csv' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'predictions_export.csv'; a.click();
        URL.revokeObjectURL(url);
    };

    function initRankings(container) {
        if (!container) return;
        var table = container.querySelector('.rankings-table');
        initHoverEffects(table);
        initTableFiltering(table);
        initKeyboardShortcuts(container);
        console.log('[Rankings] Initialized.');
    }

    window.initRankings = initRankings;
})();
