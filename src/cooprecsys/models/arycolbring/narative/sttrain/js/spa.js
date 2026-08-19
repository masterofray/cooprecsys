/**
 * spa.js — SPA Navigation Router
 * Based on reference spa.js + dashboard.js
 */

(function () {
    'use strict';

    window.switchPage = function (page) {
        // Hide all pages
        document.querySelectorAll('.page-content').forEach(function (el) {
            el.classList.remove('active');
        });
        document.querySelectorAll('.page-tab').forEach(function (el) {
            el.classList.remove('active');
        });
        document.querySelectorAll('.sidebar-link').forEach(function (el) {
            el.classList.remove('active');
        });

        // Show selected
        var target = document.getElementById('page-' + page);
        if (target) target.classList.add('active');

        // Activate tab
        document.querySelectorAll('.page-tab').forEach(function (el) {
            if (el.dataset.tab === page) el.classList.add('active');
        });

        // Activate sidebar
        document.querySelectorAll('.sidebar-link').forEach(function (el) {
            if (el.dataset.page === page) el.classList.add('active');
        });

        localStorage.setItem('dashboard_page', page);
        window.dispatchEvent(new Event('resize'));
    };

    document.addEventListener('DOMContentLoaded', function () {
        // Bind navigation clicks
        document.querySelectorAll('.sidebar-link, .page-tab').forEach(function (el) {
            el.addEventListener('click', function () {
                var page = el.dataset.page || el.dataset.tab;
                if (page) window.switchPage(page);
            });
        });

        // Restore last active page
        var saved = localStorage.getItem('dashboard_page') || 'overview';
        window.switchPage(saved);
    });
})();
