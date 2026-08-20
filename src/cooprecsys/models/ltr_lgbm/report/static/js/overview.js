// =========================================================
// OVERVIEW CHARTS
// * author   = "Aryanto"
// * modified = "2026-05-24" — line chart + fix donut/gauge canvas sizing
// =========================================================


// =========================================================
// MAIN INIT
// =========================================================
function initOverviewCharts(container) {
    if (!container) return;
    initLineCharts(container);      //dulu initBarCharts, sekarang Line
    initDonutCharts(container);
    initGaugeCharts(container);
    initMainGauge(container);
    console.log("[Overview] Semua chart berhasil dirender.");
}


// =========================================================
// HELPER: calibrateCanvas
// Set attr width/height dari ukuran pixel parent sebelum Chart.js
// membuat instance — mencegah gauge/donut tampil kosong karena
// canvas belum punya dimensi saat dirender.
// =========================================================
function calibrateCanvas(canvas) {
    const parent = canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (w > 0) canvas.setAttribute("width",  w);
    if (h > 0) canvas.setAttribute("height", h);
}


// =========================================================
// LINE CHART
// – Membaca data dari attribute:
//     data-labels = JSON array string (dari bar_labels di check.json)
//     data-values = JSON array number (dari bar_data   di check.json)
// – tension 0.4  → garis lengkung (bukan sudut tajam)
// – warna #38bdf8 → biru terang
// =========================================================
function initLineCharts(container) {
    const charts = container.querySelectorAll(".js-bar-chart");
    charts.forEach((canvas) => {
        let labels = [];
        let values = [];

        // Safe parse — hindari crash jika attribute kosong / malformed
        try { labels = JSON.parse(canvas.dataset.labels || "[]"); } catch(e) { console.warn("[LineChart] parse labels gagal:", e); }
        try { values = JSON.parse(canvas.dataset.values || "[]"); } catch(e) { console.warn("[LineChart] parse values gagal:", e); }

        calibrateCanvas(canvas);

        new Chart(canvas, {
            type: "line",
            data: {
                labels  : labels,
                datasets: [{
                    label                    : "Nilai Metrik",
                    data                     : values,
                    borderColor              : "#38bdf8",
                    backgroundColor          : "rgba(56,189,248,0.10)",
                    borderWidth              : 2.5,
                    tension                  : 0.4,          // lengkung halus
                    fill                     : true,
                    pointBackgroundColor     : "#38bdf8",
                    pointBorderColor         : "#ffffff",
                    pointBorderWidth         : 2,
                    pointRadius              : 5,
                    pointHoverRadius         : 7,
                    pointHoverBackgroundColor: "#ffffff",
                    pointHoverBorderColor    : "#38bdf8",
                }]
            },
            options: {
                responsive          : true,
                maintainAspectRatio : false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        grid : { color: "rgba(26,51,88,0.5)" },
                        ticks: { color: "#5a7aa0", font: { size: 10 } }
                    },
                    y: {
                        grid : { color: "rgba(26,51,88,0.5)" },
                        ticks: { color: "#5a7aa0", font: { size: 10 } }
                    }
                }
            }
        });
        console.log("[LineChart] Render selesai:", canvas.id);
    });
}


// =========================================================
// DONUT CHART
// =========================================================
function initDonutCharts(container) {
    const donutColors = ["#2d7ff9", "#00d68f", "#ffb800", "#a855f7"];
    container.querySelectorAll(".js-donut-chart").forEach((canvas) => {
        const percent    = Number(canvas.dataset.percent    || 0);
        const colorIndex = Number(canvas.dataset.colorIndex || 0);
        calibrateCanvas(canvas);

        new Chart(canvas, {
            type: "doughnut",
            data: {
                datasets: [{
                    data           : [percent, 100 - percent],
                    backgroundColor: [
                        donutColors[colorIndex % donutColors.length],
                        "rgba(26,51,88,0.5)"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive          : true,
                maintainAspectRatio : false,
                cutout              : "72%",
                plugins             : { legend: { display: false } }
            }
        });
        console.log("[DonutChart] Render selesai:", canvas.id);
    });
}


// =========================================================
// GAUGE CHART  (½ lingkaran)
// =========================================================
function initGaugeCharts(container) {
    const gaugeColors = ["#2d7ff9", "#00d68f", "#ffb800", "#ff4d6a", "#a855f7"];
    container.querySelectorAll(".js-gauge-chart").forEach((canvas) => {
        const percent    = Number(canvas.dataset.percent    || 0);
        const colorIndex = Number(canvas.dataset.colorIndex || 0);
        calibrateCanvas(canvas);

        new Chart(canvas, {
            type: "doughnut",
            data: {
                datasets: [{
                    data           : [percent, 100 - percent],
                    backgroundColor: [
                        gaugeColors[colorIndex % gaugeColors.length],
                        "rgba(26,51,88,0.4)"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive          : true,
                maintainAspectRatio : false,
                cutout              : "75%",
                rotation            : -90,
                circumference       : 180,
                plugins             : { legend: { display: false } }
            }
        });
        console.log("[GaugeChart] Render selesai:", canvas.id);
    });
}


// =========================================================
// MAIN GAUGE
// =========================================================
function initMainGauge(container) {
    const canvas = container.querySelector(".js-main-gauge");
    if (!canvas) { return; }
    const percent = Number(canvas.dataset.percent || 0);
    calibrateCanvas(canvas);

    new Chart(canvas, {
        type: "doughnut",
        data: {
            datasets: [{
                data           : [percent, 100 - percent],
                backgroundColor: ["#daf7e5", "rgba(26,51,88,0.4)"], // color Final gauge
                borderWidth    : 0
            }]
        },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            cutout              : "78%",
            rotation            : -90,
            circumference       : 180,
            plugins             : { legend: { display: false } }
        }
    });
    console.log("[MainGauge] Render selesai."); }

export default function initOverview(container) {
    console.log("[Overview] Initialized Module");
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            initOverviewCharts(container);
        });
    }); }

