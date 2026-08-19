/*
* author     = "Aryanto"
* copyright  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
* license    = "GNUPublic"
* version    = "0.0.1"
* email      = "aryanto.dandan@gmail.com"
* status     = "Development"
* created    = "2026-05-25"
* About
  - histrelevance.js
  - Manajemen visualisasi histogram relevance dengan verbose debugging.
 */

export function initHistRelevance(rootElement = document) {
    console.log("[HistRelevance] Initializing module.");
    const containers = rootElement.querySelectorAll(".histrelevance-container");
    if (!containers.length) {
        console.log("[HistRelevance] No histogram containers found.");
        return;}
    containers.forEach((container) => {
    const canvas = container.querySelector(".js-histrelevance-chart");
    if (!canvas) {console.warn(
        "[HistRelevance] Canvas not found in container.",
        container);
        return;}

    try {
        const rawValues = canvas.dataset.values;
        const rawBins   = canvas.dataset.bins;
        const rawStats  = canvas.dataset.stats;
        const values = JSON.parse(rawValues || "[]");
        const bins   = JSON.parse(rawBins || "[]");
        const stats  = JSON.parse(rawStats || "{}");
        const xLabel = canvas.dataset.xLabel || "";
        const yLabel = canvas.dataset.yLabel || "";

        if (!values.length || bins.length < 2) {
            console.warn("[HistRelevance] Invalid histogram data.",
            canvas.id);
            return; }

        const histogramCounts = buildHistogram(values, bins);
        const labels = buildBinLabels(bins);
        const smoothLine = smoothHistogram(histogramCounts);
        const ctx = canvas.getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {labels: labels,
                   datasets: [
                    {type               : "bar",
                     label              : "Frequency",
                     data               : histogramCounts,
                     categoryPercentage : 0.98,
                     barPercentage      : 0.96,
                     borderRadius       : 0,
                    },
                    {type       : "line",
                     label      : "Density",
                     data       : smoothLine,
                     tension    : 0.42,
                     pointRadius: 0,
                     borderWidth: 3,
                     fill: false,
                    }]
                  },
            options: {responsive: true,
                      maintainAspectRatio: false,
            plugins: {legend: {display: false},
            tooltip: {callbacks: {
                afterBody: () => {return [
                `Mean: ${stats.mean?.toFixed(4) ?? "-"}`,
                `Std: ${stats.std?.toFixed(4) ?? "-"}`,
                ];
                }}}},
            scales: {
                x: {title: {display : true,
                            text    : xLabel,
                            color   : "#FFFFFF",
                            font    : {size: 18, weight: "600"},
                            padding : {top: 18}
                            },
                    ticks: {color   : "#FFFFFF",
                            font    : {size: 12}}
                   },
                y: {title: {display : true,
                            text    : yLabel,
                            color   : "#FFFFFF",
                            font    : {size: 18, weight: "600"},
                            padding : {bottom: 12}
                           },
                    ticks: {color   : "#FFFFFF",
                            font    : {size: 12}},
                beginAtZero : true}
                }
            }});
        console.log(`[HistRelevance] Rendered histogram: ${canvas.id}`);
    } catch (error) {
        console.error("[HistRelevance] Failed rendering histogram.", error);}
    });
}


/* Build histogram counts from values + bins. */
function buildHistogram(values, bins) {
    const counts = new Array(bins.length - 1).fill(0);
    values.forEach((value) => {
        for (let i = 0; i < bins.length - 1; i++) {
            const left  = bins[i];
            const right = bins[i + 1];
            const isLastBin = i === bins.length - 2;
            if (
                (value >= left && value < right) ||
                (isLastBin && value === right)
            ) {
                counts[i]++;
                break;
            }}
        });
    return counts;}


/* Build readable labels from bin edges. */
function buildBinLabels(bins) {
    const labels = [];
    for (let i = 0; i < bins.length - 1; i++) {
        const left  = Number(bins[i]).toFixed(2);
        const right = Number(bins[i + 1]).toFixed(2);
        labels.push(`${left} – ${right}`);
        }
    return labels;}


function smoothHistogram(values) {
    const result = [];
    for (let i = 0; i < values.length; i++) {
        const prev = values[i - 1] ?? values[i];
        const curr = values[i];
        const next = values[i + 1] ?? values[i];
        result.push((prev + curr + next) / 3);
        }
    return result;
    }