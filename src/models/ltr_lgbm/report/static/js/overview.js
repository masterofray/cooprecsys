// =========================================================
// OVERVIEW CHARTS
// =========================================================
document.addEventListener("DOMContentLoaded", () => {
    console.log("[Overview] Created by Aryanto")
    console.log("[Overview] DOM selesai dimuat. Memulai render chart.");
    initOverviewCharts();
    });


// =========================================================
// MAIN INIT
// =========================================================
function initOverviewCharts() {
    initBarCharts();
    initDonutCharts();
    initGaugeCharts();
    initMainGauge();
    console.log("[Overview] Semua chart berhasil dirender.");
    }


// =========================================================
// BAR CHART
// =========================================================
function initBarCharts() {
    const charts = document.querySelectorAll(".js-bar-chart");
    charts.forEach((canvas) => {
        const labels = JSON.parse(canvas.dataset.labels || "[]");
        const values = JSON.parse(canvas.dataset.values || "[]");

    new Chart(canvas, {
        type: "bar",
        data: {labels          : labels,
               datasets        : [{
               label           : "Predictions",
               data            : values,
               backgroundColor : "rgba(45, 127, 249, 0.6)",
               borderColor     : "#2d7ff9",
               borderWidth     : 1,
               borderRadius    : 6,
               barPercentage   : 0.7,}] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {legend: {display: false}},
            scales: {
                x: {grid  : {color:"rgba(26,51,88,0.5)"},
                    ticks : {color: "#5a7aa0",
                    font  : {size: 10}}
                   },

                y: {grid  : {color: "rgba(26,51,88,0.5)"},
                    ticks : {color: "#5a7aa0",
                    font  : {size: 10}}
                    },
                }}
        });
    console.log("[BarChart] Render selesai:", canvas.id);
    });
    }



// =========================================================
// DONUT CHART
// =========================================================
function initDonutCharts() {
    const donutColors = ["#2d7ff9", "#00d68f",
                         "#ffb800","#a855f7"];
    const charts = document.querySelectorAll(".js-donut-chart");
    charts.forEach((canvas) => {
    const percent = Number(canvas.dataset.percent || 0);
    const colorIndex = Number(canvas.dataset.colorIndex || 0);
    new Chart(canvas, {
        type: "doughnut", 
        data: {datasets: [
           {data            : [percent, 100 - percent],
            backgroundColor : [donutColors[
                              colorIndex % donutColors.length],
                              "rgba(26,51,88,0.5)"],
            borderWidth     : 0}] },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            cutout              : "72%",
            plugins             : {legend: {display: false}}
            } });
    console.log("[DonutChart] Render selesai:", canvas.id);
    }); }



// =========================================================
// GAUGE CHART
// =========================================================
function initGaugeCharts() {
    const gaugeColors = ["#2d7ff9", "#00d68f",
                         "#ffb800", "#ff4d6a",
                         "#a855f7"];
    const charts = document.querySelectorAll(".js-gauge-chart");
    charts.forEach((canvas) => {
    const percent = Number(canvas.dataset.percent || 0);
    const colorIndex = Number(canvas.dataset.colorIndex || 0);
    new Chart(canvas, {
        type: "doughnut",
        data: {datasets: [{
            data: [percent, 100 - percent],
            backgroundColor: [gaugeColors[
                              colorIndex % gaugeColors.length],
                              "rgba(26,51,88,0.4)"],
            borderWidth: 0}] },
        options: {
            responsive          : true,
            maintainAspectRatio : false,
            cutout              : "75%",
            rotation            : -90,
            circumference       : 180,
            plugins             : {legend: {display: false}}
            }
        });
    console.log("[GaugeChart] Render selesai:", canvas.id);
    }); }



// =========================================================
// MAIN GAUGE
// =========================================================
function initMainGauge() {
    const canvas = document.querySelector(".js-main-gauge");
    if (!canvas) {return;}
    const percent = Number(canvas.dataset.percent || 0);
    new Chart(canvas, {
        type: "doughnut",
        data: {datasets: [{
            data: [percent, 100 - percent],
            backgroundColor: ["#2d7ff9", "rgba(26,51,88,0.4)"],
            borderWidth: 0}] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "78%",
            rotation: -90,
            circumference: 180,
            plugins: {legend: {display: false}}
            }
        });
    console.log("[MainGauge] Render selesai.");
    }

