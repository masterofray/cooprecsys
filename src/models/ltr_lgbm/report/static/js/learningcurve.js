/* =========================================================
   learningcurve.js
   Clean ML-style learning curve renderer
========================================================= */

export function initLearningCurves(scope = document) {
    console.log("[LearningCurve] Initializing.");
    const charts = scope.querySelectorAll(
        ".learningcurve-container .js-learningcurve-render");
    if (!charts.length) {return;}


    // -----------------------------------------------------
    // Helpers
    // -----------------------------------------------------
    function safeParse(value, fallback = null) {
        try {
            return JSON.parse(value);
        } catch (err) {
            console.error(err);
            return fallback;}
        }


    // -----------------------------------------------------
    // Render each chart
    // -----------------------------------------------------
    charts.forEach((chartEl) => {
        const lines = safeParse(chartEl.dataset.lines, []);
        const axis  = safeParse(chartEl.dataset.axis, {});
        const best  = safeParse(chartEl.dataset.bestIteration, null);
        if (!lines.length) {return;}

    const palette = ["#4DA3FF", "#5AD47B", "#9d68f2", "#de1b4b"];


    // -------------------------------------------------
    // Traces
    // -------------------------------------------------
    const traces = [];
    lines.forEach((line, idx) => {
    traces.push({
        x: line.x,
        y: line.y,
        type: "scatter",
        mode: "lines",
        name: line.label,
        line: {
            color: palette[idx % palette.length],
            width: 4,
            shape: "spline",
            smoothing: 1.1,
            dash:line.line_style === "dashed" ? "dash" : "solid" },
        fill: "none",
        hoverlabel: {
            bgcolor: "#13233A",
            bordercolor: palette[idx % palette.length],
            font: {color: "#FFFFFF", size: 15}
            },
        hovertemplate:
            `<b>${line.label}</b><br>` +
            `Round : %{x}<br>` +
            `NDCG : %{y:.6f}<extra></extra>`
            });
        });


    // -------------------------------------------------
    // Best iteration vertical line
    // -------------------------------------------------
    const shapes = [];
    if (best && best.x !== undefined) {
    shapes.push({
        type: "line",
        x0: best.x,
        x1: best.x,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: {
            color: "#FF5252",
            width: 3,
            dash: "dot"}
        });
    }


    // -------------------------------------------------
    // Dynamic Y range
    // -------------------------------------------------
    const allY = lines.flatMap(line => line.y);
    const ymin = Math.min(...allY);
    const ymax = Math.max(...allY);
    const padding = (ymax - ymin) * 0.25;


    // -------------------------------------------------
    // Layout
    // -------------------------------------------------
    const xmax = Math.max(...lines.flatMap(l => l.x));
    const targetTicks = 10;
    const rawStep = xmax / targetTicks;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)) );
    const residual = rawStep / magnitude;
    let niceStep;
    if (residual >= 5) {
        niceStep = 5 * magnitude;
    }
    else if (residual >= 2) {
        niceStep = 2 * magnitude;
    }
    else {
        niceStep = magnitude;
    }
    const dtick = niceStep;

    const layout = {
        autosize: true,
        margin: {l: 90, r: 30, t: 40, b: 80},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#10233B",
        hovermode: "closest",
        font: {
            family: "Inter, Segoe UI, sans-serif",
            color: "#FFFFFF",
            size: 15},
        shapes: shapes,
        xaxis: {
            title: {
                text: axis.x_label || "Boosting Round",
                font: {size: 17, color: "#FFFFFF"}
                },
            dtick : dtick,
            showgrid: true,
            gridcolor: "rgba(255,255,255,0.10)",
            gridwidth: 1,
            zeroline: false,
            showline: true,
            linecolor: "rgba(255,255,255,0.35)",
            linewidth: 2,
            tickfont: {size: 15, color: "#FFFFFF"}
            },
        yaxis: {
            title: {
                text: axis.y_label || "Metric",
                standoff: 28,
                font: {size: 17, color: "#FFFFFF"}
                },
            range: [ymin - padding, ymax + padding],
            showgrid: true,
            gridcolor: "rgba(255,255,255,0.10)",
            gridwidth: 1,
            zeroline: false,
            showline: true,
            linecolor: "rgba(255,255,255,0.35)",
            linewidth: 2,
            tickfont: {size: 15, color: "#FFFFFF"}
            },
        legend: {
            orientation: "h",
            x: 0,
            y: 1.12,
            bgcolor: "rgba(0,0,0,0)",
            font: {size: 15, color: "#FFFFFF"}
            }
        };


    // -------------------------------------------------
    // Config
    // -------------------------------------------------
    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
        scrollZoom: false
        };


    // -------------------------------------------------
    // Render
    // -------------------------------------------------
    Plotly.newPlot(
        chartEl,
        traces,
        layout,
        config).then(() => {
        // immediate resize
        Plotly.Plots.resize(chartEl);

        // delayed resize after layout stabilizes
        setTimeout(() => {Plotly.Plots.resize(chartEl); }, 120);
        setTimeout(() => {Plotly.Plots.resize(chartEl); }, 350);
        });
    
    const resizeObserver = new ResizeObserver(() => {
    Plotly.Plots.resize(chartEl); });
    resizeObserver.observe(chartEl);
    
    });
    console.log("[LearningCurve] Done for the Learning curve chart process.");
}