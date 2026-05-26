// ============================================================
// SHAP WATERFALL CHART
// Plotly.js Renderer
// Scoped module version
// ============================================================

export function initShapWaterfalls(container = document) {

    // ========================================================
    // FIND ALL CHARTS
    // ========================================================

    const charts = container.querySelectorAll(
        '.js-shap-waterfall-render'
    );

    if (!charts || charts.length === 0) {

        console.warn(
            '[SHAP] No shap waterfall containers found.'
        );

        return;
    }

    console.log(
        `[SHAP] Initializing ${charts.length} waterfall chart(s).`
    );

    // ========================================================
    // LOOP CHARTS
    // ========================================================

    charts.forEach((chartEl, chartIndex) => {

        try {

            // ====================================================
            // SAFE JSON PARSER
            // ====================================================

            const safeParse = (value, fallback = null) => {

                try {

                    if (
                        value === undefined ||
                        value === null ||
                        value === ''
                    ) {
                        return fallback;
                    }

                    return JSON.parse(value);

                } catch (err) {

                    console.warn(
                        '[SHAP] JSON parse failed:',
                        err
                    );

                    return fallback;
                }
            };

            // ====================================================
            // DATA EXTRACTION
            // ====================================================

            const title = (
                chartEl.dataset.title ||
                'SHAP Waterfall'
            );

            const sampleIdx = Number(
                chartEl.dataset.sampleIdx ?? 0
            );

            const baseValue = Number(
                safeParse(
                    chartEl.dataset.baseValue,
                    0
                )
            );

            const prediction = Number(
                safeParse(
                    chartEl.dataset.prediction,
                    0
                )
            );

            const maxDisplay = Number(
                safeParse(
                    chartEl.dataset.maxDisplay,
                    0
                )
            );

            const features = safeParse(
                chartEl.dataset.features,
                []
            );

            // ====================================================
            // VALIDATION
            // ====================================================

            if (!Array.isArray(features)) {

                console.error(
                    '[SHAP] Features is not an array.'
                );

                return;
            }

            if (features.length === 0) {

                console.warn(
                    `[SHAP] Empty features for sample ${sampleIdx}`
                );

                return;
            }

            // ====================================================
            // CLEAN FEATURES
            // ====================================================

            const cleanedFeatures = features
                .map((f) => {

                    const shapValue = Number(
                        f.shap_value ?? 0
                    );

                    const absShap = Number(
                        f.abs_shap ?? Math.abs(shapValue)
                    );

                    return {

                        feature:
                            String(
                                f.feature ?? 'Unknown'
                            ),

                        feature_value:
                            f.feature_value ?? 'N/A',

                        shap_value:
                            Number.isFinite(shapValue)
                                ? shapValue
                                : 0,

                        abs_shap:
                            Number.isFinite(absShap)
                                ? absShap
                                : 0
                    };

                })

                // sort ulang untuk safety
                .sort(
                    (a, b) =>
                        b.abs_shap - a.abs_shap
                )

                // limit ulang untuk safety
                .slice(
                    0,
                    maxDisplay > 0
                        ? maxDisplay
                        : features.length
                );

            // ====================================================
            // BUILD WATERFALL ARRAYS
            // ====================================================

            const yLabels = [];
            const xValues = [];
            const customData = [];

            cleanedFeatures.forEach((f) => {

                const shapValue = Number(
                    f.shap_value
                );

                const featureValue =
                    f.feature_value;

                yLabels.push(
                    `${f.feature}`
                );

                xValues.push(
                    shapValue
                );

                customData.push([
                    featureValue,
                    shapValue,
                    f.abs_shap
                ]);
            });

            // ====================================================
            // PLOTLY WATERFALL TRACE
            // ====================================================

            const trace = {

                type: 'waterfall',

                orientation: 'h',

                measure: xValues.map(
                    () => 'relative'
                ),

                y: yLabels,

                x: xValues,

                customdata: customData,

                connector: {

                    line: {

                        color:
                            'rgba(180,180,180,0.35)',

                        width: 1.2
                    }
                },

                increasing: {

                    marker: {

                        color:
                            'rgba(239, 83, 80, 0.90)'
                    }
                },

                decreasing: {

                    marker: {

                        color:
                            'rgba(66, 165, 245, 0.90)'
                    }
                },

                hovertemplate:
                    '<b>%{y}</b><br>' +
                    'Feature Value: %{customdata[0]}<br>' +
                    'SHAP Value: %{customdata[1]:.8f}<br>' +
                    '|SHAP|: %{customdata[2]:.8f}<extra></extra>',

                text: xValues.map((v) => {

                    const sign =
                        v >= 0 ? '+' : '';

                    return (
                        sign +
                        v.toFixed(6)
                    );
                }),

                textposition: 'outside',

                cliponaxis: false
            };

            // ====================================================
            // DYNAMIC HEIGHT
            // ====================================================

            const dynamicHeight = Math.max(
                420,
                cleanedFeatures.length * 48
            );

            // ====================================================
            // LAYOUT
            // ====================================================

            const layout = {

                title: {

                    text:
                        `${title}<br>` +
                        `<span style="font-size:12px;">` +
                        `Sample ${sampleIdx} ` +
                        `| Base Value = ${baseValue.toFixed(8)} ` +
                        `| Prediction = ${prediction.toFixed(8)}` +
                        `</span>`,

                    x: 0.02
                },

                template: 'plotly_dark',

                height: dynamicHeight,

                margin: {

                    l: 220,

                    r: 80,

                    t: 90,

                    b: 60
                },

                paper_bgcolor:
                    'rgba(0,0,0,0)',

                plot_bgcolor:
                    'rgba(0,0,0,0)',

                font: {

                    family:
                        'Inter, sans-serif',

                    size: 12,

                    color: '#EAEAEA'
                },

                showlegend: false,

                hoverlabel: {

                    font: {

                        family:
                            'Inter, sans-serif'
                    }
                },

                xaxis: {

                    title: {

                        text:
                            'SHAP Contribution'
                    },

                    zeroline: true,

                    zerolinewidth: 1.4,

                    zerolinecolor:
                        'rgba(255,255,255,0.25)',

                    gridcolor:
                        'rgba(255,255,255,0.06)',

                    tickfont: {

                        size: 11
                    }
                },

                yaxis: {

                    autorange: 'reversed',

                    automargin: true,

                    tickfont: {

                        size: 11
                    }
                }
            };

            // ====================================================
            // CONFIG
            // ====================================================

            const config = {

                responsive: true,

                displaylogo: false,

                scrollZoom: true,

                doubleClick: 'reset',

                modeBarButtonsToRemove: [

                    'lasso2d',
                    'select2d',
                    'autoScale2d',

                    'hoverClosestCartesian',
                    'hoverCompareCartesian',

                    'toggleSpikelines'
                ]
            };

            // ====================================================
            // RENDER
            // ====================================================

            Plotly.newPlot(
                chartEl,
                [trace],
                layout,
                config
            );

            // ====================================================
            // RESIZE OBSERVER
            // ====================================================

            if (
                typeof ResizeObserver !== 'undefined'
            ) {

                const resizeObserver =
                    new ResizeObserver(() => {

                        Plotly.Plots.resize(
                            chartEl
                        );
                    });

                resizeObserver.observe(
                    chartEl
                );
            }

            console.log(
                `[SHAP] Successfully rendered sample ${sampleIdx}`
            );

        } catch (err) {

            console.error(
                '[SHAP] Fatal render error:',
                err
            );
        }
    });
}
