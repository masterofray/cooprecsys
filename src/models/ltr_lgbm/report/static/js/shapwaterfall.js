// ============================================================
// SHAP WATERFALL CHART
// Plotly.js Renderer
// author   = "Aryanto"
// modified = "2026-05-27"
// ============================================================

export function initShapWaterfalls(container = document) {
    const charts = container.querySelectorAll(
    '.js-shap-waterfall-render');
    if (!charts || charts.length === 0) {
        console.warn('[SHAP] No shap waterfall containers found.');
        return; }
    console.log(`[SHAP] Initializing ${charts.length} waterfall chart(s).`);

    // LOOP CHARTS
    let resizeAttached = false;
    charts.forEach((chartEl, chartIndex) => {
        try {

        // SAFE JSON PARSER
        const safeParse = (value, fallback = null) => {
        try {
            if (value === undefined ||
                value === null ||
                value === '')
                {return fallback;}
                return JSON.parse(value);
            } catch (err) {
            console.warn('[SHAP] JSON parse failed:', err);
            return fallback;}
            };

        // DATA EXTRACTION
        const title = (chartEl.dataset.title || 'SHAP Waterfall');
        const sampleIdx = Number(chartEl.dataset.sampleIdx ?? 0);
        const baseValue = Number(safeParse(chartEl.dataset.baseValue, 0) );
        const prediction = Number(safeParse(chartEl.dataset.prediction, 0) );
        const maxDisplay = Number(safeParse(chartEl.dataset.maxDisplay, 0) );
        const features = safeParse(chartEl.dataset.features, []);
        console.log(`[SHAP] Sample ${sampleIdx}`,
            {title, baseValue, prediction, maxDisplay,
             totalFeatures: features.length} );

        // VALIDATION
        if (!Array.isArray(features)) {console.error(
            '[SHAP] Features is not an array.');
            return; }
        if (features.length === 0) {console.warn(
            `[SHAP] Empty features for sample ${sampleIdx}` );
            return; }

        // CLEAN FEATURES
        let cleanedFeatures = features.map((f) => {
        const shapValue = Number(f.shap_value ?? 0);
        const absShap = Number(f.abs_shap ?? Math.abs(shapValue));
        return {feature       : String(f.feature ?? 'Unknown'),
                feature_value : f.feature_value ?? 'N/A',
                shap_value    : Number.isFinite(shapValue) ? shapValue : 0,
                abs_shap      : Number.isFinite(absShap) ? absShap : 0 };
        });
        console.log('[SHAP] Raw cleaned features:', cleanedFeatures);

        // REMOVE DUPLICATE FEATURES
        const uniqueMap = new Map();
        cleanedFeatures.forEach((f) => {
        if (!uniqueMap.has(f.feature)) {
            uniqueMap.set(f.feature, f);
            return; }
        const existing = uniqueMap.get(f.feature);
        if (Math.abs(f.shap_value) > Math.abs(existing.shap_value)) {
            uniqueMap.set(f.feature, f); }
            });
        cleanedFeatures = Array.from(uniqueMap.values());
        console.log('[SHAP] Unique features:', cleanedFeatures);

        // SORT + LIMIT
        cleanedFeatures = cleanedFeatures.sort(
            (a, b) => b.abs_shap - a.abs_shap).slice(0,
            maxDisplay > 0 ? maxDisplay : cleanedFeatures.length);
        console.log('[SHAP] Final cleaned features:', cleanedFeatures);

        // BUILD WATERFALL ARRAYS
        const displayLabels = [];
        const xValues       = [];
        const customData    = [];
        cleanedFeatures.forEach((f) => {
            const shapValue      = Number(f.shap_value);
            const featureValue   = f.feature_value;
            const fullFeature    = String(f.feature);
            const displayFeature = fullFeature.length > 32
                                   ? fullFeature.slice(0, 32) + '…' : fullFeature;
            displayLabels.push(displayFeature);
            xValues.push(shapValue);
            customData.push([fullFeature, featureValue,
                             shapValue, f.abs_shap]);
            });

        // PLOTLY WATERFALL TRACE
        const trace = {
            type          : 'waterfall',
            orientation   : 'h',
            measure       : xValues.map(() => 'relative'),
            y             : displayLabels,
            x             : xValues,
            customdata    : customData,
            connector     : {line: {color:'rgba(180,180,180,0.35)',
                             width: 1.2} },
            increasing    : {marker: {color:'rgba(239, 83, 80, 0.90)'}},
            decreasing    : {marker: {color:'rgba(66, 165, 245, 0.90)'}},
            hovertemplate : '<b>%{customdata[0]}</b><br>' +
                            'Feature Value: %{customdata[1]}<br>' +
                            'SHAP Value: %{customdata[2]:.8f}<br>' +
                            '|SHAP|: %{customdata[3]:.8f}<extra></extra>',
            text          : xValues.map((v) => {
                            const absValue = Math.abs(v);
                            // nilai kecil jangan terlalu panjang
                            if (absValue < 0.01) {return v.toFixed(4);}
                            return v.toFixed(5);
                            }),
            textposition  : 'outside',
            cliponaxis    : false,
            constraintext : 'none',
            textangle     : 0,
            textfont      : {size: 10, family: 'Inter, sans-serif'},
            };

        // DYNAMIC HEIGHT
        const dynamicHeight = Math.max(320, 
            cleanedFeatures.length * 36);

        // LAYOUT
        const layout = {
            template      : 'plotly_dark',
            height        : dynamicHeight,
            margin        : {l: 140, r: 70, t: 10, b: 50, pad: 2},
            paper_bgcolor : 'rgba(0,0,0,0)',
            plot_bgcolor  : 'rgba(0,0,0,0)',
            font          : {family: 'Inter, sans-serif', 
                             size: 12, color: '#EAEAEA'},
            showlegend    : false,
            hoverlabel    : {font: {family: 'Inter, sans-serif'}},
            xaxis         : {title: {text:'SHAP Contribution'},
                             zeroline: true,
                             zerolinewidth: 1.4,
                             zerolinecolor: 'rgba(255,255,255,0.25)',
                             gridcolor: 'rgba(255,255,255,0.06)',
                             tickfont: {size: 11}
                             },
            yaxis         : {autorange: 'reversed',
                             automargin: false,
                             ticklabelposition: 'outside',
                             tickfont: {size: 11}
                             }
            };

        // CONFIG PLOTLY TOOLBAR
        const config = {responsive     : true,
                        displayModeBar : false,
                        displaylogo    : false,
                        scrollZoom     : false};

        // RENDER
        console.log(`[SHAP] Rendering waterfall`, {displayLabels, xValues});
        console.log(`[SHAP] Render sample ${sampleIdx}`, {
            features : cleanedFeatures.length,
            labels   : displayLabels.length,
            values   : xValues.length});
        Plotly.newPlot(chartEl, [trace], layout, config);
        if (!resizeAttached) {
            resizeAttached = true;
            window.addEventListener('resize', () => {
                if (!chartEl || !chartEl.offsetParent) {
                    return; }
                console.log(`[SHAP] Window resize sample ${sampleIdx}`);
                Plotly.Plots.resize(chartEl);
                });
            }

        // RESIZE OBSERVER
        console.log(`[SHAP] Successfully rendered sample ${sampleIdx}`);
    } catch (err) {
        console.error('[SHAP] Fatal render error:', err);
        }
    });
}
