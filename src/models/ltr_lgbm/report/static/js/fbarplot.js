/**
 * barplot.js
 * Menangani ekstraksi data dari DOM dan rendering Chart.js
 */

document.addEventListener("DOMContentLoaded", () => {
    console.log("[Init] DOM selesai dimuat. Memulai pipeline rendering chart...");
    initBarCharts();
});

// ==========================================
// 1. MAIN INITIALIZER
// ==========================================
function initBarCharts() {
    // Menarik semua canvas yang memiliki class js-bar-chart
    const chartElements = document.querySelectorAll('.js-bar-chart');
    console.log(`[Init] Ditemukan ${chartElements.length} elemen chart yang siap diproses.`);

    chartElements.forEach((canvas, index) => {
        console.group(`\n--- Memproses Chart #${index} (ID: ${canvas.id}) ---`);
        
        // Step A: Ekstrak Data dari HTML
        const rawData = extractDataFromElement(canvas);
        if (!rawData) {
            console.groupEnd();
            return; // Lewati jika data gagal ditarik
        }

        // Step B: Normalisasi Format Data
        const processedData = normalizeChartData(rawData.labels, rawData.values);
        if (!processedData) {
            handleChartError(canvas, "Data kosong atau tidak valid setelah normalisasi.");
            console.groupEnd();
            return;
        }

        // Step C: Render menggunakan Chart.js
        renderChartOnCanvas(canvas, processedData);
        console.log("[Init] Berhasil render dengan Chart.js");
        console.groupEnd();
    });
}

// ==========================================
// 2. DATA EXTRACTION
// ==========================================
function extractDataFromElement(canvas) {
    console.log(`[Data Flow - 1] Menarik data dari atribut data-* pada ${canvas.id}...`);
    try {
        const labelsAttr = canvas.getAttribute('data-labels');
        const valuesAttr = canvas.getAttribute('data-values');

        if (labelsAttr === null || valuesAttr === null) {
            throw new Error("Atribut 'data-labels' atau 'data-values' tidak ditemukan.");
        }

        if (labelsAttr.trim() === "" || valuesAttr.trim() === "") {
            throw new Error("String JSON kosong.");
        }

        // --- AKTIFKAN KEDUA BARIS INI UNTUK MELIHAT STRING ASLI DI CONSOLE ---
        console.log(`[${canvas.id}] RAW Labels String dari HTML:`, labelsAttr);
        console.log(`[${canvas.id}] RAW Values String dari HTML:`, valuesAttr);

        // Parse JSON
        const labels = JSON.parse(labelsAttr);
        const values = JSON.parse(valuesAttr);

        console.log(`[Data Flow - 1] Sukses menarik data.`);
        return { labels, values };
    } catch (error) {
        console.error(`[Data Flow Error] Gagal memparsing JSON di ${canvas.id}. Alasan:`, error.message);
        handleChartError(canvas, "Gagal membaca data grafik dari HTML.");
        return null;
    }
}

// ==========================================
// 3. DATA NORMALIZATION
// ==========================================
function normalizeChartData(rawLabels, rawValues) {
    console.log("[Data Flow - 2] Memulai normalisasi struktur data...");
    let labels = [];
    let values = [];

    // CASE 1: values = [[label,val], [label,val]]
    if (Array.isArray(rawValues) && rawValues.length > 0 && Array.isArray(rawValues[0])) {
        console.log("[Data Flow - 2] Terdeteksi format Nested Array [[label, value], ...]");
        labels = rawValues.map(v => v[0]);
        values = rawValues.map(v => {
            const num = Number(v[1]);
            return isNaN(num) ? 0 : num;
        });
    } 
    // CASE 2: values = [1,2,3] (Simple Array)
    else {
        console.log("[Data Flow - 2] Terdeteksi format Simple Array (labels dan values terpisah).");
        labels = rawLabels;
        values = rawValues.map(v => {
            const num = Number(v);
            return isNaN(num) ? 0 : num;
        });
    }

    console.log("[Data Flow - 2] Normalisasi selesai.");
    console.log(`  -> Final Labels (${labels?.length || 0} items):`, labels);
    console.log(`  -> Final Values (${values?.length || 0} items):`, values);

    // Validasi akhir sebelum masuk ke Chart.js
    // Check data apakah empty atau tidak!
    if (!labels || !labels.length || !values || !values.length) {
        console.warn("[Data Validation] Peringatan: Array labels atau values kosong!");
    return null;}
    return { labels, values };
}

// ==========================================
// 4. CHART RENDERING
// ==========================================
function renderChartOnCanvas(canvas, data) {
    console.log(`[Render] Mengecek dependensi Chart.js untuk ${canvas.id}...`);

    if (typeof Chart === "undefined") {
        console.error("[Render Error] Library Chart.js belum diload di halaman ini!");
        handleChartError(canvas, "Chart.js tidak ditemukan.");
        return;
    }

    console.log(`[Render] Chart.js tersedia. Membersihkan canvas jika ada chart sebelumnya...`);
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
        console.log(`[Render] Instance chart lama berhasil dihancurkan.`);
    }

    console.log(`[Render] Mengeksekusi 'new Chart()' ...`);
    // 1. HITUNG TINGGI CONTAINER SECARA DINAMIS
    const barHeight = 22;       // Ukuran batang bar (px)
    const barSpacing = 16;      // Jarak spasi vertikal antar bar (px)
    const paddingTotal = 60;    // Alokasi space untuk sumbu X (Top & Bottom Padding)
    
    // Formula: (Jumlah Data x Jarak Total Per Bar) + Padding Sumbu
    const calculatedHeight = (data.labels.length * (barHeight + barSpacing)) + paddingTotal;

    // 2. TERAPKAN TINGGI LANGSUNG KE PARENT WRAPPER & CANVAS
    const wrapper = canvas.parentElement;
    if (wrapper) {
        wrapper.style.height = `${calculatedHeight}px`;
        wrapper.style.minHeight = `${calculatedHeight}px`;
    }
    canvas.style.height = `${calculatedHeight}px`;
    
    // 3. INITIALIZE CHART.JS
    console.log("Begin initialized the Chart.js");
    try {
        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Importance',
                    data: data.values,
                    borderWidth: 1,
                    borderRadius: 4,
                    barThickness: barHeight
                }]
            },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false, // Wajib false agar mengikuti ukuran wrapper script
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: true }
            },
            layout: {
                padding: { left: 10, right: 25, top: 5, bottom: 5 }
            },
            scales: {
            x: {
                beginAtZero: true,
                ticks: { 
                    color: '#cbd5e1',
                    font: { size: 14 } 
                },
                grid: { color: 'rgba(255,255,255,0.06)' }
            },
            y: {
                ticks: {
                    color: '#cbd5e1',
                    font: { size: 14 },
                    autoSkip: false // Pastikan label tidak ada yang disembunyikan
                },
                grid: { display: false }
                }}}
        });
        console.log(`[Render SUCCESS] Chart berhasil digambar di UI.`);
    } catch (err) {
        console.error(`[Render Error] Chart.js gagal merender:`, err);
        handleChartError(canvas, "Gagal merender chart.");
    }
}

// ==========================================
// 5. ERROR HANDLING UI
// ==========================================
function handleChartError(canvas, message) {
    console.log(`[Fallback] Menampilkan fallback error text ke layar: "${message}"`);
    if (canvas && canvas.parentElement) {
        canvas.parentElement.innerHTML = `<p style="color:red; font-weight:bold;">${message}</p>`;
    }
}