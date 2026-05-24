/*
* author     = "Aryanto"
* copyright  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
* license    = "GNUPublic"
* version    = "0.0.1"
* email      = "aryanto.dandan@gmail.com"
* status     = "Development"
* created    = "2026-05-23"
* About
  - heatmaps.js
  - Manajemen visualisasi Heatmap Plotly dengan verbose debugging.
 */


function initializeAllHeatmaps() {
  console.group("=== [Heatmap Initialization Engine] ===");
  console.log("INFO: Memulai pemindaian elemen heatmap pada DOM...");

  // 1. Validasi keberadaan Library Plotly secara global
  if (typeof Plotly === "undefined") {
    console.error("FATAL ERROR: Library Plotly tidak terdeteksi pada global scope (window.Plotly). Proses rendering dihentikan.");
    
    // Berikan fallback visual ke semua target elemen jika plotly absen
    document.querySelectorAll(".js-heatmap-render").forEach(el => {
      el.innerHTML = '<p style="color:red; font-weight:bold;">Plotly library belum dimuat</p>';
    });
    console.groupEnd();
    return;
  }
  console.log("SUCCESS: Plotly library terdeteksi aman.");

  // 2. Ambil semua elemen chart yang membutuhkan rendering
  const heatmapElements = document.querySelectorAll(".js-heatmap-render");
  console.log(`INFO: Ditemukan ${heatmapElements.length} elemen dengan class '.js-heatmap-render'.`);

  // 3. Iterasi setiap elemen untuk ekstraksi data dan rendering
  heatmapElements.forEach((chartEl, index) => {
    const divId = chartEl.id;
    const chartTitle = chartEl.getAttribute("data-title") || `Unnamed Chart ${index}`;
    
    console.group(`-> Pemrosesan Chart [${index}] | ID: ${divId} | Title: "${chartTitle}"`);
    
    try {
      // Ekstraksi data mentah dari HTML attributes
      console.log(`[${divId}] Mengekstrak data kustom dari HTML attributes...`);
      const rawZ = chartEl.getAttribute("data-z");
      const rawX = chartEl.getAttribute("data-x");
      const rawY = chartEl.getAttribute("data-y");

      if (!rawZ || !rawX || !rawY) {
        throw new Error("Atribut data-z, data-x, atau data-y tidak ditemukan pada elemen HTML.");
      }

      // Parsing string JSON menjadi objek/array JavaScript
      const zData = JSON.parse(rawZ);
      const xData = JSON.parse(rawX);
      const yData = JSON.parse(rawY);

      // Logging spesifikasi dimensi matriks korelasi untuk debugging data science
      console.log(`[${divId}] Hasil Parsing JSON Berhasil:`, {
        "Dimensi Z (Rows)": zData.length,
        "Dimensi Z (Cols/Row 0)": zData[0] ? zData[0].length : 0,
        "Jumlah Label X": xData.length,
        "Jumlah Label Y": yData.length
      });

      // Validasi integritas struktur matriks Z
      if (!Array.isArray(zData) || zData.length === 0) {
        console.warn(`[${divId}] VALIDATION WARNING: Matriks zData kosong atau bukan sebuah array.`);
        chartEl.innerHTML = '<p style="color:red;">Correlation matrix kosong</p>';
        console.groupEnd();
        return;
      }

      // 4. Deteksi Responsivitas Gadget / Viewport
      const isMobile = window.innerWidth < 768;
      console.log(`[${divId}] Deteksi Viewport:`, { windowWidth: window.innerWidth, isMobile: isMobile });

      // Fungsi internal pemotong label panjang
      function shortenLabel(label, maxLength) {
        if (!label) return "";
        return label.length > maxLength ? label.slice(0, maxLength) + "..." : label;
      }

      const maxLabelLen = isMobile ? 10 : 20;
      const xLabels = xData.map(v => shortenLabel(v, maxLabelLen));
      const yLabels = yData.map(v => shortenLabel(v, maxLabelLen));
      
      const featureCount = xLabels.length;
      const cellSize = isMobile ? 18 : 28;
      const heatmapSize = Math.max(500, featureCount * cellSize);

      console.log(`[${divId}] Konfigurasi Dimensi Layout:`, {
        featureCount: featureCount,
        cellSizePx: cellSize,
        computedHeightPx: heatmapSize
      });

      // 5. Konstruksi Payload Parameter Plotly
      const trace = {
        type: "heatmap",
        z: zData,
        x: xLabels,
        y: yLabels,
        colorscale: "RdBu",
        zmid: 0,
        hoverongaps: false,
        xgap: 1,
        ygap: 1,
        hovertemplate: "<b>%{x}</b><br><b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>",
        colorbar: {
          thickness: isMobile ? 12 : 18,
          outlinewidth: 0,
          tickfont: {
            color: "#dbeafe",
            size: isMobile ? 10 : 12
          }
        }
      };

      const layout = {
        autosize: true,
        height: heatmapSize,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
          family: "Inter, sans-serif",
          color: "#dbeafe",
          size: isMobile ? 10 : 12
        },
        margin: {
          l: isMobile ? 100 : 180,
          r: 40,
          t: 80,
          b: isMobile ? 120 : 180
        },
        xaxis: {
          tickangle: -45,
          automargin: true,
          tickfont: {
            size: isMobile ? 9 : 11,
            color: "#cbd5e1"
          },
          gridcolor: "rgba(255,255,255,0.06)"
        },
        yaxis: {
          automargin: true,
          tickfont: {
            size: isMobile ? 9 : 11,
            color: "#cbd5e1"
          },
          gridcolor: "rgba(255,255,255,0.06)"
        }
      };

      const config = {
        responsive: true,
        displayModeBar: false,
        scrollZoom: false,
        displaylogo: false
      };

      // 6. Eksekusi Rendering Plotly Baru
      console.log(`[${divId}] Memanggil Plotly.newPlot()...`);
      Plotly.newPlot(divId, [trace], layout, config);
      console.log(`[${divId}] SUCCESS: Grafik Berhasil Dirender.`);

    } catch (err) {
      console.error(`[${divId}] RUNTIME ERROR pada proses render:`, err);
      chartEl.innerHTML = `<p style="color:red;">Gagal render chart: ${err.message}</p>`;
    }
    
    console.groupEnd();
  });

  console.log("INFO: Semua antrean pemrosesan elemen heatmap selesai dieksekusi.");
  console.groupEnd();
}

// Menjalankan fungsi dengan mekanisme penundaan (setTimeout) pasca layout window selesai dimuat sempurna
// IMPLEMENTASI PRODUCTION-GRADE
window.addEventListener("DOMContentLoaded", function () {
  console.log("EVENT: DOMContentLoaded. Menginisialisasi rendering dan ResizeObserver.");
  
  // 1. Jalankan inisialisasi awal
  initializeAllHeatmaps();

  // 2. Pasang ResizeObserver untuk stabilitas dimensi responsif
  const resizeObserver = new ResizeObserver(entries => {
    // Gunakan requestAnimationFrame untuk mencegah ResizeObserver loop error
    window.requestAnimationFrame(() => {
      for (let entry of entries) {
        if (entry.target.id) {
          // Memaksa Plotly merekonstruksi ukuran saat kontainer grid/flex benar-benar berubah
          Plotly.Plots.resize(entry.target.id);
        }}
    });
  });

  // Observasi semua elemen heatmap
  document.querySelectorAll(".js-heatmap-render").forEach(el => {
    resizeObserver.observe(el);
  });
});