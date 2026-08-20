/**
 * Sistem navigasi SPA berbasis pergantian kelas active pada DOM
 */
function switchPage(pageId) {
    console.log(`[Router] Berpindah ke modul halaman: ${pageId}`);
    
    // 1. Reset visibilitas container kontens
    document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active'));
    // 2. Reset status tombol navigasi sidebar
    document.querySelectorAll('.sidebar-link').forEach(b => b.classList.remove('active'));
    
    // 3. Aktifkan komponen target
    const targetPage = document.getElementById(pageId);
    const targetBtn = document.querySelector(`[data-page="${pageId}"]`);
    
    if (targetPage) targetPage.classList.add('active');
    if (targetBtn) targetBtn.classList.add('active');
    
    // Trigger refresh canvas jika dibutuhkan oleh Plotly/Chartjs
    window.dispatchEvent(new Event('resize'));
}