// ── LTR Dashboard JS ──

document.addEventListener('DOMContentLoaded', () => {
    // Restore last active page
    const saved = localStorage.getItem('ltr_page') || 'overview';
    switchPage(saved);
});

function switchPage(page) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.page-tab').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.sidebar-link').forEach(el => el.classList.remove('active'));

    // Show selected
    const target = document.getElementById('page-' + page);
    if (target) target.classList.add('active');

    // Activate tab
    document.querySelectorAll('.page-tab').forEach(el => {
        if (el.textContent.trim().toLowerCase().includes(page)) el.classList.add('active');
    });

    // Activate sidebar
    document.querySelectorAll('.sidebar-link').forEach(el => {
        if (el.dataset.page === page) el.classList.add('active');
    });

    localStorage.setItem('ltr_page', page);
}
