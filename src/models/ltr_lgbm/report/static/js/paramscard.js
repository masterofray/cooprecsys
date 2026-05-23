document.addEventListener('DOMContentLoaded', () => {
    // Mengambil semua elemen score card
    console.log("Mulai DOM best parameter score card")
    const scoreCards = document.querySelectorAll('.score-card');
    scoreCards.forEach(card => {
        card.addEventListener('click', () => {
            const paramName  = card.getAttribute('data-parameter');
            const paramValue = card.getAttribute('data-value');
            console.log(`Parameter Terpilih: ${paramName} = ${paramValue}`);
            
            // Fitur opsional: Otomatis salin nilai ke clipboard saat card diklik
            navigator.clipboard.writeText(paramValue).then(() => {
                console.log(`Nilai untuk ${paramName} berhasil disalin!`);
            }).catch(err => {
                console.error('Gagal menyalin teks: ', err);
            });
        });
    });
});