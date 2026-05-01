document.addEventListener("DOMContentLoaded", () => {

    // theme persistence
    const theme = localStorage.getItem("ltr_theme");
    if (theme === "light") {
        document.body.classList.add("light");
    }

    // active nav on scroll
    const links = document.querySelectorAll(".nav-links a");
    const sections = document.querySelectorAll("section[id]");

    function activateNav() {
        let current = "";

        sections.forEach(sec => {
            const top = sec.offsetTop - 140;
            if (window.scrollY >= top) {
                current = sec.id;
            }
        });

        links.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === "#" + current) {
                link.classList.add("active");
            }
        });
    }

    activateNav();
    window.addEventListener("scroll", activateNav);
});


function toggleTheme() {
    document.body.classList.toggle("light");

    const mode = document.body.classList.contains("light")
        ? "light"
        : "dark";

    localStorage.setItem("ltr_theme", mode);
}


function jumpTop() {
    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}