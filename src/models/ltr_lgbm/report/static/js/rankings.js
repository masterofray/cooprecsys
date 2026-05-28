/* =========================================================
   rankings.js
   Clean ML-style ranking table
   ========================================================= */

function initHoverEffects(table) {
    const rows = table.querySelectorAll("tbody tr");
    rows.forEach(row => {
        row.addEventListener("mouseenter", () => {
        row.classList.add("row-hover");
        });
        row.addEventListener("mouseleave", () => {
        row.classList.remove("row-hover");
        });
    });
    }


/* =========================================================
   Animate relevance bars
   ========================================================= */
function initRelevanceBars(container) {
    const bars = container.querySelectorAll(".relevance-bar");
    bars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = "0px";
        requestAnimationFrame(() => {setTimeout(() => {
        bar.style.width = targetWidth;}, 100); });
        });
    }


/* =========================================================
   Advanced Filtering Engine
   ========================================================= */
function initTableFiltering(table) {
    console.log("[Rankings] Initializing filtering...");
    const filters = table.querySelectorAll(".table-filter");
    const rows    = table.querySelectorAll("tbody tr");
    filters.forEach(filter => {
        filter.addEventListener("input", () => {
        rows.forEach(row => {
        let visible = true;
        filters.forEach(activeFilter => {
        const rawFilter = activeFilter.value.trim();
        if (!rawFilter) return;
        const columnName= activeFilter.dataset.column;
        const cell      = row.querySelector(`td[data-column="${columnName}"]`);
        if (!cell) {visible = false;
                    return;}
        const cellText  = cell.textContent.trim();
        const cellValue = parseFloat(cellText.replace(/,/g, ""));
        const isNumeric = !isNaN(cellValue);

        /* NUMERIC FILTERING  */
        if (isNumeric) {
            const passed = evaluateNumericExpression(rawFilter, cellValue);
            if (!passed) {visible = false;}
            }

        /* TEXT FILTERING */
        else {
            if (!cellText.toLowerCase().includes(rawFilter.toLowerCase())
            ) {visible = false;} }
        });

    row.style.display = visible ? "" : "none";
    }); }); });
    console.log("[Rankings] Filtering ready.");
    }


/* =========================================================
   Evaluate Numeric Expression
   ========================================================= */
function evaluateNumericExpression(expr, x) {
    expr = expr.replace(/\s+/g, " ").trim().toLowerCase();
    /* =====================================
       RANGE PATTERN
       Example:
       1 < x <= 5
       ===================================== */
    const rangeMatch =expr.match(
    /^(-?\d+(\.\d+)?)\s*(<=|<)\s*x\s*(<=|<)\s*(-?\d+(\.\d+)?)$/
    );
    if (rangeMatch) {
        const leftValue     = parseFloat(rangeMatch[1]);
        const leftOperator  = rangeMatch[3];
        const rightOperator = rangeMatch[4];
        const rightValue    = parseFloat(rangeMatch[5]);
        let leftPass        = leftOperator === "<=" ? leftValue <= x : leftValue < x;
        let rightPass       = rightOperator === "<=" ? x <= rightValue : x < rightValue;
        return leftPass && rightPass;
        }

    /* AND / OR EXPRESSIONS */
    if (expr.includes(" and ")) {
        const conditions = expr.split(/\sand\s/);
        return conditions.every(cond => evaluateSimpleCondition(cond, x));
        }
    if (expr.includes(" or ")) {
        const conditions = expr.split(/\sor\s/);
        return conditions.some(cond => evaluateSimpleCondition(cond, x));
        }

    /* SIMPLE CONDITION */
    return evaluateSimpleCondition(expr, x);
    }


/* =========================================================
   Simple Condition
   ========================================================= */
function evaluateSimpleCondition(cond, x) {
    cond               = cond.trim().toLowerCase();
    cond               = cond.replace(/^x\s*/, "");
    const match        = cond.match(/^(<=|>=|!=|=|<|>)(.*)$/);
    if (!match) {return String(x).includes(cond);}
    const operator     = match[1];
    const compareValue = parseFloat(match[2]);
    if (isNaN(compareValue)) {return false;}
    switch (operator) {
        case ">":
            return x > compareValue;
        case ">=":
            return x >= compareValue;
        case "<":
            return x < compareValue;
        case "<=":
            return x <= compareValue;
        case "=":
            return x === compareValue;
        case "!=":
            return x !== compareValue;
        default:
            return false;
        }
    }

/* =========================================================
   Main Init
   ========================================================= */
function initRankingsPage(container) {
    if (!container) {
        console.warn("[Rankings] Container tidak ditemukan.");
        return; }
    console.log("[Rankings] Initializing rankings page.");
    const table = container.querySelector(".rankings-table");
    if (!table) {
        console.warn("[Rankings] Rankings table tidak ditemukan.");
        return; }
    initHoverEffects(table);
    initRelevanceBars(container);
    initTableFiltering(table);
    console.log("[Rankings] Initialization complete.");
    }


/* =========================================================
   Export
   ========================================================= */
export {initRankingsPage};


/* =========================================================
   Shortcut
   ========================================================= */
document.addEventListener("beforeinput", (event) => {
    const container = document.getElementById("page-rankings");
    if (!container) return;
    if (event.inputType === "insertText" && event.data === "f" && event.ctrlKey) {
        event.preventDefault();}
    }, true);


document.addEventListener("keydown", (event) => {
    const container = document.getElementById("page-rankings");
    if (!container) return;
    const isInside = container.contains(document.activeElement);

    /* =========================================================
       ESC -> clear filter + reset sort + scroll top
       ========================================================= */
    if (event.key === "Escape") {
        // clear filters
        container.querySelectorAll(".table-filter").forEach(input => {
            input.value = "";
            input.dispatchEvent(new Event("input", { bubbles: true }));
        });

        // reset sort (kalau kamu pakai sort state di JS)
        if (typeof window.resetRankingsSort === "function") {
            window.resetRankingsSort(container); }

        // scroll ke atas table
        const scrollBox = container.querySelector(".table-scroll");
        if (scrollBox) {scrollBox.scrollTo({
            top: 0,
            left: 0,
            behavior: "smooth"});
        }
        container.classList.add("reset-flash");
        setTimeout(() => container.classList.remove("reset-flash"), 300);
        console.log("[Rankings] ESC → reset filters + sort + scroll top");
        }

    /* =========================================================
       Ctrl + F -> focus first filter input
       (override default browser search only inside table)
       ========================================================= */
     if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "f") {
        if (!isInside) return;
        event.preventDefault();
        event.stopPropagation();
        const firstFilter = container.querySelector(".table-filter");
        if (firstFilter) {firstFilter.focus();
            firstFilter.select?.(); }
        console.log("[Rankings] Ctrl+F intercepted");
        }
    }, true);
