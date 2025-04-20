/* ---------- Константи, потрібні лише для обробки результатів ---------- */
const emotionColors = {
    angry: "#dc3545",
    disgust: "#6f42c1",
    fear: "#9c27b0",
    happy: "#ffc107",
    sad: "#0d6efd",
    surprise: "#fd7e14",
    neutral: "#6c757d",
    contempt: "#343a40"
};
const emotionNames = {
    angry: "Злість",
    disgust: "Відраза",
    fear: "Страх",
    happy: "Радість",
    sad: "Смуток",
    surprise: "Здивування",
    neutral: "Нейтральність",
    contempt: "Презирство"
};

/* ---------- API‑запит і візуалізація ---------- */
async function analyzeImage(file) {
    showLoading();
    hideError();
    resetResults();

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(
            "https://stress-detection-api-production.up.railway.app/predict/image",
            { method: "POST", body: formData }
        );

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        displayResults(data);
    } catch (err) {
        showError("Помилка при аналізі зображення. Переконайтеся, що на зображенні є обличчя і спробуйте ще раз.");
        console.error(err);
    } finally {
        hideLoading();
    }
}

/* ---------- Опрацювання та показ результатів ---------- */
function displayResults(data) {
    if (data.status !== "success" || !data.result) {
        showError("Не вдалося отримати результати аналізу.");
        return;
    }

    const { stress_probabilities: p, predicted_stress: lvl, emotions } = data.result;
    const low = p.Low * 100, mid = p.Middle * 100, high = p.High * 100;

    /* Спідометр */
    const value = low * 16.5 + mid * 49.5 + high * 83;
    gauge.set(value / 100);

    let lbl, cls;
    switch (lvl) {
        case "Low":    lbl = "Низький рівень стресу"; cls = "text-success"; break;
        case "Middle": lbl = "Середній рівень стресу"; cls = "text-warning"; break;
        case "High":   lbl = "Високий рівень стресу";  cls = "text-danger";  break;
        default:       lbl = "Невизначений рівень";    cls = "text-secondary";
    }
    stressLevelLabel.innerHTML = lbl;
    stressLevelLabel.className = `gauge-label ${cls}`;

    /* Прогрес‑бари */
    updateBar("low", low);
    updateBar("middle", mid);
    updateBar("high", high);

    /* Емоції */
    updateEmotionCharts(emotions);
    showResults();
}

function updateEmotionCharts(raw) {
    const total = Object.values(raw).reduce((s, v) => s + v, 0) || 1;
    const list = Object.entries(raw)
        .map(([k, v]) => [k, (v / total * 100).toFixed(1)])
        .sort((a, b) => b[1] - a[1]);

    const box = document.getElementById("emotions-container");
    box.innerHTML = "";
    list.forEach(([k, pct]) => {
        const col = emotionColors[k] || "#6c757d";
        const ua  = emotionNames[k] || k;
        box.insertAdjacentHTML("beforeend", `
            <div class="mb-3">
                <div class="emotion-label">
                    <span><i class="fas fa-face-meh me-2" style="color:${col}"></i>${ua}</span>
                    <span class="badge bg-light text-dark">${pct}%</span>
                </div>
                <div class="progress" style="height:12px">
                    <div class="progress-bar" style="width:${pct}%;background:${col}"></div>
                </div>
            </div>
        `);
    });
}

/* ---------- Допоміжне ---------- */
function updateBar(which, val) {
    document.getElementById(`${which}-stress-bar`).style.width  = `${val.toFixed(1)}%`;
    document.getElementById(`${which}-stress-value`).textContent = `${val.toFixed(1)}%`;
}
