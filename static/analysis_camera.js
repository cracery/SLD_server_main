/* Const needed for results */
const emotionColors = {
    angry: "#dc3545", disgust: "#6f42c1", fear: "#9c27b0", happy: "#ffc107",
    sad: "#0d6efd", surprise: "#fd7e14", neutral: "#6c757d", contempt: "#343a40"
};
const emotionNames = {
    angry: "Angry", disgust: "Disgust", fear: "Fear", happy: "Happy",
    sad: "Sad", surprise: "Surprise", neutral: "Neutral", contempt: "Contempt"
};

/* Analyse button */
document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyze-btn");
    const preview = document.getElementById("image-preview");

    if (!analyzeBtn || !preview) {
        console.warn("Analyze button or image preview not found");
        return;
    }

    analyzeBtn.addEventListener("click", () => {
        if (!preview.fileBlob) {
            showError("No captured photo found.");
            return;
        }
        analyzeCapturedImage(preview.fileBlob);
    });
});

async function analyzeCapturedImage(blob) {
    showLoading();
    hideError();
    resetResults();

    const formData = new FormData();
    formData.append("file", blob, "captured.jpg");

    try {
        const res = await fetch("https://stress-detection-api-production.up.railway.app/predict/image", {
            method: "POST",
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        displayResults(data);
    } catch (err) {
        showError("An error occurred during camera image analysis. Make sure your face is visible and try again.");
        console.error(err);
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    if (data.status !== "success" || !data.result) {
        showError("The analysis results could not be obtained.");
        return;
    }

    const { stress_probabilities: p, predicted_stress: lvl, emotions } = data.result;
    const low = p.Low * 100, mid = p.Middle * 100, high = p.High * 100;
    const value = low * 16.5 + mid * 49.5 + high * 83;
    gauge.set(value / 100);

    let lbl, cls;
    switch (lvl) {
        case "Low": lbl = "Low stress level"; cls = "text-success"; break;
        case "Middle": lbl = "Middle stress level"; cls = "text-warning"; break;
        case "High": lbl = "High stress level"; cls = "text-danger"; break;
        default: lbl = "Unknown level"; cls = "text-secondary";
    }
    stressLevelLabel.innerHTML = lbl;
    stressLevelLabel.className = `gauge-label ${cls}`;
    updateBar("low", low); updateBar("middle", mid); updateBar("high", high);
    updateEmotionCharts(emotions); showResults();
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

function updateBar(which, val) {
    document.getElementById(`${which}-stress-bar`).style.width = `${val.toFixed(1)}%`;
    document.getElementById(`${which}-stress-value`).textContent = `${val.toFixed(1)}%`;
}