/* Capture DOM */
const dropArea  = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const imagePrev = document.getElementById("image-preview");
const analyzeBtn= document.getElementById("analyze-btn");
const loading   = document.getElementById("loading");
const errBox    = document.getElementById("error-message");
const errText   = document.getElementById("error-text");
const results   = document.getElementById("results-container");
const stressLevelLabel = document.getElementById("stress-level-label");

/* Gauge */
const gauge = new Gauge(document.getElementById("speedometer")).setOptions({
    angle: 0.15,
    lineWidth: 0.44,
    radiusScale: 1,
    pointer: { length: 0.6, strokeWidth: 0.035, color: "#000" },
    colorStart: "#6FADCF",
    colorStop: "#8FC0DA",
    strokeColor: "#E0E0E0",
    generateGradient: true,
    highDpiSupport: true,
    staticLabels: { font: "10px sans-serif", labels: [0, 33, 66, 100] },
    staticZones: [
        { strokeStyle: "#28a745", min: 0,  max: 33 },
        { strokeStyle: "#ffc107", min: 33, max: 66 },
        { strokeStyle: "#dc3545", min: 66, max: 100 }
    ]
});
gauge.maxValue = 100;
gauge.setMinValue(0);
gauge.set(0);

/* Drag & Drop*/
["dragenter","dragover","dragleave","drop"].forEach(ev =>
    dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
);
["dragenter","dragover"].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.add("highlight"))
);
["dragleave","drop"].forEach(ev =>
    dropArea.addEventListener(ev, () => dropArea.classList.remove("highlight"))
);
dropArea.addEventListener("drop", e => {
    const f = e.dataTransfer.files;
    if (f.length) handleFile(f[0]);
});
dropArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

/* Process image */
function handleFile(file) {
    if (!file.type.match("image.*")) { showError("Будь ласка, виберіть файл‑зображення."); return; }
    resetResults();

    const reader = new FileReader();
    reader.onload = e => {
        imagePrev.src   = e.target.result;
        imagePrev.style.display = "block";
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

/* Analyse button */
analyzeBtn.addEventListener("click", () => {
    if (!fileInput.files.length) { showError("First, select an image."); return; }
    analyzeImage(fileInput.files[0]);
});
/* Additional */
function resetResults() {
    results.classList.remove("results-visible");
    gauge.set(0);
    stressLevelLabel.textContent = "Analyse…";
    stressLevelLabel.className = "gauge-label text-secondary";
    ["low","middle","high"].forEach(x => updateBar(x, 0));
    document.getElementById("emotions-container").innerHTML = "";
}
function showResults() { results.classList.add("results-visible"); }
function showLoading(){ loading.classList.remove("d-none"); analyzeBtn.disabled = true; }
function hideLoading(){ loading.classList.add("d-none");   analyzeBtn.disabled = false; }
function showError(msg){ errText.textContent = msg; errBox.classList.remove("d-none"); }
function hideError(){ errBox.classList.add("d-none"); }
