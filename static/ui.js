/* Capture DOM */
var dropArea= document.getElementById("drop-area");
var fileInput= document.getElementById("file-input");
var imagePrev = document.getElementById("image-preview");
var analyzeBtn= document.getElementById("analyze-btn");
var loading =document.getElementById("loading");
var errBox  =document.getElementById("error-message");
var errText = document.getElementById("error-text");
var results = document.getElementById("results-container");
var stressLevelLabel = document.getElementById("stress-level-label");
/* Gauge */
var gauge = new Gauge(document.getElementById("speedometer")).setOptions({
    angle:0.15,
    lineWidth:0.44,
    radiusScale: 1,
    pointer: {length: 0.6, strokeWidth: 0.035, color: "#000" },
    colorStart:"#6FADCF",
    colorStop: "#8FC0DA",
    strokeColor:"#E0E0E0",
    generateGradient:true,
    highDpiSupport:true,
    staticLabels:{font:"10px sans-serif", labels:[0, 33, 66, 100]},
    staticZones:[
        {strokeStyle: "#28a745", min: 0,  max: 33 },
        {strokeStyle: "#ffc107", min: 33, max: 66 },
        {strokeStyle: "#dc3545", min: 66, max: 100 }
    ]
});
gauge.maxValue=100;
gauge.setMinValue(0);
gauge.set(0);
/* Drag & Drop*/
["dragenter","dragover","dragleave","drop"].forEach(ev =>
    dropArea.addEventListener(ev,function(e) { e.preventDefault(); e.stopPropagation(); })
);
["dragenter","dragover"].forEach(ev=>
    dropArea.addEventListener(ev,function() {dropArea.classList.add("highlight") })
);
["dragleave","drop"].forEach(ev=>
    dropArea.addEventListener(ev,function() { dropArea.classList.remove("highlight") })
);
dropArea.addEventListener("drop",function(e) {
    var f= e.dataTransfer.files;
    if (f.length> 0) handleFile(f[0]);
});
dropArea.addEventListener("click",function() {fileInput.click(); });
fileInput.addEventListener("change",function() {
    if (fileInput.files.length> 0) handleFile(fileInput.files[0]);
});
/* Process image */
function handleFile(file) {
    if (!file.type.match("image.*")) {showError("Please choose image."); return; }
    resetResults();
    var reader= new FileReader();
    reader.onload = function(e) {
        imagePrev.src= e.target.result;
        imagePrev.style.display= "block";
        analyzeBtn.disabled= false;
    };
    reader.readAsDataURL(file);
}
/* Analyse button */
analyzeBtn.addEventListener("click", function() {
    if (fileInput.files.length=== 0) { showError("First, select an image."); return; }
    analyzeImage(fileInput.files[0]);
});
/* Additional */
function resetResults() {
    results.classList.remove("results-visible");
    gauge.set(0);
    stressLevelLabel.textContent= "Analyse…";
    stressLevelLabel.className= "gauge-label text-secondary";
    ["low","middle","high"].forEach(x=> updateBar(x, 0));
    document.getElementById("emotions-container").innerHTML ="";
}
function showResults() {results.classList.add("results-visible");}
function showLoading(){loading.classList.remove("d-none"); analyzeBtn.disabled = true;}
function hideLoading(){loading.classList.add("d-none");   analyzeBtn.disabled = false;}
function showError(msg){errText.textContent = msg; errBox.classList.remove("d-none");}
function hideError(){errBox.classList.add("d-none");}
