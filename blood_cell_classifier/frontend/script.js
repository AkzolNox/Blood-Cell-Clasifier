// ============================================================
// Hematoscope — lógica de interfaz
// Conecta con el backend Flask (api/app.py) en API_URL
// ============================================================

const API_URL = "http://localhost:5000";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const liveVideo = document.getElementById("liveVideo");
const placeholder = document.getElementById("placeholder");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnModeFile = document.getElementById("btnModeFile");
const btnModeCamera = document.getElementById("btnModeCamera");
const cameraControls = document.getElementById("cameraControls");
const deviceSelect = document.getElementById("deviceSelect");
const btnStartCamera = document.getElementById("btnStartCamera");
const btnCaptureFrame = document.getElementById("btnCaptureFrame");
const captureCanvas = document.getElementById("captureCanvas");
const autoAnalyzeRow = document.getElementById("autoAnalyzeRow");
const autoAnalyzeCheckbox = document.getElementById("autoAnalyzeCheckbox");
const liveDot = document.getElementById("liveDot");

const stage1Card = document.getElementById("stage1Card");
const stage2Card = document.getElementById("stage2Card");
const stage1Result = document.getElementById("stage1Result");
const stage2Result = document.getElementById("stage2Result");
const stage1Bars = document.getElementById("stage1Bars");
const stage2Bars = document.getElementById("stage2Bars");
const cascadeArrow = document.getElementById("cascadeArrow");
const descriptionBox = document.getElementById("descriptionBox");
const descriptionText = document.getElementById("descriptionText");
const errorBox = document.getElementById("errorBox");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
document.getElementById("apiUrlLabel").textContent = API_URL;

let selectedFile = null;
let mediaStream = null;
let autoAnalyzeInterval = null;
let analysisInFlight = false;

// -------------------- Alternar entre modo archivo y modo cámara --------------------
function stopAutoAnalyze() {
  if (autoAnalyzeInterval) {
    clearInterval(autoAnalyzeInterval);
    autoAnalyzeInterval = null;
  }
  liveDot.hidden = true;
  autoAnalyzeCheckbox.checked = false;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  liveVideo.hidden = true;
  liveVideo.srcObject = null;
  btnCaptureFrame.disabled = true;
  stopAutoAnalyze();
}

btnModeFile.addEventListener("click", () => {
  btnModeFile.classList.add("active");
  btnModeCamera.classList.remove("active");
  cameraControls.hidden = true;
  autoAnalyzeRow.hidden = true;
  stopCamera();
  if (!selectedFile) placeholder.hidden = false;
  preview.hidden = !selectedFile;
});

btnModeCamera.addEventListener("click", async () => {
  btnModeCamera.classList.add("active");
  btnModeFile.classList.remove("active");
  cameraControls.hidden = false;
  autoAnalyzeRow.hidden = false;
  placeholder.hidden = true;
  preview.hidden = true;

  // Lista los dispositivos de video disponibles (incluye la cámara USB del microscopio,
  // que el sistema operativo expone como una webcam UVC estándar).
  try {
    // Pide permiso una vez para que enumerateDevices devuelva labels legibles.
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
    tmp.getTracks().forEach((t) => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === "videoinput");
    deviceSelect.innerHTML = "";
    videoInputs.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Cámara ${i + 1}`;
      deviceSelect.appendChild(opt);
    });
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = `⚠ No se pudo acceder a las cámaras: ${err.message}`;
  }
});

btnStartCamera.addEventListener("click", async () => {
  stopCamera();
  try {
    const constraints = {
      video: deviceSelect.value ? { deviceId: { exact: deviceSelect.value } } : true,
    };
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    liveVideo.srcObject = mediaStream;
    liveVideo.hidden = false;
    btnCaptureFrame.disabled = false;
    errorBox.hidden = true;
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = `⚠ No se pudo iniciar la cámara: ${err.message}`;
  }
});

function captureFrameAsFile() {
  return new Promise((resolve) => {
    const w = liveVideo.videoWidth;
    const h = liveVideo.videoHeight;
    if (!w || !h) { resolve(null); return; }

    captureCanvas.width = w;
    captureCanvas.height = h;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(liveVideo, 0, 0, w, h);

    captureCanvas.toBlob((blob) => {
      resolve(new File([blob], "captura_microscopio.png", { type: "image/png" }));
    }, "image/png");
  });
}

btnCaptureFrame.addEventListener("click", async () => {
  const file = await captureFrameAsFile();
  if (!file) return;
  selectedFile = file;
  analyzeBtn.disabled = false;
  resetResults();
});

// -------------------- Análisis automático en vivo --------------------
autoAnalyzeCheckbox.addEventListener("change", () => {
  if (autoAnalyzeCheckbox.checked) {
    if (!mediaStream) {
      autoAnalyzeCheckbox.checked = false;
      errorBox.hidden = false;
      errorBox.textContent = "⚠ Iniciá la cámara antes de activar el análisis automático.";
      return;
    }
    liveDot.hidden = false;
    autoAnalyzeInterval = setInterval(async () => {
      if (analysisInFlight) return; // evita apilar pedidos si el anterior no terminó
      const file = await captureFrameAsFile();
      if (file) await analyzeImage(file, { silent: true });
    }, 2500);
  } else {
    stopAutoAnalyze();
  }
});

// -------------------- Chequeo de estado del backend --------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`, { method: "GET" });
    if (res.ok) {
      statusDot.className = "dot online";
      statusText.textContent = "backend conectado";
    } else {
      throw new Error("no ok");
    }
  } catch {
    statusDot.className = "dot offline";
    statusText.textContent = "backend no disponible — ejecuta api/app.py";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// -------------------- Selección de imagen --------------------
dropZone.addEventListener("click", () => {
  if (btnModeCamera.classList.contains("active")) return;
  fileInput.click();
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    placeholder.hidden = true;
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  resetResults();
}

// -------------------- Análisis --------------------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  await analyzeImage(selectedFile, { silent: false });
});

async function analyzeImage(file, { silent = false } = {}) {
  analysisInFlight = true;
  errorBox.hidden = true;

  if (!silent) {
    resetResults();
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analizando…";
  }
  stage1Card.classList.add("active");
  cascadeArrow.classList.add("active");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Error desconocido del servidor");
    }

    renderResults(data);
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = `⚠ ${err.message}`;
  } finally {
    if (!silent) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analizar muestra";
    }
    stage1Card.classList.remove("active");
    cascadeArrow.classList.remove("active");
    analysisInFlight = false;
  }
}

function resetResults() {
  stage1Result.innerHTML = '<span class="result-placeholder">esperando muestra…</span>';
  stage2Result.innerHTML = '<span class="result-placeholder">depende de la etapa 1</span>';
  stage1Bars.innerHTML = "";
  stage2Bars.innerHTML = "";
  descriptionBox.hidden = true;
  stage2Card.classList.remove("active");
}

function renderBars(container, probsObj) {
  container.innerHTML = "";
  const entries = Object.entries(probsObj).sort((a, b) => b[1] - a[1]);
  const maxVal = entries.length ? entries[0][1] : 1;

  entries.forEach(([label, value], i) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const labelEl = document.createElement("span");
    labelEl.className = "bar-label";
    labelEl.textContent = label;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill" + (i === 0 ? " top" : "");
    fill.style.width = "0%";
    track.appendChild(fill);

    const valueEl = document.createElement("span");
    valueEl.className = "bar-value";
    valueEl.textContent = `${Math.round(value * 100)}%`;

    row.appendChild(labelEl);
    row.appendChild(track);
    row.appendChild(valueEl);
    container.appendChild(row);

    requestAnimationFrame(() => {
      fill.style.width = `${(value / maxVal) * 100}%`;
    });
  });
}

function renderResults(data) {
  stage1Result.innerHTML =
    `${data.coarse_class} <span class="result-confidence">${Math.round(data.coarse_confidence * 100)}% confianza</span>`;
  renderBars(stage1Bars, data.coarse_probabilities);

  if (data.fine_class) {
    stage2Card.classList.add("active");
    stage2Result.innerHTML =
      `${data.fine_class.replace(/_/g, " ")} <span class="result-confidence">${Math.round(data.fine_confidence * 100)}% confianza</span>`;
    renderBars(stage2Bars, data.fine_probabilities);
    descriptionBox.hidden = false;
    descriptionText.textContent = data.fine_description || data.coarse_description;
  } else {
    stage2Result.innerHTML = '<span class="result-placeholder">sin sub-clasificador para esta clase</span>';
    descriptionBox.hidden = false;
    descriptionText.textContent = data.coarse_description;
  }

  if (data.uncertain) {
    errorBox.hidden = false;
    errorBox.textContent = "⚠ Confianza baja en la Etapa 1 — considera repetir con una imagen más nítida.";
  }
}
