/**
 * Front-end Application Controller for Multi-Level Frequency Deepfake Detection
 * Coordinates state, API calls, Chart.js visualizations, and forensic certificate generation.
 */

// Global State
let currentSelectedSample = "sample_deepfake_faceswap";
let currentUploadedFile = null;
let currentForensicData = null;

// Chart.js Instances
let dftChart = null;
let benfordChart = null;
let hifeChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSampleChips();
  initUpload();
  initSliders();
  initModals();
  initActions();

  // Load initial sample automatically
  runAnalysisForSample(currentSelectedSample);
});

// 1. TABS SYSTEM
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-target");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

// 2. SAMPLE SELECTION CHIPS
function initSampleChips() {
  const chips = document.querySelectorAll(".sample-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");

      currentSelectedSample = chip.getAttribute("data-sample");
      currentUploadedFile = null;
      document.getElementById("dropZonePrompt").textContent = "Drag & drop facial image here";

      runAnalysisForSample(currentSelectedSample);
    });
  });
}

// 3. FILE UPLOAD HANDLING
function initUpload() {
  const dropZone = document.getElementById("uploadDropZone");
  const fileInput = document.getElementById("fileInput");

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleIncomingFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleIncomingFile(e.target.files[0]);
    }
  });
}

function handleIncomingFile(file) {
  if (!file.type.match(/image\/(jpeg|png|webp)/)) {
    alert("Please select a valid image file (JPEG, PNG, WebP).");
    return;
  }

  currentUploadedFile = file;
  document.getElementById("dropZonePrompt").textContent = `Selected: ${file.name}`;
  
  // Uncheck sample chips
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
  
  // Auto trigger analysis on file select
  runAnalysisForUploadedFile(file);
}

// 4. SLIDERS & CONTROLS
function initSliders() {
  const opacitySlider = document.getElementById("heatmapOpacitySlider");
  const opacityText = document.getElementById("opacityValueText");
  const overlayImg = document.getElementById("imgForgeryOverlay");

  if (opacitySlider && opacityText && overlayImg) {
    opacitySlider.addEventListener("input", (e) => {
      const val = e.target.value;
      opacityText.textContent = `${val}%`;
      overlayImg.style.opacity = val / 100.0;
    });
  }

  const launderingSlider = document.getElementById("launderingSlider");
  const launderingText = document.getElementById("launderingQualityText");
  if (launderingSlider && launderingText) {
    launderingSlider.addEventListener("input", (e) => {
      const q = e.target.value;
      let label = `Q=${q}`;
      if (q <= 25) label += " (Extreme Collapse)";
      else if (q <= 40) label += " (c40 Heavy)";
      else if (q <= 75) label += " (c23 Moderate)";
      else label += " (Raw/Light)";
      launderingText.textContent = label;
    });
  }
}

// 5. MODAL & CERTIFICATE
function initModals() {
  const modal = document.getElementById("daubertCertModal");
  const btnOpen = document.getElementById("btnOpenCertModal");
  const btnOpen2 = document.getElementById("btnTriggerCertModal");
  const btnClose = document.getElementById("btnCloseCertModal");
  const btnPrint = document.getElementById("btnPrintCert");

  const openModal = () => {
    if (!currentForensicData) return;
    populateCertificateData(currentForensicData);
    modal.classList.add("active");
  };

  if (btnOpen) btnOpen.addEventListener("click", openModal);
  if (btnOpen2) btnOpen2.addEventListener("click", openModal);
  if (btnClose) btnClose.addEventListener("click", () => modal.classList.remove("active"));

  if (btnPrint) {
    btnPrint.addEventListener("click", () => {
      window.print();
    });
  }

  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("active");
  });
}

// 6. ACTION BUTTONS
function initActions() {
  const btnAnalyze = document.getElementById("btnAnalyze");
  btnAnalyze.addEventListener("click", () => {
    if (currentUploadedFile) {
      runAnalysisForUploadedFile(currentUploadedFile);
    } else {
      runAnalysisForSample(currentSelectedSample);
    }
  });

  const btnLaundering = document.getElementById("btnTestLaundering");
  if (btnLaundering) {
    btnLaundering.addEventListener("click", runLaunderingSimulation);
  }
}

// ANALYSIS WORKFLOW
async function runAnalysisForSample(sampleId) {
  showLoading(true, "EXTRACTING MULTI-LEVEL WAVELET PACKETS & FOURIER PROFILES...");
  try {
    const res = await fetch("/api/analyze-sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId })
    });
    if (!res.ok) throw new Error("API analysis request failed.");
    const data = await res.json();
    currentForensicData = data;
    renderForensicTelemetry(data);
  } catch (err) {
    console.error("Forensic analysis error:", err);
    alert("Error performing forensic analysis. Check server logs.");
  } finally {
    showLoading(false);
  }
}

async function runAnalysisForUploadedFile(file) {
  showLoading(true, "DETECTING FACE CROP & DECOMPOSING SPATIO-FREQUENCY TENSORS...");
  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("API upload analysis failed.");
    const data = await res.json();
    currentForensicData = data;
    renderForensicTelemetry(data);
  } catch (err) {
    console.error("Forensic upload error:", err);
    alert("Error processing uploaded image. Check server logs.");
  } finally {
    showLoading(false);
  }
}

// RENDER ALL DATA ON DASHBOARD
function renderForensicTelemetry(data) {
  const isFake = data.verdict.includes("MANIPULATED") || data.composite_forgery_score >= 0.50;

  // 1. Verdict Hero Banner
  const hero = document.getElementById("verdictHeroBanner");
  const verdictText = document.getElementById("verdictText");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictIcon = document.getElementById("verdictIcon");
  const verdictSummary = document.getElementById("verdictSummaryText");
  const shaSpan = document.getElementById("sha256HashSpan");
  const gauge = document.getElementById("gaugeConfidence");

  if (isFake) {
    hero.className = "verdict-hero verdict-fake";
    verdictIcon.textContent = "🚨";
    verdictText.textContent = "MANIPULATED (DEEPFAKE)";
    verdictBadge.textContent = "CRITICAL MANIPULATION";
    verdictBadge.className = "verdict-badge cat-fake";
    gauge.style.color = "var(--accent-crimson)";
    gauge.textContent = `${data.confidence_percentage}%`;
    verdictSummary.textContent = "Elevated high-frequency Dirac comb checkerboard spikes, wavelet variance, and Benford AC distortion detected.";
  } else {
    hero.className = "verdict-hero verdict-real";
    verdictIcon.textContent = "🛡️";
    verdictText.textContent = "AUTHENTIC (PRISTINE)";
    verdictBadge.textContent = "VERIFIED PRISTINE";
    verdictBadge.className = "verdict-badge cat-real";
    gauge.style.color = "var(--accent-emerald)";
    gauge.textContent = `${data.confidence_percentage}%`;
    verdictSummary.textContent = "Spectral decay conforms to 1/f² power law; Daubechies wavelet sub-bands confirm natural sensor continuity.";
  }

  if (data.metadata) {
    shaSpan.textContent = data.metadata.sha256_hash;
  }

  // 2. Spatial Tab
  document.getElementById("statCompositeScore").textContent = data.composite_forgery_score.toFixed(3);
  document.getElementById("statNeuralScore").textContent = (data.scores_breakdown.dual_branch_nn_score).toFixed(3);
  document.getElementById("statRiskTier").textContent = data.risk_tier;
  document.getElementById("statRiskTier").style.color = isFake ? "var(--accent-crimson)" : "var(--accent-emerald)";
  
  if (data.images) {
    document.getElementById("imgAlignedFace").src = data.images.aligned_face;
    document.getElementById("imgForgeryOverlay").src = data.images.forgery_overlay;
    document.getElementById("imgDftSpectrum").src = data.images.dft_spectrum;
    document.getElementById("imgWaveletGrid").src = data.images.wavelet_grid;
    document.getElementById("imgWaveletHeatmap").src = data.wavelet_analysis.wavelet_spatial_heatmap;
    document.getElementById("imgDctGrid").src = data.images.dct_grid;
    document.getElementById("imgHiFeReconstructed").src = data.images.c40_hife_reconstruction;
  }

  // 3. DFT Tab Stats & Plot
  document.getElementById("statDftScore").textContent = data.scores_breakdown.dft_anomaly_index.toFixed(3);
  document.getElementById("statDftAlpha").textContent = data.dft_analysis.decay_exponent_alpha.toFixed(3);
  document.getElementById("statDftHfRatio").textContent = data.dft_analysis.high_freq_energy_ratio.toFixed(4);
  document.getElementById("statDftSpikes").textContent = data.dft_analysis.significant_spikes;
  document.getElementById("dftScientificDesc").innerHTML = `<strong>Theoretical Finding:</strong> ${data.dft_analysis.scientific_interpretation}`;

  renderDFTChart(data.dft_analysis.chart_radii, data.dft_analysis.chart_observed, data.dft_analysis.chart_natural, isFake);

  // 4. Wavelet Tab Stats
  document.getElementById("statWaveletScore").textContent = data.scores_breakdown.wavelet_variance_index.toFixed(3);
  document.getElementById("statWaveletStd").textContent = data.wavelet_analysis.std_coeff.toFixed(4);
  document.getElementById("statWaveletKurtosis").textContent = data.wavelet_analysis.kurtosis_coeff.toFixed(2);
  document.getElementById("statWaveletHfRatio").textContent = data.wavelet_analysis.high_to_total_ratio.toFixed(4);
  document.getElementById("waveletScientificDesc").innerHTML = `<strong>Wolter et al. (2022) Basis:</strong> ${data.wavelet_analysis.scientific_interpretation}`;

  // 5. DCT Tab Stats & Benford Chart
  document.getElementById("statDctScore").textContent = data.scores_breakdown.dct_benford_index.toFixed(3);
  document.getElementById("statDctChi2").textContent = data.dct_analysis.chi_square_benford.toFixed(3);
  document.getElementById("statDctKl").textContent = data.dct_analysis.kl_divergence.toFixed(3);
  document.getElementById("statDctGridRatio").textContent = data.dct_analysis.block_grid_ratio.toFixed(3);
  document.getElementById("dctScientificDesc").innerHTML = `<strong>Barni et al. (2022) Finding:</strong> ${data.dct_analysis.scientific_interpretation}`;

  renderBenfordChart(data.dct_analysis.digits, data.dct_analysis.chart_empirical, data.dct_analysis.chart_theoretical);

  // 6. HiFE Laundering Tab
  document.getElementById("statHiFeResilience").textContent = `${(data.hife_analysis.resilience_index * 100).toFixed(0)}%`;
  document.getElementById("hifeScientificDesc").innerHTML = `<strong>Jie Gao et al. (2024) Enhancement:</strong> ${data.hife_analysis.scientific_interpretation}`;

  renderHiFEChart(data.hife_analysis.compression_labels, data.hife_analysis.standard_decay_pct, data.hife_analysis.hife_enhanced_pct);

  // 7. Daubert Prongs List
  renderDaubertProngs(data.daubert_forensic_report);
}

// CHART 1: DFT RADIAL DECAY
function renderDFTChart(radii, observed, natural, isFake) {
  const ctx = document.getElementById("dftDecayChart").getContext("2d");
  if (dftChart) dftChart.destroy();

  dftChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: radii,
      datasets: [
        {
          label: "Observed Energy P(r)",
          data: observed,
          borderColor: isFake ? "#ff0055" : "#00f2fe",
          backgroundColor: isFake ? "rgba(255, 0, 85, 0.1)" : "rgba(0, 242, 254, 0.1)",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        },
        {
          label: "Expected Natural Decay (1/f²)",
          data: natural,
          borderColor: "#00f5a0",
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Radial Frequency (r)", color: "#8e9db5", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        },
        y: {
          type: "logarithmic",
          title: { display: true, text: "Normalized Power (Log)", color: "#8e9db5", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#f0f4fc", font: { size: 10 } } }
      }
    }
  });
}

// CHART 2: BENFORD'S LAW DISTRIBUTION
function renderBenfordChart(digits, empirical, theoretical) {
  const ctx = document.getElementById("benfordChart").getContext("2d");
  if (benfordChart) benfordChart.destroy();

  benfordChart = new Chart(ctx, {
    data: {
      labels: digits,
      datasets: [
        {
          type: "bar",
          label: "Empirical AC Digit Freq (%)",
          data: empirical,
          backgroundColor: "rgba(79, 172, 254, 0.7)",
          borderColor: "#4facfe",
          borderWidth: 1,
          borderRadius: 4
        },
        {
          type: "line",
          label: "Theoretical Benford's Law (%)",
          data: theoretical,
          borderColor: "#ffb703",
          borderWidth: 2,
          pointBackgroundColor: "#ffb703",
          pointRadius: 4,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "First Significant Digit (1-9)", color: "#8e9db5", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        },
        y: {
          title: { display: true, text: "Probability (%)", color: "#8e9db5", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#f0f4fc", font: { size: 10 } } }
      }
    }
  });
}

// CHART 3: HiFE LAUNDERING DEFENSE RETENTION
function renderHiFEChart(labels, standardDrop, hifeEnhanced) {
  const ctx = document.getElementById("hifeCurveChart").getContext("2d");
  if (hifeChart) hifeChart.destroy();

  hifeChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Adaptive HiFE Enhancement (%)",
          data: hifeEnhanced,
          borderColor: "#00f5a0",
          backgroundColor: "rgba(0, 245, 160, 0.15)",
          fill: true,
          borderWidth: 3,
          pointRadius: 4,
          tension: 0.2
        },
        {
          label: "Standard Detector Decay (Low-Pass Filter Drop)",
          data: standardDrop,
          borderColor: "#ff0055",
          borderDash: [4, 4],
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        },
        y: {
          min: 0,
          max: 110,
          title: { display: true, text: "Forensic Cue Retention (%)", color: "#8e9db5", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8e9db5", font: { size: 9 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#f0f4fc", font: { size: 10 } } }
      }
    }
  });
}

// DAUBERT PRONGS DISPLAY
function renderDaubertProngs(report) {
  const container = document.getElementById("daubertProngsContainer");
  if (!container || !report.daubert_prongs) return;

  container.innerHTML = "";
  report.daubert_prongs.forEach(p => {
    const item = document.createElement("div");
    item.style.cssText = "background: rgba(10, 16, 28, 0.6); padding: 12px 16px; border-radius: 8px; border-left: 3px solid var(--accent-cyan); display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;";
    item.innerHTML = `
      <div>
        <div style="font-weight: 600; font-size: 0.85rem; color: #fff;">${p.criterion}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">${p.details}</div>
      </div>
      <span style="font-family: var(--font-mono); font-size: 0.7rem; font-weight: 700; color: var(--accent-emerald); background: rgba(0, 245, 160, 0.1); padding: 4px 8px; border-radius: 4px; white-space: nowrap;">${p.status}</span>
    `;
    container.appendChild(item);
  });
}

// POPULATE COURTROOM CERTIFICATE MODAL
function populateCertificateData(data) {
  const isFake = data.verdict.includes("MANIPULATED") || data.composite_forgery_score >= 0.50;

  document.getElementById("certFileName").textContent = data.metadata.filename;
  document.getElementById("certTimestamp").textContent = new Date(data.metadata.timestamp).toLocaleString();
  document.getElementById("certSha256").textContent = data.metadata.sha256_hash;
  document.getElementById("certResolution").textContent = `${data.metadata.resolution} (Aligned Facial Crop)`;
  
  const vCell = document.getElementById("certVerdictCell");
  vCell.textContent = data.verdict;
  vCell.style.color = isFake ? "#b00" : "#080";

  document.getElementById("certDftMetric").textContent = `Decay \u03B1 = ${data.dft_analysis.decay_exponent_alpha.toFixed(3)}, ${data.dft_analysis.significant_spikes} High-Freq Spikes`;
  document.getElementById("certDftVerdict").textContent = isFake ? "Periodic Spikes Detected" : "Smooth Natural Decay";
  document.getElementById("certDftVerdict").style.color = isFake ? "#b00" : "#080";

  document.getElementById("certDwtMetric").textContent = `Coeff Std = ${data.wavelet_analysis.std_coeff.toFixed(4)}, Kurtosis = ${data.wavelet_analysis.kurtosis_coeff.toFixed(2)}`;
  document.getElementById("certDwtVerdict").textContent = isFake ? "Elevated Coefficient Variance" : "Clean Wavelet Profile";
  document.getElementById("certDwtVerdict").style.color = isFake ? "#b00" : "#080";

  document.getElementById("certDctMetric").textContent = `Benford \u03C7\u00B2 = ${data.dct_analysis.chi_square_benford.toFixed(3)}, Grid = ${data.dct_analysis.block_grid_ratio.toFixed(3)}`;
  document.getElementById("certDctVerdict").textContent = isFake ? "Quantization Distortion" : "Natural Quantization";
  document.getElementById("certDctVerdict").style.color = isFake ? "#b00" : "#080";

  document.getElementById("certExpertOpinion").textContent = data.daubert_forensic_report.forensic_expert_summary;
}

// LIVE LAUNDERING SIMULATOR
async function runLaunderingSimulation() {
  const q = parseInt(document.getElementById("launderingSlider").value);
  showLoading(true, `SIMULATING SOCIAL MEDIA LAUNDERING AT JPEG Q=${q}...`);
  try {
    const res = await fetch("/api/simulate-laundering", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quality: q,
        sample_id: currentSelectedSample
      })
    });
    const result = await res.json();
    document.getElementById("imgHiFeReconstructed").src = result.residual_hf_image;
  } catch (err) {
    console.error("Laundering simulation error:", err);
  } finally {
    showLoading(false);
  }
}

// HELPER: SPINNER OVERLAY
function showLoading(show, message = "ANALYZING MULTI-LEVEL FREQUENCIES...") {
  const overlay = document.getElementById("loadingOverlay");
  const text = document.getElementById("loadingProgressText");
  if (show) {
    text.textContent = message;
    overlay.classList.add("active");
  } else {
    overlay.classList.remove("active");
  }
}
