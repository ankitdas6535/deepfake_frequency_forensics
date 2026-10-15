# 🔬 Spectro-Forensic Lab — Deepfake Frequency Forensics

> **Dual-Branch Spatio-Frequency Deepfake Detection** using Multi-Scale Wavelet Packets (db4), 2D DFT Radial Decay, 8×8 DCT Benford Analysis, and Adaptive HiFE Compression Defense.

---

## 📸 Features

Upload any facial image (JPEG/PNG/WebP) and the system performs:
- **2D DFT** — detects Dirac comb spikes from GAN transposed convolutions
- **DWT Wavelet Packets (db4)** — exposes sub-pixel boundary artifacts
- **8×8 DCT + Benford's Law** — catches quantization fingerprint distortions
- **HiFE Compression Defense** — stays accurate even after social-media JPEG laundering
- **Daubert-Compliant Report** — generates a printable courtroom forensic certificate

---

## 🗂️ Project Structure

```
deepfake_frequency_forensics/
├── backend/
│   ├── server.py                      # FastAPI server (entry point)
│   ├── train_classifier.py            # Model training script
│   ├── core/
│   │   ├── forensic_engine.py         # Main analysis orchestrator
│   │   ├── dft_analyzer.py            # 2D Fourier analysis
│   │   ├── wavelet_analyzer.py        # DWT (db4) analysis
│   │   ├── dct_analyzer.py            # DCT + Benford analysis
│   │   ├── hife_enhancer.py           # HiFE compression defense
│   │   └── spatio_frequency_model.py  # Dual-Branch Neural Network
│   ├── samples/
│   │   └── generate_samples.py        # Generates benchmark test images
│   └── weights/                       # Trained model weights (.pt files)
├── frontend/
│   ├── index.html                     # Main UI
│   ├── app.js                         # Frontend controller
│   └── style.css                      # Dark-mode forensic UI styles
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/deepfake_frequency_forensics.git
cd deepfake_frequency_forensics
```

### 2. Create a Python virtual environment
```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate
# Activate — macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU Note:** For faster inference install the CUDA PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/)

### 4. Generate benchmark sample images
```bash
python backend/samples/generate_samples.py
```

### 5. (Optional) Train the classifier
```bash
python backend/train_classifier.py
```

### 6. Start the server
```bash
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the app
Go to **http://localhost:8000** in your browser.

---

## 🧪 Using the Website

1. **Upload** a facial image via drag & drop, or pick one of the **4 built-in benchmark samples**
2. Click **⚡ Run Multi-Level Frequency Inspection**
3. Explore the 6 forensic tabs:
   - 🖼️ Spatial Localization Heatmap
   - 🌐 2D DFT & Radial Decay
   - 🌊 DWT Wavelet Sub-Bands
   - 🔢 8×8 DCT & Benford's Law
   - 🛡️ Compression Defense (HiFE)
   - ⚖️ Daubert Legal XAI
4. Click **📜 Daubert Certificate** to export a courtroom-ready PDF report

---

## 🧬 Scientific Basis

| Method | Reference |
|---|---|
| DFT Dirac Comb Detection | Frank et al. (2020) |
| Wavelet Packet Sub-Bands | Wolter et al. (2022) |
| DCT Benford's Law | Barni et al. (2022) |
| HiFE Compression Defense | Jie Gao et al. (2024) |

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for the full package list
