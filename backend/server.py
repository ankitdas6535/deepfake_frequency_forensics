"""
Forensic Server: FastAPI Backend for Multi-Level Frequency Deepfake Detection
Serves REST API endpoints and static frontend forensic interface.
"""

import os
import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.forensic_engine import ForensicEngine

app = FastAPI(
    title="Deepfake Frequency Forensics API",
    description="Multi-Level Frequency Analysis for Deepfake Image Detection (DWT db4, 2D DFT, DCT, HiFE, Dual-Branch Net)",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Forensic Engine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")
FRONTEND_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "frontend")

engine = ForensicEngine()


class SampleRequest(BaseModel):
    sample_id: str


class CompressionRequest(BaseModel):
    quality: int = 40
    sample_id: str = "sample_deepfake_faceswap"


@app.get("/api/system-info")
async def get_system_info():
    """Returns hardware and forensic architecture specifications."""
    return {
        "status": "OPERATIONAL",
        "system_title": "Deepfake Detection Using Multi-Level Frequency Analysis",
        "device": "CUDA (GPU)" if torch.cuda.is_available() else "CPU (SIMD Optimized)",
        "neural_backbone": "Dual-Branch Spatio-Frequency Network with Two-Stage Cross-Attention",
        "wavelet_basis": "Daubechies-4 (db4) & Symlets-4 (sym4) Multi-Scale Packet Decomposition",
        "fourier_transform": "2D Fast Fourier Transform (FFT2) with Azimuthal Radial Decay Modeling",
        "dct_transform": "8x8 Block-wise Orthogonal DCT with Benford's Law Quantization Profiling",
        "laundering_defense": "Adaptive High-Frequency Enhancement (HiFE) Network for JPEG c23/c40 Robustness",
        "courtroom_standard": "Daubert Standard for Legal Admissibility (Gap 10 Compliant)"
    }


@app.get("/api/samples")
async def list_samples():
    """Lists available pre-generated benchmark samples."""
    samples = [
        {
            "id": "sample_real_ffhq",
            "title": "Authentic Portrait (FFHQ Standard)",
            "category": "REAL",
            "description": "Natural optical camera portrait with continuous 1/f^2 spectral decay, smooth skin pores, and natural Benford distribution.",
            "ground_truth": "REAL",
            "file": "sample_real_ffhq.jpg"
        },
        {
            "id": "sample_real_portrait",
            "title": "Authentic Studio Portrait",
            "category": "REAL",
            "description": "High-fidelity uncompressed studio lighting portrait with organic Daubechies wavelet sub-band coefficients.",
            "ground_truth": "REAL",
            "file": "sample_real_portrait.jpg"
        },
        {
            "id": "sample_deepfake_faceswap",
            "title": "Deepfake Face-Swap (GAN Up-sampled)",
            "category": "DEEPFAKE",
            "description": "Transposed-convolution GAN upsampling with Dirac-comb checkerboard spectral spikes and blending boundary seams.",
            "ground_truth": "DEEPFAKE",
            "file": "sample_deepfake_faceswap.jpg"
        },
        {
            "id": "sample_deepfake_c40_laundered",
            "title": "Social Media Laundered Deepfake (JPEG c40)",
            "category": "COMPRESSED_DEEPFAKE",
            "description": "Severely compressed (Q=38) deepfake designed to evaluate HiFE adaptive high-frequency reconstruction under social media laundering.",
            "ground_truth": "DEEPFAKE",
            "file": "sample_deepfake_c40_laundered.jpg"
        }
    ]
    return samples


@app.post("/api/analyze-sample")
async def analyze_sample(req: SampleRequest):
    """Analyzes a selected benchmark sample."""
    sample_file = f"{req.sample_id}.jpg"
    file_path = os.path.join(SAMPLES_DIR, sample_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Sample {req.sample_id} not found.")
        
    with open(file_path, "rb") as f:
        image_bytes = f.read()
        
    results = engine.analyze_image(image_bytes, filename=sample_file)
    return results


@app.post("/api/analyze")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """Analyzes any user-uploaded face image."""
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        results = engine.analyze_image(image_bytes, filename=file.filename)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/simulate-laundering")
async def simulate_laundering(req: CompressionRequest):
    """
    Simulates variable lossy compression laundering (Q=10 to Q=95) on a sample image.
    """
    file_path = os.path.join(SAMPLES_DIR, f"{req.sample_id}.jpg")
    if not os.path.exists(file_path):
        file_path = os.path.join(SAMPLES_DIR, "sample_deepfake_faceswap.jpg")
        
    img = cv2.imread(file_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, req.quality))]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    compressed_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    # Calculate high-frequency residual
    gray = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Laplacian(gray, cv2.CV_64F)
    norm_sobel = cv2.normalize(np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_sobel = cv2.applyColorMap(norm_sobel, cv2.COLORMAP_TURBO)
    
    _, buf1 = cv2.imencode('.jpg', compressed_img)
    _, buf2 = cv2.imencode('.png', color_sobel)
    
    return {
        "quality": req.quality,
        "compressed_image": f"data:image/jpeg;base64,{base64.b64encode(buf1).decode('utf-8')}",
        "residual_hf_image": f"data:image/png;base64,{base64.b64encode(buf2).decode('utf-8')}"
    }


# Mount frontend static directory if exists
if os.path.exists(FRONTEND_DIR):
    @app.get("/style.css")
    async def serve_css():
        return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")

    @app.get("/app.js")
    async def serve_js():
        return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
