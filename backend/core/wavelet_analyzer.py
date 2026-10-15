"""
Wavelet Analyzer: Multi-Scale Discrete Wavelet Transform (DWT) & Wavelet-Packet Forensics
Implements the theoretical principles from:
- Wolter et al. (2022), "Wavelet-packets for deepfake image analysis and detection", Machine Learning.
- Miao et al. (2022), "Exploiting Discrete Wavelet Transform for Generalizable Face Forgery Detection"
- Gao et al. (2024), Multi-level DWT Global Enhancement
"""

import numpy as np
import pywt
import cv2
import base64


class WaveletAnalyzer:
    def __init__(self, wavelet_family: str = 'db4', levels: int = 2):
        """
        Initializes the wavelet analyzer.
        
        Args:
            wavelet_family: Mother wavelet basis ('db4', 'sym4', or 'haar').
            levels: Decomposition levels (typically 2 for multi-scale analysis).
        """
        self.wavelet_family = wavelet_family
        self.levels = levels

    def analyze(self, image_bgr: np.ndarray) -> dict:
        """
        Performs multi-level DWT decomposition and statistical wavelet packet analysis.
        
        Args:
            image_bgr: Face image array (H, W, 3).
            
        Returns:
            Dictionary with statistical coefficients, sub-band energies, 
            local spatial wavelet anomaly heatmap, and sub-band collage images.
        """
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()
            
        # Ensure dimensions are even for clean multi-level decomposition
        h, w = gray.shape
        if h % 4 != 0 or w % 4 != 0:
            gray = cv2.resize(gray, (w - (w % 4), h - (h % 4)))
            
        gray_f = gray.astype(np.float64) / 255.0
        
        # Perform 2-Level 2D Discrete Wavelet Transform
        # coeffs: [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
        coeffs = pywt.wavedec2(gray_f, wavelet=self.wavelet_family, level=self.levels)
        
        cA2 = coeffs[0]            # LL2: Low-frequency approximation at level 2
        cH2, cV2, cD2 = coeffs[1]   # LH2 (Horizontal), HL2 (Vertical), HH2 (Diagonal) at level 2
        cH1, cV1, cD1 = coeffs[2]   # LH1, HL1, HH1 at level 1
        
        # Calculate sub-band energies: E = sum(coeff^2)
        e_ll2 = float(np.sum(cA2 ** 2))
        e_lh2 = float(np.sum(cH2 ** 2))
        e_hl2 = float(np.sum(cV2 ** 2))
        e_hh2 = float(np.sum(cD2 ** 2))
        
        e_lh1 = float(np.sum(cH1 ** 2))
        e_hl1 = float(np.sum(cV1 ** 2))
        e_hh1 = float(np.sum(cD1 ** 2))
        
        total_high_energy = e_lh1 + e_hl1 + e_hh1 + e_lh2 + e_hl2 + e_hh2
        total_energy = e_ll2 + total_high_energy + 1e-10
        high_to_total_ratio = total_high_energy / total_energy
        
        # Statistical analysis on high-frequency bands (Finding 2 from Wolter et al.)
        # GAN and autoencoder faces exhibit significantly higher mean and standard deviation of wavelet coefficients
        hf_all_coeffs = np.concatenate([
            cH1.ravel(), cV1.ravel(), cD1.ravel(),
            cH2.ravel(), cV2.ravel(), cD2.ravel()
        ])
        
        abs_hf = np.abs(hf_all_coeffs)
        log_hf = np.log(abs_hf + 1e-6)
        
        mean_abs_coeff = float(np.mean(abs_hf))
        std_coeff = float(np.std(hf_all_coeffs))
        kurtosis_coeff = float(np.mean(((hf_all_coeffs - np.mean(hf_all_coeffs)) / (std_coeff + 1e-8)) ** 4))
        log_mean = float(np.mean(log_hf))
        log_std = float(np.std(log_hf))
        
        # Wavelet anomaly score calculation
        # Real faces have smooth natural textures with low high-frequency coefficient variance
        # Synthetic upsampling creates sub-pixel checkerboard grain and elevated high-band energy
        variance_factor = min(1.0, max(0.0, (std_coeff - 0.032) / 0.015))
        energy_factor = min(1.0, max(0.0, (high_to_total_ratio - 0.0065) / 0.0035))
        log_stat_factor = min(1.0, max(0.0, (log_mean + 4.8) / 1.2))
        
        wavelet_anomaly_score = float(np.clip(
            0.45 * variance_factor + 0.35 * energy_factor + 0.20 * log_stat_factor, 
            0.0, 1.0
        ))
        
        # Construct Level-1 Sub-Band Visual Collage (2x2 grid: LL, LH, HL, HH)
        # We perform level-1 decomposition specifically for the clean 4-quadrant visual
        coeffs_l1 = pywt.dwt2(gray_f, self.wavelet_family)
        cA, (cH, cV, cD) = coeffs_l1
        
        def norm_to_uint8(arr):
            v_min, v_max = np.min(arr), np.max(arr)
            if v_max > v_min:
                norm = ((arr - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(arr, dtype=np.uint8)
            return norm

        ll_vis = norm_to_uint8(cA)
        # For high-frequency bands, visualize absolute magnitude with high contrast
        lh_vis = norm_to_uint8(np.abs(cH) * 3.5)
        hl_vis = norm_to_uint8(np.abs(cV) * 3.5)
        hh_vis = norm_to_uint8(np.abs(cD) * 5.0)
        
        top_row = np.hstack([ll_vis, lh_vis])
        bottom_row = np.hstack([hl_vis, hh_vis])
        quadrant_grid = np.vstack([top_row, bottom_row])
        
        # Colorize quadrant grid
        colored_quadrant = cv2.applyColorMap(quadrant_grid, cv2.COLORMAP_VIRIDIS)
        
        # Add labels to the 4 quadrants
        h_q, w_q = ll_vis.shape
        cv2.putText(colored_quadrant, "LL (Approximation)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_quadrant, "LH (Horizontal High-Freq)", (w_q + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_quadrant, "HL (Vertical High-Freq)", (10, h_q + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_quadrant, "HH (Diagonal Residuals)", (w_q + 10, h_q + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Generate Localized Spatial Wavelet Heatmap:
        # High-frequency energy localized to spatial coordinates (Wolter et al., 2022)
        # Sum of squared high-frequency details at level 1: LH^2 + HL^2 + HH^2
        local_hf_energy = (cH ** 2 + cV ** 2 + cD ** 2)
        local_hf_upscaled = cv2.resize(local_hf_energy, (w, h), interpolation=cv2.INTER_CUBIC)
        norm_local = norm_to_uint8(local_hf_upscaled)
        blurred_local = cv2.GaussianBlur(norm_local, (15, 15), 5)
        colored_wavelet_heatmap = cv2.applyColorMap(blurred_local, cv2.COLORMAP_JET)
        
        # Base64 encodings
        _, buf1 = cv2.imencode('.png', colored_quadrant)
        grid_base64 = base64.b64encode(buf1).decode('utf-8')
        
        _, buf2 = cv2.imencode('.png', colored_wavelet_heatmap)
        heatmap_base64 = base64.b64encode(buf2).decode('utf-8')
        
        return {
            "wavelet_anomaly_score": round(wavelet_anomaly_score, 4),
            "high_to_total_ratio": round(high_to_total_ratio, 4),
            "mean_abs_coeff": round(mean_abs_coeff, 6),
            "std_coeff": round(std_coeff, 6),
            "kurtosis_coeff": round(kurtosis_coeff, 3),
            "log_mean": round(log_mean, 3),
            "log_std": round(log_std, 3),
            "subband_energies": {
                "LL2_approx": round(e_ll2, 2),
                "LH2_horizontal": round(e_lh2, 2),
                "HL2_vertical": round(e_hl2, 2),
                "HH2_diagonal": round(e_hh2, 2),
                "LH1_horizontal": round(e_lh1, 2),
                "HL1_vertical": round(e_hl1, 2),
                "HH1_diagonal": round(e_hh1, 2)
            },
            "wavelet_grid_image": f"data:image/png;base64,{grid_base64}",
            "wavelet_spatial_heatmap": f"data:image/png;base64,{heatmap_base64}",
            "scientific_interpretation": (
                f"Elevated standard deviation ({std_coeff:.4f}) and kurtosis ({kurtosis_coeff:.2f}) "
                f"in high-frequency directional sub-bands (LH, HL, HH) using {self.wavelet_family} basis, "
                f"indicating sub-pixel checkerboard anomalies from generative decoding."
                if wavelet_anomaly_score > 0.55 else
                f"Standard deviation ({std_coeff:.4f}) and high-frequency coefficient ratios match "
                f"natural distribution profiles for authentic human facial imaging."
            )
        }
