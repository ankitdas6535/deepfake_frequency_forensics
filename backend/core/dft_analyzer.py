"""
DFT Analyzer: 2D Discrete Fourier Transform & Azimuthal Radial Spectrum Analysis
Implements the theoretical principles from:
- Frank et al. (2020), "Leveraging frequency analysis for deepfake image recognition"
- Dzanic et al. (2020), "Fourier spectrum discrepancies in deep network generated images"
- Durall et al. (2020), "Watch your up-convolution: CNN based generative deep neural networks"
"""

import numpy as np
import cv2
import io
import base64
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DFTAnalyzer:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def analyze(self, image_bgr: np.ndarray) -> dict:
        """
        Performs 2D DFT analysis and Azimuthal Radial Average decomposition.
        
        Args:
            image_bgr: Input face image (H, W, 3) in BGR format.
            
        Returns:
            Dictionary containing:
            - dft_anomaly_score: float [0, 1]
            - high_freq_energy_ratio: float
            - checkerboard_spike_count: int
            - decay_exponent_alpha: float
            - radial_profile: list of float values along radius
            - theoretical_decay: list of expected natural decay values
            - spectrum_base64: base64 encoded PNG of the centered power spectrum
        """
        # Convert to grayscale and resize to consistent forensic scale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()
            
        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA).astype(np.float64)
        h, w = gray.shape
        
        # Apply 2D Hanning window to prevent spectral leakage from image borders
        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window_2d = np.outer(win_y, win_x)
        windowed_gray = gray * window_2d
        
        # 2D Fast Fourier Transform and shift zero-frequency component to center
        fft2 = np.fft.fft2(windowed_gray)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)
        power_spectrum = magnitude ** 2
        
        # Normalized log magnitude for visualization: log(1 + |F(u,v)|)
        log_magnitude = np.log1p(magnitude)
        norm_log_mag = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap (Inferno/Jet) for scientific forensic representation
        colored_spectrum = cv2.applyColorMap(norm_log_mag, cv2.COLORMAP_INFERNO)
        
        # Calculate Azimuthal Radial Average (1D Power Profile)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius_grid = np.hypot(x - center_x, y - center_y).astype(int)
        
        max_radius = min(center_y, center_x)
        radial_profile = []
        radius_steps = np.arange(1, max_radius)
        
        for r in radius_steps:
            mask = (radius_grid == r)
            if np.any(mask):
                radial_profile.append(float(np.mean(power_spectrum[mask])))
            else:
                radial_profile.append(0.0)
                
        radial_profile = np.array(radial_profile)
        # Normalize radial energy for standardized comparison (exclude DC)
        norm_radial = radial_profile / (np.max(radial_profile) + 1e-8)
            
        # Theoretical natural decay P(r) = r^(-alpha)
        # Fit power law on mid-to-high frequencies (r = 5 to max_radius)
        r_fit = radius_steps[4:]
        val_fit = np.clip(norm_radial[4:], 1e-12, None)
        log_r = np.log(r_fit)
        log_val = np.log(val_fit)
        
        # Linear regression in log-log domain
        slope, intercept = np.polyfit(log_r, log_val, 1)
        decay_exponent_alpha = -float(slope)
        
        # Generate expected natural curve
        expected_natural = np.exp(intercept) * (radius_steps ** slope)
        expected_natural = expected_natural / (expected_natural[0] + 1e-8)
        
        # Detect mid-to-high frequency checkerboard spikes & energy:
        # Mid-frequency band: 0.10 * max_radius to 0.35 * max_radius
        # High-frequency band: > 0.35 * max_radius
        mid_idx_start = int(0.10 * max_radius)
        mid_idx_end = int(0.35 * max_radius)
        high_idx_start = mid_idx_end
        
        mid_energy = float(np.sum(radial_profile[mid_idx_start:mid_idx_end]) + 1e-8)
        high_energy = float(np.sum(radial_profile[high_idx_start:]) + 1e-8)
        high_to_mid_ratio = high_energy / mid_energy
        
        # Spike detection: compare against moving average
        kernel_size = 7
        pad_size = kernel_size // 2
        padded = np.pad(norm_radial, pad_size, mode='edge')
        moving_avg = np.convolve(padded, np.ones(kernel_size)/kernel_size, mode='valid')
        
        # Peak deviations in high frequencies
        diff = norm_radial[high_idx_start:] - moving_avg[high_idx_start:]
        std_diff = np.std(diff)
        significant_spikes = int(np.sum(diff > max(1e-6, std_diff * 1.5)))
        
        # Anomaly scoring:
        # Natural images have high_to_mid_ratio < 0.04
        # GAN/Deepfake images have elevated ratio (> 0.08) and periodic spikes
        ratio_anomaly = min(1.0, max(0.0, (high_to_mid_ratio - 0.035) / 0.09))
        spike_anomaly = min(1.0, significant_spikes / 8.0)
        
        # Alpha exponent anomaly: natural images have alpha around 2.8 - 3.4;
        # synthetic images have flatter tails (alpha < 2.7)
        if decay_exponent_alpha < 2.75:
            alpha_anomaly = min(1.0, (2.75 - decay_exponent_alpha) / 1.0)
        else:
            alpha_anomaly = 0.0
            
        dft_anomaly_score = float(np.clip(
            0.50 * ratio_anomaly + 0.35 * spike_anomaly + 0.15 * alpha_anomaly, 
            0.0, 1.0
        ))
        
        # Downsample radial profiles for chart serialization (e.g. 50 points)
        sample_indices = np.linspace(0, len(radius_steps) - 1, 50, dtype=int)
        chart_radii = [int(radius_steps[i]) for i in sample_indices]
        chart_observed = [round(float(norm_radial[i]), 6) for i in sample_indices]
        chart_natural = [round(float(expected_natural[i]), 6) for i in sample_indices]
        
        # Encode spectrum image to base64
        _, buffer = cv2.imencode('.png', colored_spectrum)
        spectrum_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "dft_anomaly_score": round(dft_anomaly_score, 4),
            "high_freq_energy_ratio": round(high_to_mid_ratio, 4),
            "decay_exponent_alpha": round(decay_exponent_alpha, 3),
            "significant_spikes": significant_spikes,
            "chart_radii": chart_radii,
            "chart_observed": chart_observed,
            "chart_natural": chart_natural,
            "spectrum_image": f"data:image/png;base64,{spectrum_base64}",
            "scientific_interpretation": (
                "Spectral power exhibits abnormal plateau in mid-to-high frequency bands with periodic spikes "
                "characteristic of transposed-convolution generator upsampling (Dirac comb artifact)."
                if dft_anomaly_score > 0.55 else
                "Spectral power exhibits smooth logarithmic radial decay consistent with natural unmanipulated optical imaging."
            )
        }
