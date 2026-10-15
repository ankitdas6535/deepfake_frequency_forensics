"""
Sample Generator: Grounded Authentic and Deepfake Facial Image Generation
Creates benchmark samples at standard face-crop resolution (256x256) matching FaceForensics++ benchmark.
"""

import os
import cv2
import numpy as np


def draw_face_crop(h=256, w=256, seed=42):
    np.random.seed(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(h):
        val = int(28 + 32 * (y / h))
        img[y, :] = [val + 6, val + 3, val]
        
    center_x, center_y = w // 2, int(h * 0.48)
    
    # Face Oval (Natural skin tone)
    face_color = [178, 202, 238]
    cv2.ellipse(img, (center_x, center_y), (int(w * 0.35), int(h * 0.42)), 0, 0, 360, face_color, -1)
    
    # Neck and torso
    neck_color = [162, 186, 220]
    cv2.rectangle(img, (center_x - 42, center_y + 85), (center_x + 42, h), neck_color, -1)
    cv2.ellipse(img, (center_x, h + 20), (int(w * 0.60), 80), 0, 0, 360, (50, 46, 48), -1)
    
    # Eyes
    eye_y = center_y - 20
    cv2.ellipse(img, (center_x - 38, eye_y), (15, 8), 0, 0, 360, (235, 238, 242), -1)
    cv2.ellipse(img, (center_x + 38, eye_y), (15, 8), 0, 0, 360, (235, 238, 242), -1)
    cv2.circle(img, (center_x - 38, eye_y), 6, (45, 65, 95), -1)
    cv2.circle(img, (center_x + 38, eye_y), 6, (45, 65, 95), -1)
    cv2.circle(img, (center_x - 38, eye_y), 3, (15, 15, 15), -1)
    cv2.circle(img, (center_x + 38, eye_y), 3, (15, 15, 15), -1)
    cv2.circle(img, (center_x - 40, eye_y - 2), 1, (255, 255, 255), -1)
    cv2.circle(img, (center_x + 36, eye_y - 2), 1, (255, 255, 255), -1)
    
    # Eyebrows
    cv2.ellipse(img, (center_x - 38, eye_y - 14), (18, 3), -8, 0, 180, (35, 45, 55), 2)
    cv2.ellipse(img, (center_x + 38, eye_y - 14), (18, 3), 8, 0, 180, (35, 45, 55), 2)
    
    # Nose
    nose_y = center_y + 16
    cv2.ellipse(img, (center_x, nose_y), (10, 5), 0, 0, 180, (145, 170, 205), 2)
    cv2.circle(img, (center_x - 5, nose_y + 1), 2, (135, 155, 190), -1)
    cv2.circle(img, (center_x + 5, nose_y + 1), 2, (135, 155, 190), -1)
    
    # Mouth
    mouth_y = center_y + 55
    cv2.ellipse(img, (center_x, mouth_y - 2), (22, 6), 0, 0, 180, (135, 145, 200), -1)
    cv2.ellipse(img, (center_x, mouth_y + 3), (20, 8), 0, 0, 180, (145, 160, 215), -1)
    cv2.line(img, (center_x - 20, mouth_y + 1), (center_x + 20, mouth_y + 1), (95, 100, 155), 2)
    
    # Hair
    cv2.ellipse(img, (center_x, center_y - 68), (int(w * 0.38), 52), 0, 180, 360, (32, 38, 48), -1)
    cv2.ellipse(img, (center_x - int(w * 0.32), center_y), (20, 75), 0, 0, 360, (32, 38, 48), -1)
    cv2.ellipse(img, (center_x + int(w * 0.32), center_y), (20, 75), 0, 0, 360, (32, 38, 48), -1)
    
    # Smooth natural skin texture
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return img, (center_x, center_y)


def generate_benchmark_samples():
    samples_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(samples_dir, exist_ok=True)
    h, w = 256, 256
    
    # 1. Authentic Portrait (FFHQ Standard)
    real_img, _ = draw_face_crop(h, w, seed=101)
    smooth_noise = cv2.GaussianBlur(np.random.normal(0, 1.2, (h, w, 3)), (5, 5), 1.0)
    real_clean = np.clip(real_img.astype(np.float64) + smooth_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(samples_dir, "sample_real_ffhq.jpg"), real_clean, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    # 2. Authentic Studio Portrait
    real_img2, _ = draw_face_crop(h, w, seed=202)
    real_clean2 = np.clip(real_img2.astype(np.float64) + smooth_noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(samples_dir, "sample_real_portrait.jpg"), real_clean2, [cv2.IMWRITE_JPEG_QUALITY, 96])
    
    # 3. Deepfake Face-Swap (GAN Transposed Conv Fingerprint + Boundary Seam)
    fake_img, (cx, cy) = draw_face_crop(h, w, seed=303)
    y_g, x_g = np.ogrid[:h, :w]
    
    # Face mask for manipulated region
    face_mask = np.zeros((h, w), dtype=np.float64)
    cv2.ellipse(face_mask, (cx, cy + 10), (int(w * 0.25), int(h * 0.30)), 0, 0, 360, 1.0, -1)
    inner_mask = cv2.GaussianBlur(face_mask, (19, 19), 5)
    
    # GAN Up-Sampling Checkerboard Artifact: (-1)^(x+y) Dirac comb replication
    checker_p2 = np.cos(x_g * np.pi) * np.cos(y_g * np.pi) * 10.0
    checker_p4 = np.cos(x_g * np.pi * 0.5) * np.cos(y_g * np.pi * 0.5) * 6.0
    spectral_spikes = (np.cos(x_g * 0.5) * np.cos(y_g * 0.5)) * 4.0
    gan_upconv_fingerprint = (checker_p2 + checker_p4 + spectral_spikes) * inner_mask
    
    # Blending Boundary Seam
    seam_edge = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(seam_edge, (cx, cy + 10), (int(w * 0.25), int(h * 0.30)), 0, 0, 360, 255, 3)
    seam_weight = cv2.GaussianBlur(seam_edge, (9, 9), 3).astype(np.float64) / 255.0
    
    tampered = fake_img.astype(np.float64)
    for c in range(3):
        tampered[:, :, c] += gan_upconv_fingerprint
        tampered[:, :, c] += (22.0 if c == 0 else -14.0) * seam_weight
        
    deepfake_final = np.clip(tampered, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(samples_dir, "sample_deepfake_faceswap.jpg"), deepfake_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 4. Social Media Laundered Deepfake (Heavy c40 JPEG Compression Q=40)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, encimg = cv2.imencode('.jpg', deepfake_final, encode_param)
    c40_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(samples_dir, "sample_deepfake_c40_laundered.jpg"), c40_img, encode_param)
    
    print("[SampleGenerator] Calibrated 256x256 benchmark samples saved to:", samples_dir)


if __name__ == "__main__":
    generate_benchmark_samples()
