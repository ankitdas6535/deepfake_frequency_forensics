"""
Calibrates and trains the DualBranchSpatioFrequencyNet classifier end-to-end
on authentic and deepfake multi-level frequency representations, including compressed faces.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import pywt

from core.spatio_frequency_model import DualBranchSpatioFrequencyNet
from samples.generate_samples import draw_face_base


def create_training_pair(is_deepfake: bool, seed: int, compress: bool = False):
    h, w = 256, 256
    img, (cx, cy) = draw_face_base(h, w, seed=seed)
    
    # Smooth optical noise for all faces
    smooth_noise = cv2.GaussianBlur(np.random.normal(0, 1.2, (h, w, 3)), (5, 5), 1.0)
    img_f = np.clip(img.astype(np.float64) + smooth_noise, 0, 255)
    
    if is_deepfake:
        y_g, x_g = np.ogrid[:h, :w]
        # Transposed convolution Dirac-comb artifact
        checker = np.cos(x_g * np.pi) * np.cos(y_g * np.pi) * 10.0
        checker_p4 = np.cos(x_g * np.pi * 0.5) * np.cos(y_g * np.pi * 0.5) * 6.0
        spikes = (np.cos(x_g * 0.5) * np.cos(y_g * 0.5)) * 4.0
        
        face_mask = np.zeros((h, w), dtype=np.float64)
        cv2.ellipse(face_mask, (cx, cy + 10), (int(w * 0.25), int(h * 0.30)), 0, 0, 360, 1.0, -1)
        inner_mask = cv2.GaussianBlur(face_mask, (19, 19), 5)
        
        # Boundary seam
        seam_edge = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(seam_edge, (cx, cy + 10), (int(w * 0.25), int(h * 0.30)), 0, 0, 360, 255, 3)
        seam_weight = cv2.GaussianBlur(seam_edge, (9, 9), 3).astype(np.float64) / 255.0
        
        for c in range(3):
            img_f[:, :, c] += (checker + checker_p4 + spikes) * inner_mask
            img_f[:, :, c] += (22.0 if c == 0 else -14.0) * seam_weight
            
    final_img = np.clip(img_f, 0, 255).astype(np.uint8)
    
    if compress:
        q = np.random.choice([35, 45, 60, 75])
        _, encimg = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        final_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    # RGB Tensor (3, 256, 256)
    rgb_tensor = torch.from_numpy(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    
    # 4-Channel Frequency Tensor (LL, LH, HL, HH from DWT db4)
    gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'db4')
    cA_r = cv2.resize(cA, (256, 256))
    cH_r = cv2.resize(np.abs(cH) * 5.0, (256, 256))
    cV_r = cv2.resize(np.abs(cV) * 5.0, (256, 256))
    cD_r = cv2.resize(np.abs(cD) * 8.0, (256, 256))
    freq_stack = np.stack([cA_r, cH_r, cV_r, cD_r], axis=0)
    freq_tensor = torch.from_numpy(freq_stack).float()
    
    target = 1.0 if is_deepfake else 0.0
    return rgb_tensor, freq_tensor, torch.tensor([target], dtype=torch.float32)


def train_and_save():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "spatio_frequency_weights.pt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBranchSpatioFrequencyNet(feature_dim=256).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    print("[Trainer] Generating multi-codec training dataset...")
    dataset = []
    # 40 real samples (uncompressed + compressed)
    for i in range(25):
        dataset.append(create_training_pair(is_deepfake=False, seed=100 + i, compress=False))
        dataset.append(create_training_pair(is_deepfake=False, seed=300 + i, compress=True))
    # 40 deepfake samples (uncompressed + c40/c23 compressed)
    for i in range(25):
        dataset.append(create_training_pair(is_deepfake=True, seed=600 + i, compress=False))
        dataset.append(create_training_pair(is_deepfake=True, seed=800 + i, compress=True))
        
    print(f"[Trainer] Training Dual-Branch Spatio-Frequency Network on {len(dataset)} items...")
    model.train()
    
    # Batch size 8
    batch_size = 8
    num_batches = len(dataset) // batch_size
    
    for epoch in range(12):
        np.random.shuffle(dataset)
        total_loss = 0.0
        correct = 0
        
        for b in range(num_batches):
            batch = dataset[b * batch_size:(b + 1) * batch_size]
            rgbs = torch.stack([item[0] for item in batch]).to(device)
            freqs = torch.stack([item[1] for item in batch]).to(device)
            targets = torch.stack([item[2] for item in batch]).to(device)
            
            optimizer.zero_grad()
            preds, _, _, _ = model(rgbs, freqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            correct += int(((preds > 0.5) == (targets > 0.5)).sum().item())
            
        acc = (correct / (num_batches * batch_size)) * 100.0
        print(f"Epoch {epoch+1:02d}/12 | Loss: {total_loss/(num_batches*batch_size):.4f} | Accuracy: {acc:.1f}%")
        
    torch.save(model.state_dict(), weights_path)
    print(f"[Trainer] Calibrated model weights successfully saved to: {weights_path}")


if __name__ == "__main__":
    train_and_save()
