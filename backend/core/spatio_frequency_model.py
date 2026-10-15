"""
Dual-Branch Spatio-Frequency Neural Architecture
Implements the end-to-end framework proposed in the research thesis:
- Spatial Branch: Deep convolutional representation capturing facial boundaries & local textures
- Frequency Branch: Multi-level wavelet packet and spectral decomposition stream
- Two-Stage Cross-Attention Module: Fusing spatial context with localized frequency anomalies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SpatialBranch(nn.Module):
    """
    Extracts deep spatial contextual cues, facial boundary seams, and color aberrations.
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, feature_dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat_stem = self.stem(x)       # (B, 32, 64, 64)
        feat_l1 = self.layer1(feat_stem) # (B, 64, 32, 32)
        feat_l2 = self.layer2(feat_l1)   # (B, 128, 16, 16)
        feat_map = self.layer3(feat_l2)  # (B, 256, 8, 8)
        vector = self.pool(feat_map).flatten(1) # (B, 256)
        return vector, feat_map


class FrequencyBranch(nn.Module):
    """
    Extracts multi-level spectral fingerprints (DWT sub-bands, DCT statistics, DFT spectrum).
    """
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, feature_dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat_stem = self.stem(x)
        feat_l1 = self.layer1(feat_stem)
        feat_l2 = self.layer2(feat_l1)
        feat_map = self.layer3(feat_l2)
        vector = self.pool(feat_map).flatten(1)
        return vector, feat_map


class TwoStageCrossAttention(nn.Module):
    """
    Two-Stage Cross-Attention Module for fusing spatial context with high-frequency anomalies.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.spatial_query = nn.Linear(dim, dim)
        self.spatial_key = nn.Linear(dim, dim)
        self.spatial_val = nn.Linear(dim, dim)

        self.freq_query = nn.Linear(dim, dim)
        self.freq_key = nn.Linear(dim, dim)
        self.freq_val = nn.Linear(dim, dim)

        self.scale = dim ** -0.5
        self.out_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_spatial, f_freq):
        # f_spatial: (B, D), f_freq: (B, D)
        # Stage 1: Spatial attending to Frequency
        q_s = self.spatial_query(f_spatial)
        k_f = self.freq_key(f_freq)
        v_f = self.freq_val(f_freq)
        attn_sf = torch.sigmoid(torch.sum(q_s * k_f, dim=-1, keepdim=True) * self.scale)
        s_enhanced = f_spatial + attn_sf * v_f

        # Stage 2: Frequency attending to Spatial
        q_f = self.freq_query(f_freq)
        k_s = self.spatial_key(f_spatial)
        v_s = self.spatial_val(f_spatial)
        attn_fs = torch.sigmoid(torch.sum(q_f * k_s, dim=-1, keepdim=True) * self.scale)
        f_enhanced = f_freq + attn_fs * v_s

        combined = torch.cat([s_enhanced, f_enhanced], dim=1)
        fused = self.norm(self.out_proj(combined))
        return fused, attn_sf, attn_fs


class DualBranchSpatioFrequencyNet(nn.Module):
    """
    Complete Forensic Classifier implementing joint spatio-frequency representation learning.
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.spatial_stream = SpatialBranch(in_channels=3, feature_dim=feature_dim)
        self.frequency_stream = FrequencyBranch(in_channels=4, feature_dim=feature_dim)
        self.fusion = TwoStageCrossAttention(dim=feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x_rgb, x_freq):
        vec_s, map_s = self.spatial_stream(x_rgb)
        vec_f, map_f = self.frequency_stream(x_freq)
        fused_vec, attn_sf, attn_fs = self.fusion(vec_s, vec_f)
        logits = self.classifier(fused_vec)
        prob = torch.sigmoid(logits)
        return prob, map_s, map_f, fused_vec
