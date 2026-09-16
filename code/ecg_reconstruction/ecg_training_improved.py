#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG training + evaluation + improved scatter plotting (calibrated)
- Cleaner model/loss fixes
- Consistent preprocessing
- Publication-style scatter (y=x, OLS regression, r/MSE annotations)
- Optional linear post-calibration to reduce bias and range compression
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Utilities
# ------------------------
def moving_average_np(x, k=9):
    k = max(1, int(k))
    if k % 2 == 0: k += 1
    w = np.ones(k) / k
    return np.convolve(x, w, mode='same')

def linear_calibrate(x, y):
    """Fit y ≈ a*x + b (least squares). Returns a, b."""
    x = np.asarray(x); y = np.asarray(y)
    xm, ym = x.mean(), y.mean()
    varx = (x - xm).var()
    if varx < 1e-12:
        return 1.0, 0.0
    a = ((x - xm) * (y - ym)).mean() / varx
    b = ym - a * xm
    return float(a), float(b)

def cosine_similarity(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def safe_zscore(x):
    x = np.asarray(x)
    mu, sd = x.mean(), x.std()
    if sd < 1e-8: sd = 1.0
    return (x - mu) / sd

# ------------------------
# Dataset
# ------------------------
class ECGDataset(Dataset):
    def __init__(self, signals, targets, augment=True):
        self.signals = signals
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        target = self.targets[idx].copy()

        # amplitude jitter (same scale for input/target)
        if self.augment and np.random.rand() < 0.5:
            orig_len = len(signal)
            scale = np.random.uniform(0.8, 1.2)
            signal *= scale
            target *= scale

            warp = np.random.uniform(0.9, 1.1)
            warped_len = max(2, int(round(orig_len * warp)))
            old_x = np.linspace(0.0, 1.0, orig_len)
            new_x = np.linspace(0.0, 1.0, warped_len)
            signal = np.interp(new_x, old_x, signal)
            target = np.interp(new_x, old_x, target)

            if warped_len >= orig_len:
                start = np.random.randint(0, warped_len - orig_len + 1)
                signal = signal[start:start + orig_len]
                target = target[start:start + orig_len]
            else:
                pad = orig_len - warped_len
                left = np.random.randint(0, pad + 1)
                right = pad - left
                signal = np.pad(signal, (left, right), mode="edge")
                target = np.pad(target, (left, right), mode="edge")

        # standardize per-segment (helps training stability)
        s_mu, s_sd = signal.mean(), signal.std()
        t_mu, t_sd = target.mean(), target.std()
        s_sd = s_sd if s_sd > 1e-8 else 1.0
        t_sd = t_sd if t_sd > 1e-8 else 1.0
        signal = (signal - s_mu) / s_sd
        target = (target - t_mu) / t_sd

        return torch.from_numpy(signal).float(), torch.from_numpy(target).float()

# ------------------------
# Model
# ------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c_qk = max(1, channels // 8)
        self.query = nn.Conv1d(channels, c_qk, 1)
        self.key   = nn.Conv1d(channels, c_qk, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.shape
        Q = self.query(x).view(B, -1, L).permute(0, 2, 1)  # [B, L, Cq]
        K = self.key(x).view(B, -1, L)                     # [B, Cq, L]
        V = self.value(x).view(B, -1, L)                   # [B, C,  L]
        scale = 1.0 / math.sqrt(Q.shape[-1] + 1e-8)
        attn = torch.softmax(torch.bmm(Q, K) * scale, dim=-1)  # [B, L, L]
        out  = torch.bmm(V, attn.permute(0, 2, 1))             # [B, C, L]
        return self.gamma * out + x

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            AttentionBlock(128),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)      # [B, 1, L]
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)   # [B, L, 1]

# ------------------------
# Loss (fix weighted L1)
# ------------------------
class MultiScalePeakLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = [1, 2, 4]
        self.weights = [0.5, 0.3, 0.2]

    def forward(self, pred, target):
        total = 0.0
        for scale, w in zip(self.scales, self.weights):
            if scale > 1:
                pool = nn.AvgPool1d(scale)
                p = pool(pred.transpose(1,2)).transpose(1,2)  # [B,L/scale,1]
                t = pool(target.transpose(1,2)).transpose(1,2)
            else:
                p, t = pred, target
            # peak-sensitive weight via abs gradient
            grad = torch.abs(torch.diff(t, dim=1, prepend=t[:, :1, :]))
            denom = torch.amax(grad, dim=1, keepdim=True) + 1e-8
            wt = grad / denom + 0.2  # [B,L,1]
            loss = torch.mean(wt * torch.abs(p - t))
            total = total + w * loss
        return total

class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft(pred.squeeze(-1), dim=1)
        target_fft = torch.fft.rfft(target.squeeze(-1), dim=1)
        diff = pred_fft - target_fft
        freq_dist = diff.real.pow(2) + diff.imag.pow(2)

        with torch.no_grad():
            weight = freq_dist.pow(self.alpha)
            weight = weight / (weight.mean(dim=1, keepdim=True) + self.eps)

        return torch.mean(weight * freq_dist)

class ECGReconstructionLoss(nn.Module):
    def __init__(self, peak_weight=0.99, focal_frequency_weight=0.01):
        super().__init__()
        self.peak_weight = peak_weight
        self.focal_frequency_weight = focal_frequency_weight
        self.peak_loss = MultiScalePeakLoss()
        self.focal_frequency_loss = FocalFrequencyLoss()

    def forward(self, pred, target):
        peak = self.peak_loss(pred, target)
        focal_frequency = self.focal_frequency_loss(pred, target)
        return self.peak_weight * peak + self.focal_frequency_weight * focal_frequency

# ------------------------
# Data processing
# ------------------------
def process_data(file_path, segment_length=2000, overlap=0.5):
    df = pd.read_csv(file_path)
    # generic column handling
    cols = df.columns.str.lower()
    def pick(names, default_idx):
        # Return a pandas Series; callers handle .to_numpy() after pd.to_numeric
        for n in names:
            if n in cols:
                return df.iloc[:, list(cols).index(n)]
        return df.iloc[:, default_idx]
    time  = pd.to_numeric(pick(["time"], 0), errors='coerce').to_numpy()
    ecg   = pd.to_numeric(pick(["ecg","target","gt","ground_truth"], 1), errors='coerce').to_numpy()
    mixed = pd.to_numeric(pick(["mixed","mix","observed","input"], 2), errors='coerce').to_numpy()

    # drop NaNs and align length
    L = min(len(time), len(ecg), len(mixed))
    ecg = np.asarray(ecg[:L], dtype=float)
    mixed = np.asarray(mixed[:L], dtype=float)

    # slicing
    step = max(1, int(segment_length*(1-overlap)))
    xs, ys = [], []
    for s in range(0, L - segment_length + 1, step):
        xs.append(mixed[s:s+segment_length].copy())
        ys.append(ecg[s:s+segment_length].copy())
    return np.array(xs), np.array(ys)

# ------------------------
# Plot helpers
# ------------------------
def plot_scatter_enhanced(gt, pr, out_png):
    """Publication-style scatter with calibration and annotations."""
    gt = np.asarray(gt).ravel()
    pr = np.asarray(pr).ravel()
    # Smooth both sides slightly & equalize length
    gt_f = moving_average_np(gt, 11)
    pr_f = moving_average_np(pr, 11)
    L = min(len(gt_f), len(pr_f))
    gt_f = gt_f[:L]; pr_f = pr_f[:L]

    # Optional linear calibration
    a, b = linear_calibrate(gt_f, pr_f)
    pr_cal = a * gt_f + b

    r = cosine_similarity(gt_f - gt_f.mean(), pr_f - pr_f.mean())
    mse = float(np.mean((gt_f - pr_f)**2))
    mse_cal = float(np.mean((gt_f - pr_cal)**2))

    # Scatter
    plt.figure(figsize=(6,6))
    plt.scatter(gt_f, pr_f, s=6, alpha=0.5, linewidths=0)
    # y = x
    lo = float(min(gt_f.min(), pr_f.min()))
    hi = float(max(gt_f.max(), pr_f.max()))
    plt.plot([lo, hi], [lo, hi])
    # regression line (calibrated)
    xline = np.linspace(lo, hi, 100)
    plt.plot(xline, a*xline + b, linestyle='--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Reconstructed")
    plt.title("Scatter: GT vs Reconstructed")
    # annotations
    txt = f"r = {r:.3f} | MSE = {mse:.4f}\nCalibrated: a={a:.3f}, b={b:.3f}, MSE'={mse_cal:.4f}"
    plt.gcf().text(0.02, 0.02, txt)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# ------------------------
# Main (training optional; plotting supported after eval)
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csvs", nargs="+", required=True, help="Paired ECG training CSV files: time, reference ECG, in-ear input.")
    parser.add_argument("--out_ckpt", default="best_model.pth", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seg_len", type=int, default=2000)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    # Params
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    SEG_LEN = args.seg_len
    OVERLAP = args.overlap

    # Load paired training data.
    for p in args.train_csvs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing data file: {p}")

    X_list, y_list = [], []
    for p in args.train_csvs:
        Xi, yi = process_data(p, segment_length=SEG_LEN, overlap=OVERLAP)
        X_list.append(Xi)
        y_list.append(yi)
    X = np.vstack(X_list)
    y = np.vstack(y_list)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(ECGDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val,   augment=False), batch_size=BATCH_SIZE, shuffle=False)

    model = ECGNet().to(device)
    criterion = ECGReconstructionLoss(peak_weight=0.99, focal_frequency_weight=0.01)
    optimzr = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=max(10, EPOCHS//3), eta_min=1e-6)

    best = float('inf')
    for ep in range(EPOCHS):
        model.train(); tr = 0.0
        for xb, yb in train_loader:
            xb = xb.unsqueeze(-1).to(device)
            yb = yb.unsqueeze(-1).to(device)
            optimzr.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimzr.step()
            tr += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.unsqueeze(-1).to(device)
                yb = yb.unsqueeze(-1).to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vl += loss.item()
        sched.step()
        trm = tr / max(1, len(train_loader))
        vlm = vl / max(1, len(val_loader))
        if vlm < best:
            best = vlm
            torch.save(model.state_dict(), args.out_ckpt)
        print(f"Epoch {ep+1}/{EPOCHS} | Train {trm:.4f} | Val {vlm:.4f}")

    # Eval a batch for plotting
    model.load_state_dict(torch.load(args.out_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val[:8]).float().unsqueeze(-1).to(device)  # [N,L,1]
        yb = torch.from_numpy(y_val[:8]).float().unsqueeze(-1).to(device)
        pr = model(xb).cpu().numpy().squeeze(-1)  # [N,L]
        gt = yb.cpu().numpy().squeeze(-1)

    # Build vectors for scatter (concatenate several segments)
    pr_vec = pr.reshape(-1)
    gt_vec = gt.reshape(-1)
    plot_scatter_enhanced(gt_vec, pr_vec, out_png="scatter_calibrated.png")

    print("[OK] Saved: scatter_calibrated.png")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(str(e))
