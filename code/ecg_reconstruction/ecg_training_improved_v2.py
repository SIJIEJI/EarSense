#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG training + evaluation + improved scatter plotting (calibrated)
Fixed: process_data() now handles numpy/Series robustly (no .to_numpy() on ndarray).
"""

import os
import math
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

def moving_average_np(x, k=9):
    k = max(1, int(k))
    if k % 2 == 0: k += 1
    w = np.ones(k) / k
    return np.convolve(x, w, mode='same')

def linear_calibrate(x, y):
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
        if self.augment and np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            signal *= scale; target *= scale
        s_mu, s_sd = signal.mean(), signal.std()
        t_mu, t_sd = target.mean(), target.std()
        s_sd = s_sd if s_sd > 1e-8 else 1.0
        t_sd = t_sd if t_sd > 1e-8 else 1.0
        signal = (signal - s_mu) / s_sd
        target = (target - t_mu) / t_sd
        return torch.from_numpy(signal).float(), torch.from_numpy(target).float()

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
        Q = self.query(x).view(B, -1, L).permute(0, 2, 1)
        K = self.key(x).view(B, -1, L)
        V = self.value(x).view(B, -1, L)
        scale = 1.0 / math.sqrt(Q.shape[-1] + 1e-8)
        attn = torch.softmax(torch.bmm(Q, K) * scale, dim=-1)
        out  = torch.bmm(V, attn.permute(0, 2, 1))
        return self.gamma * out + x

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.InstanceNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.InstanceNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            AttentionBlock(128),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.InstanceNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)

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
                p = pool(pred.transpose(1,2)).transpose(1,2)
                t = pool(target.transpose(1,2)).transpose(1,2)
            else:
                p, t = pred, target
            grad = torch.abs(torch.diff(t, dim=1, prepend=t[:, :1, :]))
            denom = torch.amax(grad, dim=1, keepdim=True) + 1e-8
            wt = grad / denom + 0.2
            loss = torch.mean(wt * torch.abs(p - t))
            total = total + w * loss
        return total

def process_data(file_path, segment_length=2000, overlap=0.5):
    df = pd.read_csv(file_path)
    cols = df.columns.str.lower()
    def pick(names, default_idx):
        for n in names:
            if n in cols:
                return df.iloc[:, list(cols).index(n)]
        return df.iloc[:, default_idx]
    # pick returns a Series; pd.to_numeric keeps array-like; ensure ndarray with np.asarray
    time  = np.asarray(pd.to_numeric(pick(["time"], 0), errors='coerce'))
    ecg   = np.asarray(pd.to_numeric(pick(["ecg","target","gt","ground_truth"], 1), errors='coerce'))
    mixed = np.asarray(pd.to_numeric(pick(["mixed","mix","observed","input"], 2), errors='coerce'))

    # drop NaNs and align
    L = min(len(time), len(ecg), len(mixed))
    time, ecg, mixed = time[:L], ecg[:L], mixed[:L]
    mask = np.isfinite(time) & np.isfinite(ecg) & np.isfinite(mixed)
    time, ecg, mixed = time[mask], ecg[mask], mixed[mask]

    step = max(1, int(segment_length*(1-overlap)))
    xs, ys = [], []
    for s in range(0, len(ecg) - segment_length + 1, step):
        xs.append(mixed[s:s+segment_length].copy())
        ys.append(ecg[s:s+segment_length].copy())
    return np.array(xs), np.array(ys)

def plot_scatter_enhanced(gt, pr, out_png):
    gt = np.asarray(gt).ravel()
    pr = np.asarray(pr).ravel()
    gt_f = moving_average_np(gt, 11)
    pr_f = moving_average_np(pr, 11)
    L = min(len(gt_f), len(pr_f))
    gt_f = gt_f[:L]; pr_f = pr_f[:L]
    # linear calib y≈a*x+b
    xm, ym = gt_f.mean(), pr_f.mean()
    varx = (gt_f - xm).var() + 1e-12
    a = ((gt_f - xm)*(pr_f - ym)).mean() / varx
    b = ym - a*xm
    r = float(np.dot(gt_f-gt_f.mean(), pr_f-pr_f.mean()) / (np.linalg.norm(gt_f-gt_f.mean())*np.linalg.norm(pr_f-pr_f.mean()) + 1e-12))
    mse = float(np.mean((gt_f - pr_f)**2))
    mse_cal = float(np.mean((gt_f - (a*gt_f + b))**2))

    plt.figure(figsize=(6,6))
    plt.scatter(gt_f, pr_f, s=6, alpha=0.5, linewidths=0)
    lo = float(min(gt_f.min(), pr_f.min()))
    hi = float(max(gt_f.max(), pr_f.max()))
    #plt.plot([lo, hi], [lo, hi])
    xline = np.linspace(lo, hi, 100)
    plt.plot(xline, a*xline + b, linestyle='--')
    plt.xlabel("Ground Truth"); plt.ylabel("Reconstructed")
    plt.title("Scatter: GT vs Reconstructed")
    txt = f"r = {r:.3f} | MSE = {mse:.4f}\nCalibrated: a={a:.3f}, b={b:.3f}, MSE'={mse_cal:.4f}"
    plt.gcf().text(0.02, 0.02, txt)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main():
    BATCH_SIZE = 32; LR = 1e-3; EPOCHS = 30
    SEG_LEN = 500; OVERLAP = 0.5
    for p in ["1.csv", "2.csv"]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing data file: {p}")
    X1, y1 = process_data('1.csv', segment_length=SEG_LEN, overlap=OVERLAP)
    X2, y2 = process_data('2.csv', segment_length=SEG_LEN, overlap=OVERLAP)
    X = np.vstack([X1, X2]); y = np.vstack([y1, y2])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(ECGDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val,   augment=False), batch_size=BATCH_SIZE, shuffle=False)
    model = ECGNet().to(device)
    criterion = MultiScalePeakLoss()
    optimzr = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=max(10, EPOCHS//3), eta_min=1e-6)
    best = float('inf')
    for ep in range(EPOCHS):
        model.train(); tr = 0.0
        for xb, yb in train_loader:
            xb = xb.unsqueeze(-1).to(device); yb = yb.unsqueeze(-1).to(device)
            optimzr.zero_grad(); out = model(xb)
            loss = criterion(out, yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimzr.step(); tr += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.unsqueeze(-1).to(device); yb = yb.unsqueeze(-1).to(device)
                out = model(xb); vl += criterion(out, yb).item()
        sched.step()
        trm = tr / max(1, len(train_loader)); vlm = vl / max(1, len(val_loader))
        if vlm < best: best = vlm; torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch {ep+1}/{EPOCHS} | Train {trm:.4f} | Val {vlm:.4f}")
    model.load_state_dict(torch.load("best_model.pth", map_location=device)); model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val[:8]).float().unsqueeze(-1).to(device)
        yb = torch.from_numpy(y_val[:8]).float().unsqueeze(-1).to(device)
        pr = model(xb).cpu().numpy().squeeze(-1); gt = yb.cpu().numpy().squeeze(-1)
    plot_scatter_enhanced(gt.reshape(-1), pr.reshape(-1), out_png="scatter_calibrated.png")
    print("[OK] Saved: scatter_calibrated.png")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(str(e))
