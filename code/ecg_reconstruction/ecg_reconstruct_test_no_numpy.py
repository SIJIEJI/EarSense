#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG long-sequence reconstruction & visualization (NumPy-free)

- No dependency on numpy/pandas. Uses only: torch, csv, matplotlib.
- Reads CSV with columns: time, ecg (optional), mixed (flexible column names).
- Does overlap-add with Hann window and light Gaussian smoothing (torch only).
- Computes MSE & cosine similarity if GT exists.
"""

import argparse
import os
# Mitigate Windows OpenMP duplicate runtime crash when multiple runtimes are present.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import csv
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Model definition (match training)
# -----------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c_qk = max(channels // 8, 1)
        self.query = nn.Conv1d(channels, c_qk, 1)
        self.key   = nn.Conv1d(channels, c_qk, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        Q = self.query(x).view(B, -1, L).permute(0, 2, 1)        # [B, L, Cq]
        K = self.key(x).view(B, -1, L)                           # [B, Cq, L]
        V = self.value(x).view(B, -1, L)                         # [B, C,  L]
        scale = 1.0 / math.sqrt(max(K.shape[1], 1))
        attn = torch.softmax(torch.bmm(Q, K) * scale, dim=-1)    # [B, L, L]
        out = torch.bmm(V, attn.permute(0, 2, 1))                # [B, C, L]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 1]
        x = x.permute(0, 2, 1)  # [B, 1, L]
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)  # [B, L, 1]

# -----------------------------
# Utils (torch only)
# -----------------------------

def to_tensor(lst: List[float]) -> torch.Tensor:
    return torch.tensor(lst, dtype=torch.float32)

def slice_signal(sig: torch.Tensor, window: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overlapping windows on 1D signal.
    sig: [L]
    return: windows [N, W], starts [N]
    """
    L = sig.numel()
    if L < window:
        pad = window - L
        sig = torch.cat([sig, torch.zeros(pad, dtype=sig.dtype)])
        L = window
    starts = list(range(0, max(L - window + 1, 1), step))
    ws = []
    for s in starts:
        ws.append(sig[s:s+window])
    windows = torch.stack(ws, dim=0) if ws else sig.unsqueeze(0)
    return windows, torch.tensor(starts, dtype=torch.long)

def hann_window(W: int, device: torch.device) -> torch.Tensor:
    # match numpy.hanning behavior: 0 at ends
    n = torch.arange(W, device=device, dtype=torch.float32)
    if W <= 1:
        return torch.ones(W, device=device, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (W - 1))

def overlap_add(windows: torch.Tensor, starts: torch.Tensor, total_len: int, device: torch.device) -> torch.Tensor:
    """
    Weighted overlap-add with Hann window.
    windows: [N, W]
    starts:  [N]
    """
    N, W = windows.shape
    acc = torch.zeros(total_len, device=device, dtype=torch.float32)
    wts = torch.zeros(total_len, device=device, dtype=torch.float32)
    w = hann_window(W, device=device)
    for i in range(N):
        s = int(starts[i].item())
        e = min(s + W, total_len)
        seg_len = e - s
        acc[s:e] += windows[i, :seg_len] * w[:seg_len]
        wts[s:e] += w[:seg_len]
    wts = torch.where(wts == 0, torch.ones_like(wts), wts)
    return acc / wts

def gaussian_kernel1d(k: int = 31, sigma: float = 5.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    # Create a 1D Gaussian kernel normalized to sum=1
    x = torch.linspace(-3, 3, steps=k, device=device)
    g = torch.exp(-x * x / (2 * (sigma**2 / 9)))  # shape similar to e^{-x^2}
    g = g / g.sum()
    return g

def gaussian_smooth(x: torch.Tensor, k: int = 31, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if k < 3: 
        return x
    g = gaussian_kernel1d(k=k, device=device).view(1, 1, -1)           # [1,1,k]
    y = x.view(1, 1, -1)
    pad = (k - 1) // 2
    y = F.pad(y, (pad, pad), mode='reflect')
    y = F.conv1d(y, g)
    return y.view(-1)

def reconstruct_full_ecg(mixed: torch.Tensor,
                         model: nn.Module,
                         device: torch.device,
                         segment_length: int = 2000,
                         overlap: float = 0.5,
                         batch_size: int = 64) -> torch.Tensor:
    step = max(int(segment_length * (1 - overlap)), 1)
    windows, starts = slice_signal(mixed, segment_length, step)  # [N,W], [N]
    N = windows.shape[0]

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            chunk = windows[i:i+batch_size].to(device)             # [B, W]
            t = chunk.unsqueeze(-1)                                # [B, W, 1]
            y = model(t).squeeze(-1)                               # [B, W]
            preds.append(y.cpu())
    preds = torch.cat(preds, dim=0)                                # [N, W]
    out = overlap_add(preds.to(device), starts.to(device), total_len=mixed.numel(), device=device)
    out = gaussian_smooth(out, k=31, device=device)
    return out

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def load_csv(csv_path: str, no_gt: bool = False) -> Tuple[List[float], List[float], Optional[List[float]]]:
    """
    Very lightweight CSV reader. Tries to infer columns by name:
    time, ecg/target/gt/ground_truth, mixed/mix/observed/input
    Fallbacks to the first/second/third columns if not found.
    """
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV.")

    header = rows[0]
    has_header = any(not safe_float(x) for x in header)
    data_rows = rows[1:] if has_header else rows

    def col_index(names):
        if not has_header:
            return None
        lower = [h.strip().lower() for h in header]
        for name in names:
            if name in lower:
                return lower.index(name)
        return None

    idx_time = col_index(["time"])
    idx_mixed = col_index(["mixed", "mix", "observed", "input"])
    idx_ecg = None if no_gt else col_index(["ecg", "target", "gt", "ground_truth"])

    # Fallbacks by position
    if idx_time is None and len(header) >= 1: idx_time = 0
    if idx_ecg is None and (not no_gt) and len(header) >= 2: idx_ecg = 1
    if idx_mixed is None and len(header) >= 3: idx_mixed = 2
    if idx_mixed is None and len(header) >= 2: idx_mixed = 1  # minimal fallback

    times, mixed, ecg = [], [], []

    for r in data_rows:
        if len(r) == 0:
            continue
        t = safe_float(r[idx_time]) if idx_time is not None and idx_time < len(r) else None
        m = safe_float(r[idx_mixed]) if idx_mixed is not None and idx_mixed < len(r) else None
        e = safe_float(r[idx_ecg]) if (idx_ecg is not None and idx_ecg < len(r)) else None
        if t is None or m is None:
            continue
        times.append(t)
        mixed.append(m)
        if (idx_ecg is not None) and (e is not None):
            ecg.append(e)

    if not times or not mixed:
        raise ValueError("Failed to parse CSV columns for time/mixed.")

    if no_gt or (idx_ecg is None) or (len(ecg) == 0):
        ecg_out = None
    else:
        # align lengths
        L = min(len(times), len(mixed), len(ecg))
        times = times[:L]
        mixed = mixed[:L]
        ecg_out = ecg[:L]

    return times, mixed, ecg_out

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best_model.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seg_len", type=int, default=2000)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="recon")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data via csv module
    t_list, mixed_list, ecg_list = load_csv('1.csv', no_gt=args.no_gt)

    # Tensors
    t = to_tensor(t_list)                 # [L]
    mixed = to_tensor(mixed_list)         # [L]
    ecg = to_tensor(ecg_list) if ecg_list is not None else None

    # Build model & load weights
    model = ECGNet().to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)

    # Reconstruct
    recon = reconstruct_full_ecg(
        mixed=mixed.to(device),
        model=model,
        device=device,
        segment_length=args.seg_len,
        overlap=args.overlap,
        batch_size=args.batch_size,
    ).cpu()

    # Metrics
    if ecg is not None and ecg.numel() > 0:
        L = min(ecg.numel(), recon.numel())
        gt = ecg[:L]
        pr = recon[:L]
        mse = torch.mean((gt - pr)**2).item()
        cos = torch.dot(gt, pr).item() / (gt.norm().item() * pr.norm().item() + 1e-12)
        print(f"[Metrics] MSE={mse:.6f} | CosSim={cos:.6f}")

    # Visualization (lists; avoid numpy)
    #max_points = 200000
    max_points = 500000
    def downsample_list(a: List[float]) -> List[float]:
        if len(a) <= max_points:
            return a
        step = max(1, len(a) // max_points)
        return a[::step]

    tt = downsample_list(t.tolist())
    mx = downsample_list(mixed.tolist())
    rc = downsample_list(recon.tolist())

    plt.figure(figsize=(14, 6))
    plt.plot(tt[:len(mx)], mx, label="EarEEG", alpha=0.6)
    if ecg is not None and ecg.numel() > 0:
        gg = downsample_list(ecg.tolist())
        if len(gg) != len(tt):
            # fallback uniform time if needed
            dt = (t[1] - t[0]).item() if t.numel() > 1 else 1.0
            tt = [tt[0] + i * dt for i in range(len(gg))]
        plt.plot(tt[:len(gg)], gg, label="Ground Truth ECG", linewidth=1.5)
    plt.plot(tt[:len(rc)], rc, label="Reconstructed ECG", linewidth=1.5)
    plt.title("Long-Sequence ECG Reconstruction (NumPy-free)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #保存重建结果 recon.npy
    np.save("recon.npy", recon.cpu().numpy())
    #将用于画图的数据保存为csv文件 recon.csv
    with open("recon_full.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "mixed", "reconstructed"] + (["ground_truth"] if ecg is not None and ecg.numel() > 0 else []))
        for i in range(len(tt)):
            row = [tt[i], mx[i], rc[i]]
            if ecg is not None and ecg.numel() > 0 and i < len(ecg):
                row.append(ecg[i].item())
            writer.writerow(row)

if __name__ == "__main__":
    main()
