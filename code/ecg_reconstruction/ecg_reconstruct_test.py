#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG long-sequence reconstruction & visualization

Requirements:
- Python 3.9+
- torch, numpy, pandas, matplotlib, scipy (optional)
- A trained model checkpoint 'best_model.pth'
- A CSV with columns: [time, ecg (ground truth optional), mixed]

Usage examples:
1) With ground truth column available:
   python ecg_reconstruct_test.py --csv test.csv --ckpt best_model.pth --seg_len 2000 --overlap 0.5

2) Without ground truth (no 'ecg' column in CSV):
   python ecg_reconstruct_test.py --csv test_no_gt.csv --ckpt best_model.pth --seg_len 2000 --overlap 0.5 --no_gt

Outputs:
- A plot window with the reconstructed ECG overlaid (and GT if available)
- Optional: saves numpy arrays with --save_npy
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = RELEASE_ROOT / "data" / "ecg_reconstruction" / "test_data.csv"
DEFAULT_CKPT = RELEASE_ROOT / "models" / "ecg_reconstruction" / "best_model.pth"

# -----------------------------
# Model definition (must match training)
# -----------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, C, L = x.size()
        Q = self.query(x).view(b, -1, L).permute(0, 2, 1)  # [B, L, C//8]
        K = self.key(x).view(b, -1, L)                     # [B, C//8, L]
        V = self.value(x).view(b, -1, L)                   # [B, C, L]
        attn = torch.softmax(torch.bmm(Q, K) / np.sqrt(max(C // 8, 1)), dim=-1)  # [B, L, L]
        out = torch.bmm(V, attn.permute(0, 2, 1))          # [B, C, L]
        return self.gamma * out + x

class ECGNet(nn.Module):
    def __init__(self):
        super(ECGNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            AttentionBlock(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def forward(self, x):
        # x: [B, L, 1]
        x = x.permute(0, 2, 1)  # [B, 1, L]
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)  # [B, L, 1]

# -----------------------------
# Utils
# -----------------------------

def slice_signal(sig: np.ndarray, window: int, step: int):
    """Return overlapping windows [num_windows, window]."""
    if len(sig) < window:
        # zero-pad if too short
        pad = window - len(sig)
        sig = np.pad(sig, (0, pad))
    idxs = range(0, max(len(sig) - window + 1, 1), step)
    return np.stack([sig[i:i+window] for i in idxs], axis=0), np.array(list(idxs))

def overlap_add(windows: np.ndarray, starts: np.ndarray, total_len: int, window_fn: str = "hann"):
    """
    Reconstruct 1D sequence via weighted overlap-add.
    windows: [N, W]
    starts:  [N] starting indices
    total_len: length of the output signal
    window_fn: one of ['hann', 'tri', 'ones']
    """
    W = windows.shape[1]
    if window_fn == "hann":
        w = np.hanning(W)
    elif window_fn == "tri":
        w = 1.0 - np.abs(np.linspace(-1, 1, W))
    else:
        w = np.ones(W, dtype=np.float64)
    acc = np.zeros(total_len, dtype=np.float64)
    weight = np.zeros(total_len, dtype=np.float64)
    for seg, s in zip(windows, starts):
        e = min(s + W, total_len)
        seg_len = e - s
        acc[s:e] += seg[:seg_len] * w[:seg_len]
        weight[s:e] += w[:seg_len]
    weight[weight == 0] = 1.0
    return acc / weight

def gaussian_smooth(x: np.ndarray, k: int = 31):
    g = np.exp(-np.linspace(-3, 3, k) ** 2)
    g /= g.sum()
    return np.convolve(x, g, mode="same")

def reconstruct_full_ecg(mixed: np.ndarray,
                         model: nn.Module,
                         device: torch.device,
                         segment_length: int = 2000,
                         overlap: float = 0.5,
                         batch_size: int = 64):
    """Run the model over the entire mixed signal with overlap-add fusion."""
    step = int(segment_length * (1 - overlap))
    step = max(step, 1)
    # slice
    windows, starts = slice_signal(mixed, segment_length, step)
    # inference in batches
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i+batch_size]
            t = torch.from_numpy(chunk).float().unsqueeze(-1).to(device)  # [B, W, 1]
            y = model(t).squeeze(-1).cpu().numpy()                        # [B, W]
            preds.append(y)
    preds = np.concatenate(preds, axis=0)
    # overlap-add
    out = overlap_add(preds, starts, total_len=len(mixed), window_fn="hann")
    # light smoothing to reduce seams
    out = gaussian_smooth(out, k=31)
    return out

def load_csv(csv_path: str, no_gt: bool = False):
    df = pd.read_csv(csv_path)
    # try to infer columns
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    time_col = col_map.get("time", df.columns[0])
    mixed_col = None
    for key in ["mixed", "mix", "observed", "input"]:
        if key in cols:
            mixed_col = col_map[key]
            break
    if mixed_col is None:
        mixed_col = df.columns[2] if len(df.columns) >= 3 else df.columns[1]
    if no_gt:
        ecg = None
    else:
        ecg = None
        for key in ["ecg", "target", "gt", "ground_truth"]:
            if key in cols:
                ecg = df[col_map[key]].to_numpy(dtype=float)
                break
        if ecg is None and len(df.columns) >= 2:
            # best-effort guess: assume 2nd column is ecg
            ecg = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    mixed = pd.to_numeric(df[mixed_col], errors="coerce").to_numpy()
    # drop NaNs consistently
    n = min(len(t), len(mixed), len(ecg) if ecg is not None else len(mixed))
    t = t[:n]
    mixed = mixed[:n]
    if ecg is not None:
        ecg = ecg[:n]
    return t, mixed, ecg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to CSV with [time, ecg?, mixed]")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seg_len", type=int, default=2000)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_gt", action="store_true", help="Set if CSV has no ground-truth ECG column")
    parser.add_argument("--save_npy", action="store_true", help="Save reconstructed numpy arrays")
    parser.add_argument("--out_prefix", type=str, default="recon")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load data
    t, mixed, ecg = load_csv(args.csv, no_gt=args.no_gt)

    # Build model & load weights
    model = ECGNet().to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # Reconstruct long signal
    recon = reconstruct_full_ecg(
        mixed=mixed,
        model=model,
        device=device,
        segment_length=args.seg_len,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )

    # Metrics (if GT available)
    if ecg is not None:
        # align lengths
        L = min(len(ecg), len(recon))
        gt = ecg[:L]
        pr = recon[:L]
        # MSE
        mse = float(np.mean((gt - pr) ** 2))
        # Cosine similarity
        denom = (np.linalg.norm(gt) * np.linalg.norm(pr)) + 1e-12
        cos = float(np.dot(gt, pr) / denom)
        print(f"[Metrics] MSE={mse:.6f} | CosSim={cos:.6f}")
    else:
        gt = None
        pr = recon

    # Visualization
    plt.figure(figsize=(14, 6))
    # Downsample for faster rendering if extremely long
    max_points = 200000
    def maybe_down(x):
        if len(x) <= max_points:
            return x
        step = int(np.ceil(len(x) / max_points))
        return x[::step]

    tt = maybe_down(t)
    mx = maybe_down(mixed)
    rc = maybe_down(recon)
    if gt is not None:
        gg = maybe_down(gt)
        # if time length mismatched after downsampling, rebuild a uniform time vector
        if len(tt) != len(gg):
            tt = np.linspace(t[0], t[0] + (len(gg)-1)*(t[1]-t[0] if len(t)>1 else 1.0), num=len(gg))

    plt.plot(tt[:len(mx)], mx, label="Mixed/Input", alpha=0.6)
    if gt is not None:
        plt.plot(tt[:len(gg)], gg, label="Ground Truth ECG", linewidth=1.0)
    plt.plot(tt[:len(rc)], rc, label="Reconstructed ECG", linewidth=1.2)
    plt.title("Long-Sequence ECG Reconstruction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save arrays if requested
    if args.save_npy:
        np.save(f"{args.out_prefix}_time.npy", t)
        np.save(f"{args.out_prefix}_mixed.npy", mixed)
        np.save(f"{args.out_prefix}_recon.npy", recon)
        if ecg is not None:
            np.save(f"{args.out_prefix}_ecg.npy", ecg)
        print(f"Saved NPY arrays with prefix: {args.out_prefix}_*.npy")

if __name__ == "__main__":
    main()
