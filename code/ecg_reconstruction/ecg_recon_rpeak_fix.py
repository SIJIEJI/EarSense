#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG reconstruction evaluation toolkit (robust R-peak version)

Fixes & additions vs previous:
- Robust handling for time irregularities; --fs overrides sampling rate detection.
- Guards against empty/degenerate arrays and zero-variance signals.
- Safer R-peak detector (threshold floors, min/max bounds, device-safe ops).
- R-peak overlay uses each signal's own amplitudes to avoid index mismatch.
- Better error messages & optional debug dumps (--debug_rpeaks).
"""

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = RELEASE_ROOT / "data" / "ecg_reconstruction" / "test_data.csv"

# -----------------------------
# Model definition (placeholder; match your training if needed)
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
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)

# -----------------------------
# I/O and helpers
# -----------------------------

def to_tensor(lst: List[float]) -> torch.Tensor:
    return torch.tensor(lst, dtype=torch.float32)

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def load_csv(csv_path: str, no_gt: bool = False) -> Tuple[List[float], List[float], Optional[List[float]]]:
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

    if idx_time is None and len(header) >= 1: idx_time = 0
    if idx_ecg is None and (not no_gt) and len(header) >= 2: idx_ecg = 1
    if idx_mixed is None and len(header) >= 3: idx_mixed = 2
    if idx_mixed is None and len(header) >= 2: idx_mixed = 1

    times, mixed, ecg = [], [], []
    for r in data_rows:
        if len(r) == 0:
            continue
        t = safe_float(r[idx_time]) if idx_time is not None and idx_time < len(r) else None
        m = safe_float(r[idx_mixed]) if idx_mixed is not None and idx_mixed < len(r) else None
        e = safe_float(r[idx_ecg]) if (idx_ecg is not None and idx_ecg < len(r)) else None
        if t is None or m is None:
            continue
        times.append(t); mixed.append(m)
        if (idx_ecg is not None) and (e is not None):
            ecg.append(e)

    if not times or not mixed:
        raise ValueError("Failed to parse CSV columns for time/mixed.")

    if no_gt or (idx_ecg is None) or (len(ecg) == 0):
        ecg_out = None
    else:
        L = min(len(times), len(mixed), len(ecg))
        times = times[:L]; mixed = mixed[:L]; ecg_out = ecg[:L]

    return times, mixed, ecg_out

# -----------------------------
# Reconstruction utilities
# -----------------------------

def slice_signal(sig: torch.Tensor, window: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L = sig.numel()
    if L < window:
        pad = window - L
        sig = torch.cat([sig, torch.zeros(pad, dtype=sig.dtype)])
        L = window
    starts = list(range(0, max(L - window + 1, 1), step))
    ws = [sig[s:s+window] for s in starts]
    windows = torch.stack(ws, dim=0) if ws else sig.unsqueeze(0)
    return windows, torch.tensor(starts, dtype=torch.long)

def hann_window(W: int, device: torch.device) -> torch.Tensor:
    n = torch.arange(W, device=device, dtype=torch.float32)
    if W <= 1:
        return torch.ones(W, device=device, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (W - 1))

def overlap_add(windows: torch.Tensor, starts: torch.Tensor, total_len: int, device: torch.device) -> torch.Tensor:
    N, W = windows.shape
    acc = torch.zeros(total_len, device=device, dtype=torch.float32)
    wts = torch.zeros(total_len, device=device, dtype=torch.float32)
    w = hann_window(W, device=device)
    for i in range(N):
        s = int(starts[i].item()); e = min(s + W, total_len)
        seg_len = e - s
        acc[s:e] += windows[i, :seg_len] * w[:seg_len]
        wts[s:e] += w[:seg_len]
    wts = torch.where(wts == 0, torch.ones_like(wts), wts)
    return acc / wts

def gaussian_kernel1d(k: int = 31, sigma: float = 5.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    x = torch.linspace(-3, 3, steps=k, device=device)
    g = torch.exp(-x * x / (2 * (sigma**2 / 9)))
    g = g / g.sum()
    return g

def gaussian_smooth(x: torch.Tensor, k: int = 31, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if k < 3: 
        return x
    g = gaussian_kernel1d(k=k, device=device).view(1, 1, -1)
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
    windows, starts = slice_signal(mixed, segment_length, step)
    N = windows.shape[0]

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            chunk = windows[i:i+batch_size].to(device)
            y = model(chunk.unsqueeze(-1)).squeeze(-1)
            preds.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    out = overlap_add(preds.to(device), starts.to(device), total_len=mixed.numel(), device=device)
    out = gaussian_smooth(out, k=31, device=device)
    return out

# -----------------------------
# Robust R-peak detection & metrics
# -----------------------------

def estimate_fs_from_time(t: torch.Tensor, fs_override: Optional[float] = None) -> float:
    if fs_override and fs_override > 0:
        return float(fs_override)
    if t.numel() < 2:
        return 250.0
    dt = t[1:] - t[:-1]
    dt = dt[dt > 0]
    if dt.numel() == 0:
        return 250.0
    med_dt = torch.median(dt).item()
    fs = 1.0 / max(med_dt, 1e-6)
    # clamp to a reasonable range
    return float(min(max(fs, 20.0), 2000.0))

def moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    k = int(max(1, k))
    if k <= 1:
        return x
    w = torch.ones(1, 1, k, dtype=x.dtype, device=x.device) / float(k)
    y = F.conv1d(x.view(1, 1, -1), w, padding=k//2)
    return y.view(-1)

def bandpass_baseline_removal(x: torch.Tensor, fs: float) -> torch.Tensor:
    # Remove slow-varying baseline & lightly smooth
    long_k = max(3, int(round(0.2 * fs)))   # ~200 ms
    x1 = x - moving_average(x, long_k)
    smooth_k = max(3, int(round(0.03 * fs)))
    if smooth_k % 2 == 0: smooth_k += 1
    x1 = gaussian_smooth(x1, k=smooth_k, device=x.device)
    return x1

def pan_tompkins_lite(x: torch.Tensor, fs: float) -> torch.Tensor:
    # Robust to dtype/device
    k = torch.tensor([-1.0, 0.0, 1.0], dtype=x.dtype, device=x.device).view(1,1,-1) / 2.0
    y = F.conv1d(x.view(1,1,-1), k, padding=1).view(-1)  # derivative
    y = y * y                                            # square
    win = max(3, int(round(0.15 * fs)))                 # integrate ~150 ms
    y = moving_average(y, win)
    return y

def detect_r_peaks(sig: torch.Tensor, fs: float, adaptive_k: float = 0.6,
                   refractory_ms: float = 200.0, debug: bool = False) -> torch.Tensor:
    sig = sig.to(torch.float32)
    if sig.numel() < 5:
        return torch.empty(0, dtype=torch.long)
    pre = bandpass_baseline_removal(sig, fs)
    feat = pan_tompkins_lite(pre, fs)

    # Threshold with floors to avoid zero-variance or too strict thresholds
    sigma = feat.std(unbiased=False)
    mu = feat.mean()
    thr = mu + adaptive_k * max(sigma.item(), 1e-6)
    # Also ensure an absolute minimum threshold relative to feature scale
    thr = float(max(thr, (feat.max().item() * 0.1)))

    above = (feat >= thr).to(torch.int32)
    if above.numel() < 2:
        return torch.empty(0, dtype=torch.long)

    # rising edges of threshold crossings
    crossings = torch.where((above[1:] > above[:-1]))[0] + 1
    refractory = int(max(1, round((refractory_ms / 1000.0) * fs)))
    peaks = []
    last_idx = -10**9
    for s in crossings.tolist():
        if s <= last_idx + refractory:
            continue
        # local refinement
        w = int(max(1, round(0.15 * fs)))
        e = min(s + w, feat.numel())
        if s >= e:
            continue
        seg = feat[s:e]
        rel = int(torch.argmax(seg).item())
        cand = s + rel
        w2 = int(max(1, round(0.05 * fs)))
        s2 = max(0, cand - w2); e2 = min(sig.numel(), cand + w2 + 1)
        raw_seg = pre[s2:e2]
        if raw_seg.numel() == 0:
            continue
        rel2 = int(torch.argmax(raw_seg).item())
        peak_idx = s2 + rel2
        if peak_idx > last_idx + refractory:
            peaks.append(peak_idx)
            last_idx = peak_idx

    if debug:
        torch.save({"sig": sig, "pre": pre, "feat": feat, "thr": thr, "crossings": crossings},
                   "rpeak_debug.pt")

    if len(peaks) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(peaks, dtype=torch.long)

def match_peaks(gt_idx: torch.Tensor, pr_idx: torch.Tensor, fs: float, tol_ms: float = 50.0) -> Tuple[int, int, int, torch.Tensor]:
    tol = int(max(0, round((tol_ms / 1000.0) * fs)))
    i = j = 0
    TP = 0
    matched_diffs = []
    gt_idx = gt_idx.tolist(); pr_idx = pr_idx.tolist()
    while i < len(gt_idx) and j < len(pr_idx):
        di = pr_idx[j] - gt_idx[i]
        if abs(di) <= tol:
            TP += 1
            matched_diffs.append(di)
            i += 1; j += 1
        elif pr_idx[j] < gt_idx[i]:
            j += 1
        else:
            i += 1
    FP = len(pr_idx) - TP
    FN = len(gt_idx) - TP
    diffs = torch.tensor(matched_diffs, dtype=torch.float32) if matched_diffs else torch.empty(0)
    return TP, FP, FN, diffs

# -----------------------------
# Clinical features from RR intervals (same as before)
# -----------------------------

def rr_intervals_ms(peaks: torch.Tensor, fs: float) -> torch.Tensor:
    if peaks.numel() < 2:
        return torch.empty(0)
    rr = (peaks[1:] - peaks[:-1]).to(torch.float32) * (1000.0 / fs)
    return rr

# -----------------------------
# Minimal demo main for R-peak part with plotting overlay
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--no_gt", action="store_true")
    p.add_argument("--fs", type=float, default=0.0, help="Override sampling rate in Hz (if 0, infer from time column).")
    p.add_argument("--rpeak_tol_ms", type=float, default=50.0)
    p.add_argument("--rpeak_adaptive_k", type=float, default=0.6)
    p.add_argument("--debug_rpeaks", action="store_true")
    p.add_argument("--out_prefix", type=str, default="rpeak_fix")
    args = p.parse_args()

    # Load
    t_list, mixed_list, ecg_list = load_csv(args.csv, no_gt=args.no_gt)
    t = to_tensor(t_list)
    # If time not strictly increasing, sort everything by time
    if t.numel() > 1 and not torch.all(t[1:] > t[:-1]):
        idx = torch.argsort(t)
        t = t[idx]
        mixed = to_tensor(mixed_list)[idx]
        ecg = to_tensor(ecg_list)[idx] if ecg_list is not None else None
    else:
        mixed = to_tensor(mixed_list)
        ecg = to_tensor(ecg_list) if ecg_list is not None else None

    fs = estimate_fs_from_time(t, fs_override=args.fs)

    # Choose a signal to test R-peak detection: if GT available, compare both
    sig_gt = ecg if (ecg is not None and ecg.numel() > 0) else mixed
    sig_pr = mixed  # as placeholder; replace with your reconstruction if needed

    # Detect peaks with guards
    try:
        gt_idx = detect_r_peaks(sig_gt, fs, adaptive_k=args.rpeak_adaptive_k, debug=args.debug_rpeaks)
    except Exception as e:
        print(f"[ERROR] R-peak detection on GT signal failed: {e}")
        gt_idx = torch.empty(0, dtype=torch.long)

    try:
        pr_idx = detect_r_peaks(sig_pr, fs, adaptive_k=args.rpeak_adaptive_k, debug=args.debug_rpeaks)
    except Exception as e:
        print(f"[ERROR] R-peak detection on PR signal failed: {e}")
        pr_idx = torch.empty(0, dtype=torch.long)

    # Basic overlay plot (uses each signal's own amplitude to avoid mismatches)
    max_points = 200000
    def downsample(a: torch.Tensor) -> torch.Tensor:
        if a.numel() <= max_points: return a
        step = max(1, a.numel() // max_points)
        return a[::step]

    tt = downsample(t).cpu().numpy()
    gg = downsample(sig_gt).cpu().numpy()
    rr = downsample(sig_pr).cpu().numpy()

    plt.figure(figsize=(14,5))
    plt.plot(tt[:gg.shape[0]], gg, label="Signal A (GT if available)", linewidth=1.0)
    plt.plot(tt[:rr.shape[0]], rr, label="Signal B (Recon or Mixed)", linewidth=1.0, alpha=0.9)

    def safe_scatter(idx: torch.Tensor, base_sig: torch.Tensor, label: str, marker: str):
        if idx.numel() == 0:
            return
        idx = idx.clamp(min=0, max=base_sig.numel()-1)
        times = t[idx].cpu().numpy()
        amps  = base_sig[idx].cpu().numpy()
        plt.scatter(times, amps, s=18, marker=marker, label=label, zorder=5)

    safe_scatter(gt_idx, sig_gt, "R peaks (A)", "o")
    safe_scatter(pr_idx, sig_pr, "R peaks (B)", "x")
    plt.title(f"R-peak overlay (fs={fs:.1f} Hz, k={args.rpeak_adaptive_k}, tol={args.rpeak_tol_ms} ms)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.legend(); plt.tight_layout()
    out_img = f"{args.out_prefix}_rpeaks_overlay.png"
    plt.savefig(out_img, dpi=220); plt.close()
    print(f"[OK] Saved: {out_img}")

if __name__ == "__main__":
    main()
