#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG reconstruction evaluation toolkit (Torch-only metrics + visualizations)

Adds to previous script:
1) Extra metrics: MAE, Pearson r, SNR, PRD, DTW (with Sakoe-Chiba band)
2) R-peak metrics: timing deviation (ms) + Precision/Recall/F1 with tolerance (ms)
   - Torch-only peak detector (Pan–Tompkins-lite): bandpass (baseline removal), derivative, square, moving integration, adaptive threshold + refractory
   - Overlays R-peaks on waveform figure
3) Clinical feature comparisons from R-R intervals: HR, SDNN, RMSSD, pNN50 (+ bar plot) and Bland–Altman for HR

Dependencies: torch, matplotlib, csv (stdlib), numpy (only for saving .npy). No scipy/sklearn required.
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
DEFAULT_CKPT = RELEASE_ROOT / "models" / "ecg_reconstruction" / "best_model.pth"

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
# Metrics (overall, windowed, DTW)
# -----------------------------

def summarize_with_ci95(x: torch.Tensor) -> Tuple[float, float, float, int]:
    n = int(x.numel())
    mean = x.mean()
    std  = x.std(unbiased=True) if n > 1 else torch.tensor(0.0, dtype=x.dtype, device=x.device)
    se   = std / math.sqrt(max(n, 1))
    ci95 = 1.96 * se
    return mean.item(), std.item(), ci95.item(), n

def windowed_metrics(gt: torch.Tensor, pr: torch.Tensor, frame_len: int = 2000, frame_overlap: float = 0.0) -> Dict:
    L = min(gt.numel(), pr.numel())
    gt = gt[:L]; pr = pr[:L]

    step = max(int(frame_len * (1.0 - frame_overlap)), 1)
    starts = list(range(0, max(L - frame_len + 1, 1), step)) or [0]

    mses, maes, coss = [], [], []
    eps = 1e-12

    for s in starts:
        e = min(s + frame_len, L)
        g = gt[s:e]; p = pr[s:e]
        if g.numel() < 2:
            continue
        mses.append(torch.mean((g - p) ** 2))
        maes.append(torch.mean(torch.abs(g - p)))
        coss.append(torch.dot(g, p) / (g.norm() * p.norm() + eps))

    if len(mses) == 0:
        g = gt; p = pr
        mses = [torch.mean((g - p) ** 2)]
        maes = [torch.mean(torch.abs(g - p))]
        coss = [torch.dot(g, p) / (g.norm() * p.norm() + eps)]

    mses = torch.stack(mses); maes = torch.stack(maes); coss = torch.stack(coss)

    def pack(x):
        m, s, ci, n = summarize_with_ci95(x)
        return {"mean": m, "std": s, "ci95": ci, "N": n, "all": x}

    return {"mse": pack(mses), "mae": pack(maes), "coss": pack(coss), "frame_len": frame_len, "frame_overlap": frame_overlap}

def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(); y = y - y.mean()
    r = torch.dot(x, y) / (x.norm() * y.norm() + 1e-12)
    return r.item()

def snr_db(gt: torch.Tensor, pr: torch.Tensor) -> float:
    num = torch.mean(gt ** 2)
    den = torch.mean((gt - pr) ** 2) + 1e-12
    return 10.0 * torch.log10(num / den).item()

def prd_percent(gt: torch.Tensor, pr: torch.Tensor) -> float:
    num = torch.norm(gt - pr)
    den = torch.norm(gt) + 1e-12
    return 100.0 * (num / den).item()

def dtw_distance_torch(x: torch.Tensor, y: torch.Tensor, radius: int = None) -> float:
    x = x.view(-1); y = y.view(-1)
    nx = (x - x.mean()) / (x.std(unbiased=False) + 1e-12)
    ny = (y - y.mean()) / (y.std(unbiased=False) + 1e-12)
    n = nx.numel(); m = ny.numel()
    if radius is None:
        radius = max(1, int(0.05 * max(n, m)))  # 5% by default
    INF = 1e18
    prev = torch.full((m + 1,), INF, dtype=torch.float32)
    curr = torch.full((m + 1,), INF, dtype=torch.float32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        lb = max(1, i - radius)
        ub = min(m, i + radius)
        curr[0] = INF
        if lb > 1:
            curr[1:lb] = INF
        if ub < m:
            curr[ub+1:] = INF
        for j in range(lb, ub + 1):
            cost = (nx[i-1] - ny[j-1]).abs()
            curr[j] = cost + torch.min(torch.stack([prev[j], curr[j-1], prev[j-1]]))
        prev, curr = curr, prev
    dist = prev[m].item()
    return dist

# -----------------------------
# R-peak detection & metrics
# -----------------------------

def estimate_fs_from_time(t: torch.Tensor) -> float:
    if t.numel() < 2:
        return 250.0
    dt = t[1:] - t[:-1]
    dt = dt[dt > 0]
    if dt.numel() == 0:
        return 250.0
    med_dt = dt.median().item()
    return 1.0 / max(med_dt, 1e-6)

def moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    w = torch.ones(1, 1, k, dtype=x.dtype, device=x.device) / float(k)
    y = F.conv1d(x.view(1, 1, -1), w, padding=k//2)
    return y.view(-1)

def bandpass_baseline_removal(x: torch.Tensor, fs: float) -> torch.Tensor:
    long_k = max(1, int(0.2 * fs))   # ~200 ms
    x1 = x - moving_average(x, long_k)
    x1 = gaussian_smooth(x1, k=max(3, int(0.03 * fs // 2 * 2 + 1)), device=x.device)  # ~30 ms
    return x1

def pan_tompkins_lite(x: torch.Tensor, fs: float) -> torch.Tensor:
    k = torch.tensor([-1.0, 0.0, 1.0], dtype=x.dtype, device=x.device).view(1,1,-1) / 2.0
    y = F.conv1d(x.view(1,1,-1), k, padding=1).view(-1)
    y = y * y
    win = max(1, int(0.15 * fs))
    y = moving_average(y, win)
    return y

def detect_r_peaks(sig: torch.Tensor, fs: float, adaptive_k: float = 0.5, refractory_ms: float = 200.0) -> torch.Tensor:
    sig = sig.to(torch.float32)
    pre = bandpass_baseline_removal(sig, fs)
    feat = pan_tompkins_lite(pre, fs)

    mu, sigma = feat.mean(), feat.std(unbiased=False) + 1e-12
    thr = mu + adaptive_k * sigma
    above = (feat >= thr).to(torch.int32)

    crossings = torch.where((above[1:] > above[:-1]))[0] + 1
    refractory = int((refractory_ms / 1000.0) * fs)
    peaks = []
    last_idx = -10**9
    for s in crossings.tolist():
        if s <= last_idx + refractory:
            continue
        w = int(0.15 * fs)
        e = min(s + w, feat.numel())
        if s >= e:
            continue
        seg = feat[s:e]
        rel = int(torch.argmax(seg).item())
        cand = s + rel
        w2 = int(0.05 * fs)
        s2 = max(0, cand - w2); e2 = min(sig.numel(), cand + w2 + 1)
        raw_seg = pre[s2:e2]
        if raw_seg.numel() == 0:
            continue
        rel2 = int(torch.argmax(raw_seg).item())
        peak_idx = s2 + rel2
        if peak_idx > last_idx + refractory:
            peaks.append(peak_idx)
            last_idx = peak_idx
    if len(peaks) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(peaks, dtype=torch.long)

def match_peaks(gt_idx: torch.Tensor, pr_idx: torch.Tensor, fs: float, tol_ms: float = 50.0) -> Tuple[int, int, int, torch.Tensor]:
    tol = int((tol_ms / 1000.0) * fs)
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

def rpeak_metrics(time: torch.Tensor,
                  gt_sig: torch.Tensor,
                  pr_sig: torch.Tensor,
                  tol_ms: float = 50.0,
                  adaptive_k: float = 0.5) -> Dict:
    fs = estimate_fs_from_time(time)
    gt_idx = detect_r_peaks(gt_sig, fs, adaptive_k=adaptive_k)
    pr_idx = detect_r_peaks(pr_sig, fs, adaptive_k=adaptive_k)
    TP, FP, FN, diffs = match_peaks(gt_idx, pr_idx, fs, tol_ms=tol_ms)
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    if diffs.numel() > 0:
        abs_ms = (diffs.abs() * 1000.0 / fs)
        mean_ms = abs_ms.mean().item()
        med_ms  = abs_ms.median().item()
        std_ms  = abs_ms.std(unbiased=True).item() if abs_ms.numel() > 1 else 0.0
    else:
        mean_ms = med_ms = std_ms = float('nan')
        abs_ms = torch.empty(0)
    return {
        "fs": fs,
        "gt_idx": gt_idx,
        "pr_idx": pr_idx,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": prec, "recall": rec, "f1": f1,
        "timing_abs_ms": abs_ms,
        "timing_mean_ms": mean_ms,
        "timing_median_ms": med_ms,
        "timing_std_ms": std_ms
    }

# -----------------------------
# Clinical features from RR intervals
# -----------------------------

def rr_intervals_ms(peaks: torch.Tensor, fs: float) -> torch.Tensor:
    if peaks.numel() < 2:
        return torch.empty(0)
    rr = (peaks[1:] - peaks[:-1]).to(torch.float32) * (1000.0 / fs)
    return rr

def clinical_features_from_rr(rr_ms: torch.Tensor) -> Dict[str, float]:
    if rr_ms.numel() == 0:
        return {"HR_bpm": float('nan'), "SDNN_ms": float('nan'),
                "RMSSD_ms": float('nan'), "pNN50_%": float('nan')}
    HR = 60000.0 / rr_ms.mean().item()
    SDNN = rr_ms.std(unbiased=True).item() if rr_ms.numel() > 1 else 0.0
    diff = rr_ms[1:] - rr_ms[:-1]
    RMSSD = torch.sqrt(torch.mean(diff * diff)).item() if diff.numel() > 0 else 0.0
    pNN50 = (torch.mean((diff.abs() > 50.0).float()).item() * 100.0) if diff.numel() > 0 else 0.0
    return {"HR_bpm": HR, "SDNN_ms": SDNN, "RMSSD_ms": RMSSD, "pNN50_%": pNN50}

def bland_altman(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    diff = b - a
    md = diff.mean().item()
    sd = diff.std(unbiased=True).item() if diff.numel() > 1 else 0.0
    loa_low = md - 1.96 * sd
    loa_up  = md + 1.96 * sd
    return md, loa_low, loa_up

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seg_len", type=int, default=2000)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="recon_plus")

    # Metrics options
    parser.add_argument("--metric_frame_len", type=int, default=2000)
    parser.add_argument("--metric_frame_overlap", type=float, default=0.0)
    parser.add_argument("--dtw_band_ratio", type=float, default=0.05, help="Sakoe–Chiba radius as a fraction of length (5% default).")

    # R-peak options
    parser.add_argument("--rpeak_tol_ms", type=float, default=50.0)
    parser.add_argument("--rpeak_adaptive_k", type=float, default=0.6, help="Threshold = mean + k*std on integrated signal.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load
    t_list, mixed_list, ecg_list = load_csv(args.csv, no_gt=args.no_gt)
    t = to_tensor(t_list)
    mixed = to_tensor(mixed_list)
    ecg = to_tensor(ecg_list) if ecg_list is not None else None

    # Model
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
        mixed=mixed.to(device), model=model, device=device,
        segment_length=args.seg_len, overlap=args.overlap, batch_size=args.batch_size
    ).cpu()

    # Overall/basic metrics
    if ecg is not None and ecg.numel() > 0:
        L = min(ecg.numel(), recon.numel())
        gt = ecg[:L]
        pr = recon[:L]

        #mse_all = gt - pr ** 2
        #mae_all = torch.abs(gt - pr)
        mse_overall = torch.mean((gt - pr) ** 2).item()
        mae_overall = torch.mean(torch.abs(gt - pr)).item()
        cos_overall = float(torch.dot(gt, pr) / (gt.norm() * pr.norm() + 1e-12))
        r_overall   = pearsonr_torch(gt, pr)
        snr = snr_db(gt, pr)
        prd = prd_percent(gt, pr)

        print(f"[Overall] MSE={mse_overall:.6f} | MAE={mae_overall:.6f} | CosSim={cos_overall:.6f} | Pearson r={r_overall:.6f} | SNR(dB)={snr:.2f} | PRD(%)={prd:.2f}")

        # Windowed metrics (for error bars)
        wm = windowed_metrics(gt, pr, frame_len=args.metric_frame_len, frame_overlap=args.metric_frame_overlap)
        print(f"[Windowed] MSE mean±CI: {wm['mse']['mean']:.6f} ± {wm['mse']['ci95']:.6f} (N={wm['mse']['N']}) | "
              f"MAE mean±CI: {wm['mae']['mean']:.6f} ± {wm['mae']['ci95']:.6f} | "
              f"CosSim mean±CI: {wm['coss']['mean']:.6f} ± {wm['coss']['ci95']:.6f} | N={wm['coss']['N']}"
              f" | Pearson mean±CI: {wm['r']['mean']:.6f} ± {wm['r']['ci95']:.6f} | N={wm['r']['N']}"
              f" | SNR mean±CI: {wm['snr']['mean']:.6f} ± {wm['snr']['ci95']:.6f} | N={wm['snr']['N']}"
              f" | PRD mean±CI: {wm['prd']['mean']:.6f} ± {wm['prd']['ci95']:.6f} | N={wm['prd']['N']}"
              )

        # DTW (optional, guarded to avoid long runtimes)
        dtw_dist = float('nan')
        try:
            band = int(args.dtw_band_ratio * L)
            # Light downsampling for speed if very long sequences
            if L > 20000:
                ds = max(1, L // 20000)
                gt_ds = gt[::ds]
                pr_ds = pr[::ds]
                band = max(1, int(args.dtw_band_ratio * gt_ds.numel()))
                dtw_dist = dtw_distance_torch(gt_ds, pr_ds, radius=max(1, band))
            else:
                dtw_dist = dtw_distance_torch(gt, pr, radius=max(1, band))
            # Uncomment to log
            # print(f"[DTW] band_ratio={args.dtw_band_ratio:.3f} -> radius={band} samples | DTW={dtw_dist:.4f}")
        except Exception:
            # Keep DTW optional; proceed without blocking the pipeline
            dtw_dist = float('nan')

        # R-peak metrics
        rp = rpeak_metrics(t[:L], gt, pr, tol_ms=args.rpeak_tol_ms, adaptive_k=args.rpeak_adaptive_k)
        print(f"[R-peak] fs={rp['fs']:.2f} Hz | tol={args.rpeak_tol_ms:.1f} ms | "
              f"Precision={rp['precision']:.3f} Recall={rp['recall']:.3f} F1={rp['f1']:.3f} | "
              f"Timing(mean±std)={rp['timing_mean_ms']:.1f}±{rp['timing_std_ms']:.1f} ms | "
              f"N_TP={rp['TP']} FP={rp['FP']} FN={rp['FN']}")

        # Clinical features (RR-based)
        gt_rr = rr_intervals_ms(rp["gt_idx"], rp["fs"])
        pr_rr = rr_intervals_ms(rp["pr_idx"], rp["fs"])
        gt_feat = clinical_features_from_rr(gt_rr)
        pr_feat = clinical_features_from_rr(pr_rr)

        # ---- Figures ----
        outp = args.out_prefix

        # 1) Waveforms with R-peak overlay
        max_points = 200000
        def downsample(a: torch.Tensor) -> torch.Tensor:
            if a.numel() <= max_points: return a
            step = max(1, a.numel() // max_points)
            return a[::step]

        tt = downsample(t[:L]); gg = downsample(gt); rc = downsample(pr)
        plt.figure(figsize=(14, 6))
        plt.plot(tt.numpy()[:gg.numel()], gg.numpy(), label="Ground Truth ECG", linewidth=1.0)
        plt.plot(tt.numpy()[:rc.numel()], rc.numpy(), label="Reconstructed ECG", linewidth=1.0, alpha=0.9)
        def plot_peaks(idx: torch.Tensor, label: str, marker: str):
            if idx.numel() == 0: return
            times = t[idx]
            amps  = gt[idx.clamp_max(gt.numel()-1)]
            plt.scatter(times.numpy(), amps.numpy(), s=20, marker=marker, label=label, zorder=5)
        plot_peaks(rp["gt_idx"], "R peaks (GT)", "o")
        plot_peaks(rp["pr_idx"], "R peaks (Recon)", "x")
        plt.title("ECG Waveforms with R-peak Overlay")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outp}_wave_peaks.png", dpi=200)
        plt.close()

        # 2) Error-bar bar chart (windowed metrics)
        plt.figure(figsize=(6,4))
        metrics = ["MSE", "MAE", "CosSim"]
        means = [wm["mse"]["mean"], wm["mae"]["mean"], wm["coss"]["mean"]]
        yerr  = [wm["mse"]["ci95"], wm["mae"]["ci95"], wm["coss"]["ci95"]]
        plt.bar(metrics, means, yerr=yerr, capsize=6)
        plt.ylabel("Value"); plt.title("Windowed Metrics (mean ± 95% CI)")
        plt.tight_layout(); plt.savefig(f"{outp}_metrics_ci.png", dpi=200); plt.close()

        # 3) Clinical features comparison (bar)
        feat_names = ["HR_bpm", "SDNN_ms", "RMSSD_ms", "pNN50_%"]
        gt_vals = [gt_feat[k] for k in feat_names]
        pr_vals = [pr_feat[k] for k in feat_names]
        x = np.arange(len(feat_names))
        width = 0.35
        plt.figure(figsize=(7,4))
        plt.bar(x - width/2, gt_vals, width, label="GT")
        plt.bar(x + width/2, pr_vals, width, label="Recon")
        plt.xticks(x, feat_names, rotation=15)
        plt.ylabel("Value"); plt.title("Clinical Features (RR-based)")
        plt.legend(); plt.tight_layout(); plt.savefig(f"{outp}_clinical_features.png", dpi=200); plt.close()

        # 4) Bland–Altman for HR
        fs = rp["fs"]
        def hr_series(peaks: torch.Tensor) -> torch.Tensor:
            if peaks.numel() < 2: return torch.empty(0)
            hrs = []
            start = 0
            while start < peaks.numel()-1:
                end = min(peaks.numel(), start + 20)  # ~20 beats window
                rr = rr_intervals_ms(peaks[start:end], fs=fs)
                if rr.numel() > 0:
                    hrs.append(60000.0 / rr.mean())
                start += 10
            return torch.stack(hrs) if len(hrs) else torch.empty(0)
        hr_gt = hr_series(rp["gt_idx"]); hr_pr = hr_series(rp["pr_idx"])
        if hr_gt.numel() > 0 and hr_pr.numel() > 0:
            n = min(hr_gt.numel(), hr_pr.numel())
            a = hr_gt[:n]; b = hr_pr[:n]
            md, lo, hi = bland_altman(a, b)
            mean_ab = 0.5 * (a + b)
            diff = (b - a)
            plt.figure(figsize=(6,4))
            plt.scatter(mean_ab.numpy(), diff.numpy(), s=15)
            plt.axhline(md, linestyle='--')
            plt.axhline(lo, linestyle=':')
            plt.axhline(hi, linestyle=':')
            plt.xlabel("Mean HR (bpm)"); plt.ylabel("Recon - GT (bpm)")
            plt.title("Bland–Altman: HR")
            plt.tight_layout(); plt.savefig(f"{outp}_bland_altman_hr.png", dpi=200); plt.close()

        # Save arrays
        np.save(f"{outp}.npy", pr.numpy())
        torch.save({
            "overall": {"mse": mse_overall, "mae": mae_overall, "cos": cos_overall, "pearson_r": r_overall, "snr_db": snr, "prd_percent": prd},
            "windowed": {
                "mse_mean": wm["mse"]["mean"], "mse_ci95": wm["mse"]["ci95"],
                "mae_mean": wm["mae"]["mean"], "mae_ci95": wm["mae"]["ci95"],
                "coss_mean": wm["coss"]["mean"], "coss_ci95": wm["coss"]["ci95"]
            },
            "dtw": {"band_ratio": args.dtw_band_ratio, "distance": dtw_dist},
            "rpeak": {
                "fs": rp["fs"], "precision": rp["precision"], "recall": rp["recall"], "f1": rp["f1"],
                "TP": rp["TP"], "FP": rp["FP"], "FN": rp["FN"],
                "timing_mean_ms": rp["timing_mean_ms"], "timing_std_ms": rp["timing_std_ms"],
                "gt_idx": rp["gt_idx"], "pr_idx": rp["pr_idx"], "timing_abs_ms": rp["timing_abs_ms"]
            },
            "clinical": {"gt": gt_feat, "recon": pr_feat}
        }, f"{outp}_metrics.pt")

    else:
        np.save(f"{args.out_prefix}.npy", recon.numpy())
        print("No ground-truth ECG found; saved reconstruction only.")

if __name__ == "__main__":
    main()
