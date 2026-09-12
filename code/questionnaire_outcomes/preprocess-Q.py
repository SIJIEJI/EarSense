#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pyedflib
from scipy.signal import butter, sosfiltfilt, welch, resample_poly, find_peaks


# -------------------------
# Utilities
# -------------------------
def log(msg: str):
    print(msg, flush=True)

def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_file_by_candidates(folder: str, candidates: List[str]) -> Optional[str]:
    files = os.listdir(folder)
    lower = {f.lower(): f for f in files}
    for c in candidates:
        if c.lower() in lower:
            return os.path.join(folder, lower[c.lower()])
    # fuzzy contains
    for c in candidates:
        for f in files:
            if c.lower() in f.lower():
                return os.path.join(folder, f)
    return None

def safe_resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample using integer ratio via resample_poly; robust to float fs."""
    if fs_out is None or fs_out <= 0 or abs(fs_out - fs_in) < 1e-6:
        return x.astype(np.float32)
    fin = int(round(float(fs_in)))
    fout = int(round(float(fs_out)))
    if fin <= 0 or fout <= 0:
        return x.astype(np.float32)
    g = np.gcd(fout, fin)
    up = fout // g
    down = fin // g
    return resample_poly(x, up, down).astype(np.float32)

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    if fs <= 0:
        return x.astype(np.float32)
    nyq = 0.5 * fs
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    if lo_n >= hi_n:
        return x.astype(np.float32)
    sos = butter(order, [lo_n, hi_n], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x.astype(np.float32)

def lowpass(x: np.ndarray, fs: float, hi: float, order: int = 4) -> np.ndarray:
    if fs <= 0:
        return x.astype(np.float32)
    nyq = 0.5 * fs
    hi_n = min(0.999999, hi / nyq)
    sos = butter(order, hi_n, btype="low", output="sos")
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x.astype(np.float32)

def read_edf_channels(edf_path: str) -> Dict[str, Tuple[np.ndarray, float]]:
    """Read all channels (label -> (signal, fs))."""
    out = {}
    with pyedflib.EdfReader(edf_path) as f:
        labels = [lab.strip() for lab in f.getSignalLabels()]
        for i, lab in enumerate(labels):
            fs = float(f.getSampleFrequency(i))
            sig = f.readSignal(i).astype(np.float32)
            out[lab] = (sig, fs)
    return out

def pick_channel_by_contains(ch_dict: Dict[str, Tuple[np.ndarray, float]],
                            keywords: List[str],
                            exclude_keywords: List[str] = None) -> Dict[str, Tuple[np.ndarray, float]]:
    """Return subset where label contains any keyword; optionally exclude."""
    out = {}
    exclude_keywords = exclude_keywords or []
    for lab, (sig, fs) in ch_dict.items():
        low = lab.lower()
        if any(k.lower() in low for k in keywords) and not any(e.lower() in low for e in exclude_keywords):
            out[lab] = (sig, fs)
    return out

def choose_ppg_channels(ppg_dict: Dict[str, Tuple[np.ndarray, float]]) -> Dict[str, Tuple[np.ndarray, float]]:
    """Prefer raw PPG green/red/ir if present; otherwise keep up to 3."""
    if len(ppg_dict) == 0:
        return {}
    # Prefer those containing green/red/ir
    preferred = {}
    for key in ["green", "red", "ir"]:
        for lab, v in ppg_dict.items():
            if key in lab.lower():
                preferred[lab] = v
                break
    if len(preferred) > 0:
        return preferred
    # fallback: take first 3
    out = {}
    for lab in sorted(ppg_dict.keys())[:3]:
        out[lab] = ppg_dict[lab]
    return out

def minute_features_generic(x: np.ndarray, fs: float) -> np.ndarray:
    """Return [mean, std, rms, line_length] for a 1-min chunk."""
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-8)
    rms = float(np.sqrt(np.mean(x ** 2)))
    ll = float(np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0.0)
    return np.array([mu, sd, rms, ll], dtype=np.float32)

def minute_features_eeg_bands(x: np.ndarray, fs: float) -> np.ndarray:
    """EEG: generic stats + relative bandpowers delta/theta/alpha/sigma/beta."""
    stats = minute_features_generic(x, fs)
    # Welch PSD
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))
    def bp(lo, hi):
        m = (f >= lo) & (f <= hi)
        if not np.any(m):
            return 0.0
        return float(np.trapz(Pxx[m], f[m]))
    total = bp(0.5, 30.0) + 1e-12
    delta = bp(0.5, 4) / total
    theta = bp(4, 8) / total
    alpha = bp(8, 12) / total
    sigma = bp(12, 16) / total
    beta  = bp(16, 30) / total
    return np.concatenate([stats, np.array([delta, theta, alpha, sigma, beta], dtype=np.float32)], axis=0)

def minute_features_hr_proxy(x: np.ndarray, fs: float) -> np.ndarray:
    """
    ECG/PPG: generic stats + peaks-per-minute (proxy for HR).
    Not perfect, but stable and useful with tiny datasets.
    """
    stats = minute_features_generic(x, fs)
    z = (x - np.median(x)) / (np.std(x) + 1e-8)
    min_dist = max(1, int(fs * 0.3))  # ~200 bpm upper bound
    peaks, _ = find_peaks(z, distance=min_dist, prominence=0.3)
    if len(peaks) < 3:
        peaks, _ = find_peaks(-z, distance=min_dist, prominence=0.3)
    ppm = float(len(peaks))  # per 60s chunk
    return np.concatenate([stats, np.array([ppm], dtype=np.float32)], axis=0)

def build_sequence_from_signal(x: np.ndarray, fs: float, feat_fn, minutes: int) -> np.ndarray:
    """Convert whole-night signal into per-minute feature sequence: (minutes, fdim)."""
    chunk = int(round(fs * 60))
    if chunk <= 10:
        raise ValueError("fs too small for per-minute chunking")
    need = minutes * chunk
    x = x[:need]
    seq = []
    for i in range(minutes):
        seg = x[i * chunk:(i + 1) * chunk]
        seq.append(feat_fn(seg, fs))
    return np.stack(seq, axis=0).astype(np.float32)

def compute_common_minutes(signals: List[Tuple[np.ndarray, float]], max_minutes: int) -> int:
    """Use min duration across modalities as usable minutes."""
    if len(signals) == 0:
        return 0
    durs = []
    for sig, fs in signals:
        if fs <= 0:
            continue
        durs.append(len(sig) / fs)
    if len(durs) == 0:
        return 0
    minutes = int(np.floor(min(durs) / 60.0))
    return int(max(1, min(minutes, max_minutes)))


# -------------------------
# Subject processing
# -------------------------
def extract_subject_sequence(patient_dir: str,
                             eeg_fs: float,
                             cardio_fs: float,
                             acc_fs: float,
                             max_minutes: int,
                             prefer_ear_eeg: bool = True) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Returns:
      X_seq: (T, F)
      feature_names: list length F
      meta: dict for debugging (used channels, durations, etc.)
    """
    meta = {"patient_dir": patient_dir, "used": {}}

    # Find EDFs
    sleep_all = find_file_by_candidates(patient_dir, ["Sleep_all.edf", "sleep_all.edf"])
    ear_eeg  = find_file_by_candidates(patient_dir, ["Ear EEG_200.edf", "Ear_EEG_200.edf", "ear eeg_200.edf", "Ear EEG.edf", "Ear_EEG.edf"])

    if sleep_all is None and ear_eeg is None:
        raise FileNotFoundError(f"No EDF found in {patient_dir}")

    # Read Sleep_all channels (ECG/PPG/ACC)
    sleep_ch = {}
    if sleep_all is not None:
        sleep_ch = read_edf_channels(sleep_all)

    # EEG: prefer ear EEG file if present
    eeg_sig = None
    if prefer_ear_eeg and ear_eeg is not None:
        eeg_ch = read_edf_channels(ear_eeg)
        # choose channel containing 'eeg' else first
        eeg_candidates = pick_channel_by_contains(eeg_ch, ["eeg"])
        if len(eeg_candidates) > 0:
            lab = sorted(eeg_candidates.keys())[0]
            eeg_sig = (eeg_candidates[lab][0], eeg_candidates[lab][1], lab, "Ear EEG_200.edf")
        else:
            lab = sorted(eeg_ch.keys())[0]
            eeg_sig = (eeg_ch[lab][0], eeg_ch[lab][1], lab, "Ear EEG_200.edf")
    else:
        eeg_candidates = pick_channel_by_contains(sleep_ch, ["eeg"])
        if len(eeg_candidates) > 0:
            lab = sorted(eeg_candidates.keys())[0]
            eeg_sig = (eeg_candidates[lab][0], eeg_candidates[lab][1], lab, "Sleep_all.edf")

    # ECG
    ecg_candidates = pick_channel_by_contains(sleep_ch, ["ecg"])
    ecg_sig = None
    if len(ecg_candidates) > 0:
        lab = sorted(ecg_candidates.keys())[0]
        ecg_sig = (ecg_candidates[lab][0], ecg_candidates[lab][1], lab)

    # PPG (exclude 30s summary)
    ppg_candidates = pick_channel_by_contains(sleep_ch, ["ppg"], exclude_keywords=["30s", "annotation"])
    ppg_candidates = choose_ppg_channels(ppg_candidates)

    # ACC axes
    acc_candidates = pick_channel_by_contains(sleep_ch, ["acce", "acc"], exclude_keywords=["annotation"])
    ax = ay = az = None
    fs_acc = None
    for lab, (sig, fs) in acc_candidates.items():
        low = lab.lower()
        if "ax" in low:
            ax, fs_acc = sig, fs
        elif "ay" in low:
            ay, fs_acc = sig, fs
        elif "az" in low:
            az, fs_acc = sig, fs

    # Prepare signals list for common duration
    sig_list = []
    if eeg_sig is not None:
        sig_list.append((eeg_sig[0], eeg_sig[1]))
    if ecg_sig is not None:
        sig_list.append((ecg_sig[0], ecg_sig[1]))
    for lab, (sig, fs) in ppg_candidates.items():
        sig_list.append((sig, fs))
    if ax is not None and ay is not None and az is not None and fs_acc is not None:
        sig_list.append((ax, fs_acc))
        sig_list.append((ay, fs_acc))
        sig_list.append((az, fs_acc))

    T_minutes = compute_common_minutes(sig_list, max_minutes=max_minutes)
    if T_minutes <= 0:
        raise RuntimeError(f"Cannot compute usable minutes for {patient_dir}")

    meta["minutes_used"] = T_minutes

    seq_parts = []
    feat_names = []

    # EEG features
    if eeg_sig is not None:
        sig, fs, lab, src = eeg_sig
        x = bandpass(sig, fs, 0.5, 30.0, order=4)
        x = safe_resample(x, fs, eeg_fs)
        x = bandpass(x, eeg_fs, 0.5, 30.0, order=4)
        s = build_sequence_from_signal(x, eeg_fs, minute_features_eeg_bands, minutes=T_minutes)
        seq_parts.append(s)
        base = f"EEG_{normalize_name(lab)}"
        feat_names += [f"{base}_mean", f"{base}_std", f"{base}_rms", f"{base}_ll",
                       f"{base}_bp_delta", f"{base}_bp_theta", f"{base}_bp_alpha", f"{base}_bp_sigma", f"{base}_bp_beta"]
        meta["used"]["EEG"] = {"label": lab, "src": src, "fs_in": fs, "fs_out": eeg_fs, "n": len(sig)}

    # ECG features
    if ecg_sig is not None:
        sig, fs, lab = ecg_sig
        x = bandpass(sig, fs, 0.5, 40.0, order=3)
        x = safe_resample(x, fs, cardio_fs)
        x = bandpass(x, cardio_fs, 0.5, 40.0, order=3)
        s = build_sequence_from_signal(x, cardio_fs, minute_features_hr_proxy, minutes=T_minutes)
        seq_parts.append(s)
        base = f"ECG_{normalize_name(lab)}"
        feat_names += [f"{base}_mean", f"{base}_std", f"{base}_rms", f"{base}_ll", f"{base}_peaks_per_min"]
        meta["used"]["ECG"] = {"label": lab, "src": "Sleep_all.edf", "fs_in": fs, "fs_out": cardio_fs, "n": len(sig)}

    # PPG features
    for lab, (sig, fs) in ppg_candidates.items():
        x = bandpass(sig, fs, 0.5, 8.0, order=3)
        x = safe_resample(x, fs, cardio_fs)
        x = bandpass(x, cardio_fs, 0.5, 8.0, order=3)
        s = build_sequence_from_signal(x, cardio_fs, minute_features_hr_proxy, minutes=T_minutes)
        seq_parts.append(s)
        base = f"PPG_{normalize_name(lab)}"
        feat_names += [f"{base}_mean", f"{base}_std", f"{base}_rms", f"{base}_ll", f"{base}_peaks_per_min"]
        meta["used"].setdefault("PPG", []).append({"label": lab, "src": "Sleep_all.edf", "fs_in": fs, "fs_out": cardio_fs, "n": len(sig)})

    # ACC features (magnitude only)
    if ax is not None and ay is not None and az is not None and fs_acc is not None:
        ax2 = safe_resample(ax, fs_acc, acc_fs)
        ay2 = safe_resample(ay, fs_acc, acc_fs)
        az2 = safe_resample(az, fs_acc, acc_fs)
        # light smoothing
        ax2 = lowpass(ax2, acc_fs, 10.0, order=3)
        ay2 = lowpass(ay2, acc_fs, 10.0, order=3)
        az2 = lowpass(az2, acc_fs, 10.0, order=3)
        mag = np.sqrt(ax2**2 + ay2**2 + az2**2).astype(np.float32)
        s = build_sequence_from_signal(mag, acc_fs, minute_features_generic, minutes=T_minutes)
        seq_parts.append(s)
        base = "ACC_mag"
        feat_names += [f"{base}_mean", f"{base}_std", f"{base}_rms", f"{base}_ll"]
        meta["used"]["ACC"] = {"labels": ["Ax","Ay","Az"], "src": "Sleep_all.edf", "fs_in": fs_acc, "fs_out": acc_fs, "n": len(mag)}

    if len(seq_parts) == 0:
        raise RuntimeError(f"No modalities extracted for {patient_dir}")

    # concat along feature dim
    X_seq = np.concatenate(seq_parts, axis=1).astype(np.float32)  # (T, F)
    return X_seq, feat_names, meta


# -------------------------
# Build dataset
# -------------------------
def build_dataset(root_dir: str,
                  label_csv: str,
                  out_path: str,
                  id_col: Optional[str],
                  max_minutes: int,
                  eeg_fs: float,
                  cardio_fs: float,
                  acc_fs: float,
                  prefer_ear_eeg: bool,
                  truncate: str = "first"):
    df = pd.read_csv(label_csv)

    # ID column: first column by default
    if id_col is None:
        id_col = df.columns[0]
    if id_col not in df.columns:
        raise ValueError(f"id_col={id_col} not in label.csv columns")

    label_cols = [c for c in df.columns if c != id_col]
    if len(label_cols) == 0:
        raise ValueError("label.csv must have at least 1 label column besides id column")

    # Convert labels to numeric
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing id
    df = df.dropna(subset=[id_col])
    df[id_col] = df[id_col].astype(str).str.strip()

    # Drop rows where all labels are NaN
    df = df.dropna(subset=label_cols, how="all").reset_index(drop=True)

    X_list = []
    y_list = []
    mask_list = []
    ids = []
    metas = []

    feat_names_ref = None

    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        pdir = os.path.join(root_dir, pid)
        if not os.path.isdir(pdir):
            log(f"[SKIP] {pid}: folder not found: {pdir}")
            continue

        y = row[label_cols].values.astype(np.float32)
        if np.any(np.isnan(y)):
            log(f"[SKIP] {pid}: labels contain NaN -> {dict(zip(label_cols, y))}")
            continue

        try:
            X_seq, feat_names, meta = extract_subject_sequence(
                pdir, eeg_fs=eeg_fs, cardio_fs=cardio_fs, acc_fs=acc_fs,
                max_minutes=max_minutes, prefer_ear_eeg=prefer_ear_eeg
            )
        except Exception as e:
            log(f"[SKIP] {pid}: failed to extract: {e}")
            continue

        # Feature name consistency
        if feat_names_ref is None:
            feat_names_ref = feat_names
        else:
            if feat_names != feat_names_ref:
                raise RuntimeError(
                    f"Feature names mismatch at {pid}. "
                    f"This usually means channels differ across subjects. "
                    f"Fix by enforcing same channel set."
                )

        T, F = X_seq.shape
        T_max = max_minutes
        if T > T_max:
            if truncate == "first":
                X_cut = X_seq[:T_max]
            elif truncate == "center":
                start = (T - T_max) // 2
                X_cut = X_seq[start:start+T_max]
            else:
                X_cut = X_seq[:T_max]
            m = np.ones((T_max,), dtype=np.float32)
        else:
            X_cut = np.zeros((T_max, F), dtype=np.float32)
            X_cut[:T] = X_seq
            m = np.zeros((T_max,), dtype=np.float32)
            m[:T] = 1.0

        X_list.append(X_cut)
        y_list.append(y)
        mask_list.append(m)
        ids.append(pid)
        meta["patient_id"] = pid
        metas.append(meta)

        log(f"[OK] {pid}: X_seq={X_seq.shape} -> padded {X_cut.shape}, labels={dict(zip(label_cols, y))}")

    if len(X_list) == 0:
        raise RuntimeError("No subjects built. Check root_dir structure and label.csv.")

    X = np.stack(X_list, axis=0).astype(np.float32)      # (N, T_max, F)
    y = np.stack(y_list, axis=0).astype(np.float32)      # (N, K)
    mask = np.stack(mask_list, axis=0).astype(np.float32) # (N, T_max)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(out_path,
                        X=X,
                        y=y,
                        mask=mask,
                        subject_ids=np.array(ids),
                        label_names=np.array(label_cols),
                        feature_names=np.array(feat_names_ref))

    with open(os.path.join(out_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2)

    log("\n[DONE]")
    log(f"  Saved: {out_path}")
    log(f"  X: {X.shape}  y: {y.shape}  mask: {mask.shape}")
    log(f"  labels: {label_cols}")
    log(f"  features: {len(feat_names_ref)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=False,default="./questionare", help="root folder containing patient subfolders")
    ap.add_argument("--label_csv", required=False, default="./questionare/label.csv", help="label.csv (first col is folder name)")
    ap.add_argument("--out_path", default="./nn_dataset_subject_level.npz")
    ap.add_argument("--id_col", default=None, help="folder-id column name (default: first col)")
    ap.add_argument("--max_minutes", type=int, default=600, help="pad/trim minutes (e.g., 600=10h)")
    ap.add_argument("--eeg_fs", type=float, default=100.0)
    ap.add_argument("--cardio_fs", type=float, default=50.0)
    ap.add_argument("--acc_fs", type=float, default=25.0)
    ap.add_argument("--prefer_ear_eeg", action="store_true", help="use Ear EEG_200.edf if exists")
    ap.add_argument("--truncate", choices=["first", "center"], default="first")
    args = ap.parse_args()

    build_dataset(root_dir=args.root_dir,
                  label_csv=args.label_csv,
                  out_path=args.out_path,
                  id_col=args.id_col,
                  max_minutes=args.max_minutes,
                  eeg_fs=args.eeg_fs,
                  cardio_fs=args.cardio_fs,
                  acc_fs=args.acc_fs,
                  prefer_ear_eeg=args.prefer_ear_eeg,
                  truncate=args.truncate)


if __name__ == "__main__":
    main()
