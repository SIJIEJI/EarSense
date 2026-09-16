#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-subject sleep dataset builder (data processing only)

What it does:
- Scan root directory: each subfolder = 1 subject
- In each subject folder: find one EDF + one CSV ("Sleep stage.csv")
- Align CSV timestamps to EDF start/end time (handle midnight rollover)
- EDF is record-based (typically 30s per record): build one sample per record
- For each sample: multi-modal channels -> resample to target_fs -> bandpass -> z-score
- Drop records without labels
- Save per-subject cache + one merged dataset:
    out_dir/
      subjects/<subject_name>/
        X.npy, y.npy, meta.json
      merged/
        X.npy, y.npy, sid.npy, meta.json

Dependencies:
  pip install numpy pandas scipy
"""

import os
import re
import json
import math
import datetime as dt
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly


# -------------------------
# EDF minimal reader
# -------------------------
def _safe_int(s: str, default: int = 0) -> int:
    s = (s or "").strip()
    try:
        return int(float(s))
    except Exception:
        return default


def _safe_float(s: str, default: float = 0.0) -> float:
    s = (s or "").strip()
    try:
        return float(s)
    except Exception:
        return default


def read_edf_header(path: str) -> dict:
    """Minimal EDF header parser (int16 typical EDF)."""
    with open(path, "rb") as f:
        fixed = f.read(256)
        if len(fixed) != 256:
            raise ValueError("File too small to be a valid EDF.")

        start_date = fixed[168:176].decode("ascii", errors="ignore").strip()  # dd.mm.yy
        start_time = fixed[176:184].decode("ascii", errors="ignore").strip()  # hh.mm.ss
        header_bytes = _safe_int(fixed[184:192].decode("ascii", errors="ignore"), default=0)
        n_records = _safe_int(fixed[236:244].decode("ascii", errors="ignore"), default=-1)
        record_duration = _safe_float(fixed[244:252].decode("ascii", errors="ignore"), default=1.0)
        n_signals = _safe_int(fixed[252:256].decode("ascii", errors="ignore"), default=0)

        if n_signals <= 0 or header_bytes <= 0:
            raise ValueError("Invalid EDF header: n_signals/header_bytes look wrong.")

        def read_field(field_len: int):
            return [f.read(field_len).decode("ascii", errors="ignore").strip() for _ in range(n_signals)]

        labels = read_field(16)
        _transducer = read_field(80)
        phys_dim = read_field(8)
        phys_min = np.array([_safe_float(x, 0.0) for x in read_field(8)], dtype=float)
        phys_max = np.array([_safe_float(x, 0.0) for x in read_field(8)], dtype=float)
        dig_min = np.array([_safe_int(x, 0) for x in read_field(8)], dtype=int)
        dig_max = np.array([_safe_int(x, 0) for x in read_field(8)], dtype=int)
        _prefilter = read_field(80)
        spr = np.array([_safe_int(x, 0) for x in read_field(8)], dtype=int)  # samples per record
        _sig_reserved = read_field(32)

    # Parse start datetime
    try:
        day, month, yy = [int(x) for x in start_date.split(".")]
        year = 1900 + yy if yy >= 85 else 2000 + yy
        hh, mm, ss = [int(x) for x in start_time.split(".")]
        start_datetime = dt.datetime(year, month, day, hh, mm, ss)
    except Exception:
        start_datetime = None

    # If n_records unknown, infer from file size
    if n_records <= 0:
        file_size = os.path.getsize(path)
        bytes_per_record = int(spr.sum() * 2)  # int16 => 2 bytes
        n_records = int((file_size - header_bytes) // bytes_per_record)

    duration_sec = n_records * record_duration
    sfreq = spr / record_duration

    denom = (dig_max - dig_min).astype(float)
    denom[denom == 0] = 1e-12
    scale = (phys_max - phys_min) / denom
    offset = phys_min - scale * dig_min

    return dict(
        start_datetime=start_datetime,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration=record_duration,
        n_signals=n_signals,
        labels=labels,
        spr=spr,
        sfreq=sfreq,
        duration_sec=duration_sec,
        scale=scale,
        offset=offset,
        phys_dim=phys_dim,
    )


def iter_edf_records(path: str, hdr: dict, select_indices: list[int]):
    """Yield per EDF record for selected channels only."""
    spr = hdr["spr"]
    bytes_per_record = int(spr.sum() * 2)
    header_bytes = hdr["header_bytes"]
    labels = hdr["labels"]
    sfreqs = hdr["sfreq"]
    scale = hdr["scale"]
    offset = hdr["offset"]

    select_set = set(select_indices)

    with open(path, "rb") as f:
        f.seek(header_bytes)
        for rec in range(hdr["n_records"]):
            buf = f.read(bytes_per_record)
            if len(buf) != bytes_per_record:
                raise RuntimeError("Unexpected EOF while reading EDF records.")

            ptr = 0
            out = {}
            for ch_i in range(hdr["n_signals"]):
                n_s = int(spr[ch_i])
                byte_len = n_s * 2
                if ch_i in select_set:
                    x_i16 = np.frombuffer(buf, dtype="<i2", count=n_s, offset=ptr)
                    x = x_i16.astype(np.float32) * float(scale[ch_i]) + float(offset[ch_i])
                    out[labels[ch_i]] = (x, float(sfreqs[ch_i]))
                ptr += byte_len
            yield rec, out


# -------------------------
# CSV parsing + alignment
# -------------------------
def _find_time_col(cols):
    for c in cols:
        if "hh:mm:ss" in c.lower():
            return c
    for c in cols:
        if "absolute" in c.lower() and "position" in c.lower():
            return c
    return None


def _find_stage_col(cols):
    for c in cols:
        if "default staging set" in c.lower() and "stage" in c.lower():
            return c
    for c in cols:
        if "stage" in c.lower():
            return c
    return None


def parse_csv_timestamps(csv_path: str, base_date: dt.date) -> pd.DataFrame:
    """
    CSV has time-of-day only. Rebuild absolute datetime using base_date and midnight rollover.
    Expected time format: HH:MM:SS.mmm
    """
    df = pd.read_csv(csv_path)
    time_col = _find_time_col(df.columns)
    stage_col = _find_stage_col(df.columns)
    if time_col is None:
        raise ValueError(f"Cannot find time column in CSV: {csv_path}")
    if stage_col is None:
        raise ValueError(f"Cannot find stage column in CSV: {csv_path}")

    ts = []
    cur_date = base_date
    prev_sec = None

    for s in df[time_col].astype(str).tolist():
        m = re.match(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d{1,3})\s*$", s)
        if not m:
            raise ValueError(f"Bad time format in CSV: {s} ({csv_path})")
        hh, mm, ss, ms = map(int, m.groups())
        sec = hh * 3600 + mm * 60 + ss + ms / 1000.0
        if prev_sec is not None and sec < prev_sec - 1.0:
            cur_date = cur_date + dt.timedelta(days=1)
        prev_sec = sec
        ts.append(dt.datetime.combine(cur_date, dt.time(hh, mm, ss, ms * 1000)))

    df = df.copy()
    df["ts"] = ts
    df["_time_col"] = time_col
    df["_stage_col"] = stage_col
    return df


def build_record_labels(df_ts: pd.DataFrame, edf_start: dt.datetime, edf_end: dt.datetime, record_duration: float):
    """One label per EDF record window by mode label within the window."""
    stage_col = df_ts["_stage_col"].iloc[0]
    df_aligned = df_ts[(df_ts["ts"] >= edf_start) & (df_ts["ts"] < edf_end)].copy()

    n_records = int(math.floor((edf_end - edf_start).total_seconds() / record_duration))
    labels = []
    for i in range(n_records):
        s = edf_start + dt.timedelta(seconds=i * record_duration)
        e = s + dt.timedelta(seconds=record_duration)
        seg = df_aligned[(df_aligned["ts"] >= s) & (df_aligned["ts"] < e)][stage_col]
        labels.append(seg.value_counts().idxmax() if len(seg) else None)
    return labels, df_aligned


# -------------------------
# Preprocess
# -------------------------
def make_sos(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    low = max(low_hz, 1e-4) / nyq
    high = min(high_hz, nyq - 1e-4) / nyq
    return butter(order, [low, high], btype="band", output="sos")


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample 1D signal x from fs_in to fs_out using polyphase resampling.
    Works with float sampling rates in Python 3.11+.
    """
    fs_in = float(fs_in)
    fs_out = float(fs_out)

    if not np.isfinite(fs_in) or fs_in <= 0:
        raise ValueError(f"Invalid fs_in={fs_in}. Check EDF header/channel sampling rate.")
    if not np.isfinite(fs_out) or fs_out <= 0:
        raise ValueError(f"Invalid fs_out={fs_out}. target_fs must be > 0.")

    if abs(fs_in - fs_out) < 1e-9:
        return x

    ratio = fs_out / fs_in
    frac = Fraction(ratio).limit_denominator(2000)  # <-- single-arg Fraction OK for float
    up, down = frac.numerator, frac.denominator

    return resample_poly(x, up, down)

# def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
#     if abs(fs_in - fs_out) < 1e-6:
#         return x
#     frac = Fraction(fs_out, fs_in).limit_denominator(2000)
#     return resample_poly(x, frac.numerator, frac.denominator)


def preprocess_window(x: np.ndarray, sos, target_len: int) -> np.ndarray:
    # pad/trim
    if x.shape[0] > target_len:
        x = x[:target_len]
    elif x.shape[0] < target_len:
        x = np.pad(x, (0, target_len - x.shape[0]), mode="edge")

    # filter
    try:
        x = sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        x = x.astype(np.float32)

    # per-window z-score
    mu = float(x.mean())
    sd = float(x.std()) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}


def _pick_channel(labels: list[str], candidates: list[str]) -> str:
    """Pick the first exact match, else substring case-insensitive match."""
    lab_set = set(labels)
    for c in candidates:
        if c in lab_set:
            return c
    lower = [x.lower() for x in labels]
    for c in candidates:
        cl = c.lower()
        for i, lab in enumerate(lower):
            if cl in lab:
                return labels[i]
    return ""


def find_first_match(folder: str, patterns: list[str]) -> str:
    """
    Find the first filename matching any regex pattern (case-insensitive) inside folder.
    """
    files = os.listdir(folder)
    for pat in patterns:
        r = re.compile(pat, flags=re.IGNORECASE)
        for fn in files:
            if r.fullmatch(fn) or r.search(fn):
                return os.path.join(folder, fn)
    return ""


def build_one_subject(edf_path: str, csv_path: str, target_fs: int = 100):
    """
    Build one-subject dataset:
      X: (N_records, C, T)
      y: (N_records,)
      meta: dict
    """
    hdr = read_edf_header(edf_path)
    if hdr["start_datetime"] is None:
        raise ValueError(f"EDF start time could not be parsed: {edf_path}")

    edf_start = hdr["start_datetime"]
    edf_end = edf_start + dt.timedelta(seconds=float(hdr["duration_sec"]))

    df_ts = parse_csv_timestamps(csv_path, base_date=edf_start.date())
    record_labels_raw, df_aligned = build_record_labels(
        df_ts, edf_start=edf_start, edf_end=edf_end, record_duration=float(hdr["record_duration"])
    )

    labels = hdr["labels"]

    # Robust channel selection
    ch_ecg = _pick_channel(labels, ["ECG", "ECG I", "ECG1", "ECG_1"])
    ch_eeg = _pick_channel(labels, ["Ear EEG", "EEG", "EEG1", "EEG_1"])
    ch_ppg_g = _pick_channel(labels, ["Raw PPG_Green", "PPG_Green", "PPG Green"])
    ch_ppg_r = _pick_channel(labels, ["Raw PPG_Red", "PPG_Red", "PPG Red"])
    ch_ppg_ir = _pick_channel(labels, ["Raw PPG_IR", "PPG_IR", "PPG IR"])
    ch_ax = _pick_channel(labels, ["Acce_Ax", "AccX", "Acc X", "ACCX"])
    ch_ay = _pick_channel(labels, ["Acce_Ay", "AccY", "Acc Y", "ACCY"])
    ch_az = _pick_channel(labels, ["Acce_Az", "AccZ", "Acc Z", "ACCZ"])

    wanted = [ch_ecg, ch_eeg, ch_ppg_g, ch_ppg_r, ch_ppg_ir, ch_ax, ch_ay, ch_az]
    if any(x == "" for x in wanted):
        missing = [name for name, picked in zip(
            ["ECG", "EEG", "PPG_G", "PPG_R", "PPG_IR", "AccX", "AccY", "AccZ"], wanted
        ) if picked == ""]
        raise ValueError(f"Missing required channels: {missing}\nEDF labels: {labels}")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    select_indices = [label_to_idx[w] for w in wanted]

    record_sec = float(hdr["record_duration"])
    target_len = int(target_fs * record_sec)

    # Filters designed at target_fs
    sos_by = {
        wanted[0]: make_sos(0.5, 40, target_fs),   # ECG
        wanted[1]: make_sos(0.5, 45, target_fs),   # EEG
        wanted[2]: make_sos(0.5, 8, target_fs),    # PPG
        wanted[3]: make_sos(0.5, 8, target_fs),    # PPG
        wanted[4]: make_sos(0.5, 8, target_fs),    # PPG
        wanted[5]: make_sos(0.1, 15, target_fs),   # ACC
        wanted[6]: make_sos(0.1, 15, target_fs),   # ACC
        wanted[7]: make_sos(0.1, 15, target_fs),   # ACC
    }

    N = hdr["n_records"]
    C = len(wanted)
    X = np.zeros((N, C, target_len), dtype=np.float32)
    y = np.full((N,), -1, dtype=np.int64)

    for rec, out in iter_edf_records(edf_path, hdr, select_indices):
        for c, ch_name in enumerate(wanted):
            x_raw, fs_in = out[ch_name]
            x_rs = resample_to(x_raw, fs_in=float(fs_in), fs_out=float(target_fs)).astype(np.float32)
            X[rec, c, :] = preprocess_window(x_rs, sos=sos_by[ch_name], target_len=target_len)

        st = record_labels_raw[rec] if rec < len(record_labels_raw) else None
        y[rec] = STAGE_MAP.get(str(st).strip(), -1)

    # Drop unlabeled
    keep = np.where(y >= 0)[0]
    X = X[keep]
    y = y[keep]

    counts = Counter(y.tolist())
    meta = {
        "edf": edf_path,
        "csv": csv_path,
        "edf_start": edf_start.isoformat(),
        "edf_end": edf_end.isoformat(),
        "record_duration_sec": record_sec,
        "target_fs": target_fs,
        "channels": wanted,
        "csv_time_col": df_ts["_time_col"].iloc[0],
        "csv_stage_col": df_ts["_stage_col"].iloc[0],
        "csv_start": df_ts["ts"].min().isoformat(),
        "csv_end": df_ts["ts"].max().isoformat(),
        "aligned_csv_rows": int(len(df_aligned)),
        "n_labeled_records": int(len(y)),
        "label_counts": dict(counts),
        "missing_classes": [k for k in range(5) if k not in counts],
    }
    return X, y, meta


def build_all_subjects(root_dir: str, out_dir: str, target_fs: int = 100):
    """
    Build merged dataset from your structure:
      root_dir/
        25_09_15_XXX/
          Sleep_all.edf
          Sleep stage.csv
        ...
    Output:
      out_dir/subjects/<name>/X.npy, y.npy, meta.json
      out_dir/merged/X.npy, y.npy, sid.npy, meta.json
    """
    os.makedirs(out_dir, exist_ok=True)
    subjects_dir = os.path.join(out_dir, "subjects")
    merged_dir = os.path.join(out_dir, "merged")
    os.makedirs(subjects_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    subfolders = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    if len(subfolders) == 0:
        raise ValueError(f"No subject folders found under {root_dir}")

    X_list, y_list, sid_list = [], [], []
    subject_metas = []

    sid = 0
    for sub in subfolders:
        subj_name = os.path.basename(sub).strip()
        subj_out = os.path.join(subjects_dir, subj_name)
        os.makedirs(subj_out, exist_ok=True)

        # If cached, load
        x_path = os.path.join(subj_out, "X.npy")
        y_path = os.path.join(subj_out, "y.npy")
        m_path = os.path.join(subj_out, "meta.json")

        if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(m_path):
            X = np.load(x_path, mmap_mode=None)
            y = np.load(y_path, mmap_mode=None)
            with open(m_path, "r") as f:
                meta = json.load(f)
            print(f"[OK] Loaded cache for {subj_name}: {X.shape}, labels={Counter(y.tolist())}")
        else:
            # Find EDF/CSV (robust patterns)
            edf_path = find_first_match(sub, [r".*sleep[_\s-]*all.*\.edf$", r".*\.edf$"])
            csv_path = find_first_match(sub, [r".*sleep\s*stage.*\.csv$", r".*stage.*\.csv$"])
            if not edf_path or not csv_path:
                print(f"[SKIP] {subj_name}: EDF/CSV not found")
                continue

            X, y, meta = build_one_subject(edf_path, csv_path, target_fs=target_fs)

            np.save(x_path, X.astype(np.float32))
            np.save(y_path, y.astype(np.int64))
            with open(m_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"[OK] Built {subj_name}: X={X.shape}, labels={Counter(y.tolist())}")

        # append to merged
        X_list.append(X)
        y_list.append(y)
        sid_list.append(np.full((len(y),), sid, dtype=np.int64))

        subject_metas.append({
            "sid": sid,
            "name": subj_name,
            "n_samples": int(len(y)),
            "label_counts": dict(Counter(y.tolist())),
            "missing_classes": meta.get("missing_classes", []),
            "edf_start": meta.get("edf_start", None),
            "edf_end": meta.get("edf_end", None),
        })
        sid += 1

    if len(X_list) == 0:
        raise ValueError("No subjects were successfully processed.")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)
    sid_all = np.concatenate(sid_list, axis=0).astype(np.int64)

    meta_all = {
        "root_dir": root_dir,
        "out_dir": out_dir,
        "target_fs": target_fs,
        "num_subjects": int(len(subject_metas)),
        "total_samples": int(len(y_all)),
        "global_label_counts": dict(Counter(y_all.tolist())),
        "subjects": subject_metas,
    }

    np.save(os.path.join(merged_dir, "X.npy"), X_all)
    np.save(os.path.join(merged_dir, "y.npy"), y_all)
    np.save(os.path.join(merged_dir, "sid.npy"), sid_all)
    with open(os.path.join(merged_dir, "meta.json"), "w") as f:
        json.dump(meta_all, f, indent=2)

    print(f"[DONE] merged X={X_all.shape}, y={y_all.shape}, sid={sid_all.shape}")
    print(f"[DONE] global label counts: {meta_all['global_label_counts']}")
    print(f"[DONE] saved to: {merged_dir}")
    return X_all, y_all, sid_all, meta_all


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=False, default="raw_sleep_edf", help="Root folder containing one subfolder per subject")
    p.add_argument("--out_dir", required=False, default="sleep_dataset_built", help="Output folder")
    p.add_argument("--target_fs", type=int, default=100)
    args = p.parse_args()

    build_all_subjects(args.root_dir, args.out_dir, target_fs=args.target_fs)
