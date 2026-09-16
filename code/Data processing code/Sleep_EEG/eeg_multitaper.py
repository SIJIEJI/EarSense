#!/usr/bin/env python3
"""Compare sample-aligned PSG and ear-EEG using multitaper spectra.

Input text files: column 0 = time in seconds, column 1 = EEG in microvolts.
Correlations are descriptive; no pseudo-replicated p-values are reported.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy import signal

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), "beta": (16.0, 30.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("psg", type=Path)
    p.add_argument("ear", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("multitaper_results"))
    p.add_argument("--target-fs", type=float, default=200.0)
    p.add_argument("--cut-first", type=float, default=0.0)
    p.add_argument("--notch", type=float, default=None)
    p.add_argument("--win-sec", type=float, default=4.0)
    p.add_argument("--hop-sec", type=float, default=2.0)
    p.add_argument("--bandwidth", type=float, default=2.0)
    p.add_argument("--smooth-sec", type=float, default=120.0)
    p.add_argument("--export-spectrogram-csv", action="store_true")
    return p.parse_args()


def load_eeg(path: Path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected time and EEG columns")
    t, x = data[:, 0].astype(float), data[:, 1].astype(float)
    if len(t) < 3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: data contain NaN/Inf or are too short")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt
    jitter = np.max(np.abs(dt - median_dt))
    if jitter > max(0.05 * median_dt, 1e-6):
        raise ValueError(f"{path}: irregular timestamps or dropped samples; max dt error={jitter:.6g}s")
    return t, x, fs


def align_and_validate(t1, x1, fs1, t2, x2, fs2):
    if not np.isclose(fs1, fs2, rtol=1e-4, atol=1e-6):
        raise ValueError(f"Sampling rates differ: PSG={fs1:.9g}, EAR={fs2:.9g} Hz")
    n = min(len(t1), len(t2))
    tolerance = max(0.1 / fs1, 1e-6)
    relative1, relative2 = t1[:n] - t1[0], t2[:n] - t2[0]
    error = np.max(np.abs(relative1 - relative2))
    if error > tolerance:
        raise ValueError(f"Signals are not sample-aligned; maximum relative-time error={error:.6g}s")
    return relative1, x1[:n], x2[:n], float((fs1 + fs2) / 2)


def resample_pair(t, x1, x2, fs_old, fs_new):
    if np.isclose(fs_old, fs_new, rtol=1e-8, atol=1e-8):
        return t, x1, x2, fs_old
    ratio = Fraction(float(fs_new / fs_old)).limit_denominator(100_000)
    y1 = signal.resample_poly(x1, ratio.numerator, ratio.denominator)
    y2 = signal.resample_poly(x2, ratio.numerator, ratio.denominator)
    n = min(len(y1), len(y2))
    return np.arange(n) / fs_new, y1[:n], y2[:n], fs_new


def preprocess(x, fs, band=(0.5, 30.0), notch=None):
    x = signal.detrend(np.asarray(x, float), type="constant")
    if notch is not None:
        if not 0 < notch < fs / 2:
            raise ValueError("Notch frequency must be below Nyquist")
        sos = signal.tf2sos(*signal.iirnotch(notch, Q=30.0, fs=fs))
        x = signal.sosfiltfilt(sos, x)
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def make_windows(x, fs, win_sec, hop_sec):
    win, hop = int(round(win_sec * fs)), int(round(hop_sec * fs))
    if win < 16 or hop < 1 or len(x) < win:
        raise ValueError("Invalid window/hop or recording shorter than one window")
    starts = np.arange(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts]), starts


def window_qc(w1, w2):
    """Reject nonfinite, flat, or robustly extreme RMS/peak-to-peak windows."""
    finite = np.all(np.isfinite(w1), axis=1) & np.all(np.isfinite(w2), axis=1)
    def channel_qc(w):
        rms = np.sqrt(np.mean(w ** 2, axis=1))
        ptp = np.ptp(w, axis=1)
        ok = (rms > np.finfo(float).eps) & (ptp > np.finfo(float).eps)
        for feature in (np.log(rms + 1e-30), np.log(ptp + 1e-30)):
            med = np.median(feature[ok]) if ok.any() else np.nan
            mad = 1.4826 * np.median(np.abs(feature[ok] - med)) if ok.any() else np.nan
            if np.isfinite(mad) and mad > 0:
                ok &= np.abs(feature - med) <= 5 * mad
        return ok
    return finite & channel_qc(w1) & channel_qc(w2)


def multitaper(windows_uV, fs, fmin, fmax, bandwidth):
    """MNE adaptive multitaper PSD in V^2/Hz, frequency x time-window."""
    psd, freqs = psd_array_multitaper(
        windows_uV * 1e-6,
        sfreq=float(fs),
        fmin=float(fmin),
        fmax=float(fmax),
        bandwidth=float(bandwidth),
        adaptive=True,
        low_bias=True,
        normalization="full",
        verbose=False,
    )
    return freqs, psd.T


def corr(a, b, min_n=10):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def centered_smooth(x, bins):
    if bins <= 1:
        return np.asarray(x, float)
    return pd.Series(x).rolling(bins, center=True, min_periods=bins).mean().to_numpy()


def bandpower(psd, freqs, band):
    index = (freqs >= band[0]) & (freqs < band[1])
    return np.trapezoid(psd[index], x=freqs[index], axis=0)


def band_traces(psd, freqs, smooth_bins):
    total = bandpower(psd, freqs, (0.5, 30.0))
    out = {}
    for name, limits in BANDS.items():
        power = bandpower(psd, freqs, limits)
        out[f"{name}_absolute_dB"] = centered_smooth(10 * np.log10(power + 1e-30), smooth_bins)
        out[f"{name}_relative_dB"] = centered_smooth(10 * np.log10(power / (total + 1e-30) + 1e-30), smooth_bins)
    out["total_absolute_dB"] = centered_smooth(10 * np.log10(total + 1e-30), smooth_bins)
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t1, x1, fs1 = load_eeg(args.psg)
    t2, x2, fs2 = load_eeg(args.ear)
    t, x1, x2, fs = align_and_validate(t1, x1, fs1, t2, x2, fs2)
    if args.cut_first < 0 or args.cut_first >= t[-1]:
        raise ValueError("--cut-first must be nonnegative and shorter than the recording")
    cut = int(np.searchsorted(t, args.cut_first, side="left"))
    t, x1, x2 = t[cut:] - t[cut], x1[cut:], x2[cut:]
    t, x1, x2, fs = resample_pair(t, x1, x2, fs, args.target_fs)
    x1, x2 = preprocess(x1, fs, notch=args.notch), preprocess(x2, fs, notch=args.notch)

    w1, starts = make_windows(x1, fs, args.win_sec, args.hop_sec)
    w2, starts2 = make_windows(x2, fs, args.win_sec, args.hop_sec)
    if not np.array_equal(starts, starts2):
        raise RuntimeError("Internal window alignment failure")
    valid = window_qc(w1, w2)
    times = (starts + w1.shape[1] / 2) / fs
    freqs, s1 = multitaper(w1, fs, 0.5, 30.0, args.bandwidth)
    freqs2, s2 = multitaper(w2, fs, 0.5, 30.0, args.bandwidth)
    if not np.allclose(freqs, freqs2):
        raise RuntimeError("PSG and ear-EEG frequency grids differ")
    s1[:, ~valid], s2[:, ~valid] = np.nan, np.nan
    log1, log2 = 10 * np.log10(s1 + 1e-30), 10 * np.log10(s2 + 1e-30)

    spectral_shape_r = np.array([corr(log1[:, i], log2[:, i], 6) for i in range(len(times))])
    temporal_r = np.array([corr(log1[k], log2[k]) for k in range(len(freqs))])
    smooth_bins = max(1, int(round(args.smooth_sec / args.hop_sec)))
    raw_traces1, raw_traces2 = band_traces(s1, freqs, 1), band_traces(s2, freqs, 1)
    traces1, traces2 = band_traces(s1, freqs, smooth_bins), band_traces(s2, freqs, smooth_bins)

    summary = {
        "valid_windows": int(valid.sum()), "total_windows": len(valid),
        "valid_window_fraction": float(valid.mean()),
        "overall_linear_psd_r_descriptive": corr(s1.ravel(), s2.ravel()),
        "overall_log_psd_r_descriptive": corr(log1.ravel(), log2.ravel()),
        "spectral_shape_r_time_mean": float(np.nanmean(spectral_shape_r)),
        "spectral_shape_r_time_median": float(np.nanmedian(spectral_shape_r)),
        "temporal_r_frequency_mean": float(np.nanmean(temporal_r)),
        "temporal_r_frequency_median": float(np.nanmedian(temporal_r)),
    }
    for name, limits in BANDS.items():
        index = (freqs >= limits[0]) & (freqs < limits[1])
        summary[f"flattened_log_psd_r_{name}_descriptive"] = corr(log1[index].ravel(), log2[index].ravel())
    for key in traces1:
        summary[f"timecourse_r_{key}_raw"] = corr(raw_traces1[key], raw_traces2[key])
        summary[f"timecourse_r_{key}_smoothed"] = corr(traces1[key], traces2[key])
    pd.DataFrame([summary]).to_csv(args.out_dir / "correlation_summary.csv", index=False)

    timecourse = {"time_s": times, "valid_window": valid, "spectral_shape_r": spectral_shape_r}
    for key in traces1:
        timecourse[f"PSG_{key}_raw"] = raw_traces1[key]
        timecourse[f"EAR_{key}_raw"] = raw_traces2[key]
        timecourse[f"PSG_{key}"] = traces1[key]
        timecourse[f"EAR_{key}"] = traces2[key]
    pd.DataFrame(timecourse).to_csv(args.out_dir / "bandpower_timecourses.csv", index=False)
    pd.DataFrame({"frequency_Hz": freqs, "temporal_r": temporal_r}).to_csv(
        args.out_dir / "correlation_by_frequency.csv", index=False)

    settings = vars(args).copy()
    settings.update({"measured_input_fs": fs1, "analysis_fs": fs, "smooth_bins": smooth_bins})
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    np.savez_compressed(args.out_dir / "spectrograms_and_correlations.npz",
                        frequencies_Hz=freqs, times_s=times, PSG_psd_V2_per_Hz=s1,
                        EAR_psd_V2_per_Hz=s2, valid_window=valid,
                        spectral_shape_r_by_time=spectral_shape_r,
                        temporal_r_by_frequency=temporal_r,
                        settings_json=json.dumps(settings), summary_json=json.dumps(summary))
    if args.export_spectrogram_csv:
        pd.DataFrame(s1, index=freqs, columns=times).to_csv(args.out_dir / "PSG_multitaper_PSD_V2_per_Hz.csv")
        pd.DataFrame(s2, index=freqs, columns=times).to_csv(args.out_dir / "EAR_multitaper_PSD_V2_per_Hz.csv")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
