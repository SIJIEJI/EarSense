#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOSO regression for PVT lapse-rate (bounded) with:
1) Point prediction (median model)
2) 95% prediction interval via Conformalized Quantile Regression (CQR)
3) Visualization: y=x diagonal + diagonal band + per-point 95% vertical error bars

Run:
  python PVTregression_CI_band.py \
    --npz nn_dataset_subject_level.npz \
    --label_xlsx ./sleep_questionnaires_scores_one_row_per_subject_class.xlsx \
    --id_col subject \
    --target_col lapse_rate_gt_500ms \
    --out_dir ./loso_lapse_hgb_cqr

Notes:
- Requires scikit-learn where HistGradientBoostingRegressor supports loss="quantile".
"""

import os
import re
import json
import argparse
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use("Agg") 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------
# label parsing
# -----------------------
def parse_lapse_value(v) -> float:
    """
    Robustly parse lapse_rate values possibly formatted as:
      - "1.52%"  -> 0.0152
      - "4.74"   -> 0.0474   (auto treat >1.0 as percent points)
      - 0.01515  -> 0.01515  (already ratio)
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan

    if isinstance(v, (int, np.integer, float, np.floating)):
        x = float(v)
        if x > 1.0:  # likely percent points
            x = x / 100.0
        return x

    s = str(v).strip().replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if not m:
        return np.nan

    x = float(m.group(1))
    if "%" in s or x > 1.0:
        x = x / 100.0
    return x


# -----------------------
# feature extraction
# -----------------------
def _masked_valid(x: np.ndarray, m: np.ndarray):
    mv = m.astype(bool)
    if mv.sum() == 0:
        return None
    return x[mv]


def extract_subject_features(x_tf: np.ndarray, m_t: np.ndarray) -> np.ndarray:
    """
    x_tf: (T,F), m_t: (T,)
    Per-feature stats over time:
      mean, std, q10, q50, q90,
      mean_abs_diff, std_diff,
      half_mean_diff (first_half_mean - second_half_mean)
    Total dims = 8 * F
    """
    x_valid = _masked_valid(x_tf, m_t)
    T, F = x_tf.shape

    if x_valid is None or x_valid.shape[0] < 5:
        return np.zeros((8 * F,), dtype=np.float32)

    mu = x_valid.mean(axis=0)
    sd = x_valid.std(axis=0) + 1e-6
    q10 = np.quantile(x_valid, 0.10, axis=0)
    q50 = np.quantile(x_valid, 0.50, axis=0)
    q90 = np.quantile(x_valid, 0.90, axis=0)

    dx = np.diff(x_valid, axis=0)
    mean_abs_diff = np.mean(np.abs(dx), axis=0)
    std_diff = np.std(dx, axis=0) + 1e-6

    mid = T // 2
    x1 = _masked_valid(x_tf[:mid], m_t[:mid])
    x2 = _masked_valid(x_tf[mid:], m_t[mid:])
    if x1 is None or x2 is None or x1.shape[0] < 2 or x2.shape[0] < 2:
        half_diff = np.zeros((F,), dtype=np.float32)
    else:
        half_diff = x1.mean(axis=0) - x2.mean(axis=0)

    feat = np.concatenate(
        [mu, sd, q10, q50, q90, mean_abs_diff, std_diff, half_diff],
        axis=0
    ).astype(np.float32)
    return feat


# -----------------------
# bounded regression via logit
# -----------------------
def y_to_z(y, ymax=0.05, eps=1e-4):
    y = np.asarray(y, dtype=float)
    y01 = y / ymax
    y01 = np.clip(y01, eps, 1.0 - eps)
    return np.log(y01 / (1.0 - y01))


def z_to_y(z, ymax=0.05):
    z = np.asarray(z, dtype=float)
    y01 = 1.0 / (1.0 + np.exp(-z))
    return ymax * y01


# -----------------------
# plots
# -----------------------
def plot_scatter_with_interval_and_band(y_true, y_pred, y_lo, y_hi, out_path, title, band=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)

    plt.figure(figsize=(6.4, 5.8))
        # --- FIX: enforce valid intervals so yerr is non-negative ---
    # ensure lo <= pred <= hi
    y_lo = np.minimum(y_lo, y_hi)            # fix swapped bounds if any
    y_pred = np.clip(y_pred, y_lo, y_hi)     # force pred to be inside interval

    lower = np.maximum(0.0, y_pred - y_lo)   # non-negative
    upper = np.maximum(0.0, y_hi - y_pred)   # non-negative
    yerr = np.vstack([lower, upper])
    # vertical error bars
    yerr = np.vstack([y_pred - y_lo, y_hi - y_pred])
    plt.errorbar(y_true, y_pred, yerr=yerr, fmt='o', capsize=2)

    lo = float(min(y_true.min(), y_lo.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_hi.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad

    xs = np.linspace(lo, hi, 300)
    plt.plot(xs, xs)  # diagonal

    if band is not None and band > 0:
        plt.fill_between(xs, xs - band, xs + band, alpha=0.2)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True (ratio)")
    plt.ylabel("Pred (ratio)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_abs_error_bar(subject_ids, abs_err, out_path, thr=0.01):
    abs_err = np.asarray(abs_err, dtype=float)
    order = np.argsort(abs_err)[::-1]
    plt.figure(figsize=(11, 4.2))
    plt.bar(np.arange(len(abs_err)), abs_err[order])
    plt.axhline(thr, linestyle="--")
    plt.xticks(np.arange(len(abs_err)), np.array(subject_ids)[order], rotation=60, ha="right")
    plt.ylabel("|Pred-True| (ratio)")
    plt.title(f"Absolute Error per Subject (LOSO)  |  threshold={thr}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_error_cdf(abs_err, out_path, thr=0.01):
    abs_err = np.sort(np.asarray(abs_err, dtype=float))
    y = np.arange(1, len(abs_err) + 1) / len(abs_err)

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(abs_err, y, marker="o")
    plt.axvline(thr, linestyle="--")
    plt.xlabel("Absolute Error (ratio)")
    plt.ylabel("CDF")
    plt.title("Error CDF (LOSO)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------
# model factory
# -----------------------
def make_hgb(
    loss: str,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
    l2: float,
    min_samples_leaf: int,
    random_state: int,
    quantile: float = None
):
    kwargs = dict(
        loss=loss,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    if loss == "quantile":
        if quantile is None:
            raise ValueError("quantile must be provided when loss='quantile'")
        kwargs["quantile"] = float(quantile)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("hgb", HistGradientBoostingRegressor(**kwargs))
    ])


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", required=False,
                    default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--target_col", default="lapse_rate_gt_500ms")
    ap.add_argument("--out_dir", default="./loso_lapse_hgb_cqr")

    # bounds / transforms
    ap.add_argument("--ymax", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-4)

    # model params
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_iter", type=int, default=400)   # default slightly safer vs overfit
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--min_samples_leaf", type=int, default=5)

    # accuracy / reporting
    ap.add_argument("--thr", type=float, default=0.01, help="target abs error threshold")

    # interval (CQR)
    ap.add_argument("--alpha", type=float, default=0.05, help="miscoverage, 0.05 -> 95% interval")
    ap.add_argument("--calib_frac", type=float, default=0.25, help="fraction of train used for conformal calibration")

    # visualization band
    ap.add_argument("--diag_band", type=float, default=None,
                    help="width of diagonal band in y units; default uses --thr")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load npz
    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)          # (N,T,F)
    mask = d["mask"].astype(np.float32)    # (N,T)
    subject_ids = d["subject_ids"].astype(str)

    # Load labels
    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    if args.target_col not in df.columns:
        raise ValueError(f"{args.target_col} not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[args.target_col])
    label_map = df.set_index(args.id_col)

    # Align subjects
    keep = []
    y = []
    kept_subjects = []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            val = parse_lapse_value(label_map.loc[sid, args.target_col])
            if np.isnan(val):
                continue
            keep.append(i)
            y.append(val)
            kept_subjects.append(sid)

    keep = np.array(keep, dtype=np.int64)
    if len(keep) < 6:
        raise ValueError(f"Too few matched subjects: {len(keep)}. Check id alignment / NaNs.")

    X = X[keep]
    mask = mask[keep]
    subject_ids = np.array(kept_subjects, dtype=str)
    y = np.array(y, dtype=np.float32)

    N, T, Fdim = X.shape
    print("N:", N, "T:", T, "F:", Fdim, "y range:", float(y.min()), float(y.max()))

    # Suggest a safer ymax if needed
    if y.max() >= args.ymax:
        print(f"[WARN] y.max()={float(y.max()):.6f} >= ymax={args.ymax:.6f}. "
              f"Consider setting --ymax to > y.max(), e.g. {float(y.max()*1.1):.6f}")

    # Tabular features
    Xtab = np.stack([extract_subject_features(X[i], mask[i]) for i in range(N)], axis=0)

    # LOSO with CQR
    ymax = float(args.ymax)
    eps = float(args.eps)
    alpha = float(args.alpha)
    calib_frac = float(args.calib_frac)

    rng = np.random.RandomState(7)

    preds = np.zeros((N,), dtype=np.float32)
    lo95 = np.zeros((N,), dtype=np.float32)
    hi95 = np.zeros((N,), dtype=np.float32)

    for test_idx in range(N):
        train_idx = np.array([i for i in range(N) if i != test_idx], dtype=np.int64)

        # split train into fit/calibration for conformal
        rel = np.arange(len(train_idx))
        fit_rel, cal_rel = train_test_split(
            rel,
            test_size=calib_frac,
            random_state=int(rng.randint(0, 10_000))
        )
        fit_idx = train_idx[fit_rel]
        cal_idx = train_idx[cal_rel]

        # train targets in z-space (bounded regression)
        z_fit = y_to_z(y[fit_idx], ymax=ymax, eps=eps)

        # models
        m_med = make_hgb(
            loss="absolute_error",
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            l2=args.l2,
            min_samples_leaf=args.min_samples_leaf,
            random_state=7,
        )
        m_lo = make_hgb(
            loss="quantile",
            quantile=alpha / 2,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            l2=args.l2,
            min_samples_leaf=args.min_samples_leaf,
            random_state=7,
        )
        m_hi = make_hgb(
            loss="quantile",
            quantile=1 - alpha / 2,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            l2=args.l2,
            min_samples_leaf=args.min_samples_leaf,
            random_state=7,
        )

        # fit
        m_med.fit(Xtab[fit_idx], z_fit)
        m_lo.fit(Xtab[fit_idx], z_fit)
        m_hi.fit(Xtab[fit_idx], z_fit)

        # calibration: predicted interval on calibration set (y-space)
        z_cal_lo = m_lo.predict(Xtab[cal_idx])
        z_cal_hi = m_hi.predict(Xtab[cal_idx])
        y_cal_lo = z_to_y(z_cal_lo, ymax=ymax)
        y_cal_hi = z_to_y(z_cal_hi, ymax=ymax)
        y_cal_true = y[cal_idx].astype(float)

        # conformal score: how far y falls outside interval
        s = np.maximum(y_cal_lo - y_cal_true, y_cal_true - y_cal_hi)
        s = np.maximum(s, 0.0)

        # expansion amount
        qhat = float(np.quantile(s, 1 - alpha))

        # test prediction
        z_pred = float(m_med.predict(Xtab[[test_idx]])[0])
        z_pred_lo = float(m_lo.predict(Xtab[[test_idx]])[0])
        z_pred_hi = float(m_hi.predict(Xtab[[test_idx]])[0])

        y_pred = float(z_to_y(z_pred, ymax=ymax))
        y_lo = float(z_to_y(z_pred_lo, ymax=ymax)) - qhat
        y_hi = float(z_to_y(z_pred_hi, ymax=ymax)) + qhat

        # clamp to [0, ymax]
        y_lo = max(0.0, y_lo)
        y_hi = min(ymax, y_hi)

        preds[test_idx] = y_pred
        lo95[test_idx] = y_lo
        hi95[test_idx] = y_hi

    # Metrics
    abs_err = np.abs(preds - y)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds)) if N >= 3 else float("nan")
    pct_within = float(np.mean(abs_err <= args.thr))

    covered = (y >= lo95) & (y <= hi95)
    coverage = float(np.mean(covered))
    avg_width = float(np.mean(hi95 - lo95))

    metrics = {
        "N": int(N),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        f"pct_within_{args.thr}": pct_within,
        "coverage_95": coverage,
        "avg_interval_width": avg_width,
        "ymax": ymax,
        "model_point": "StandardScaler + HGBRegressor(loss=absolute_error) on logit(y/ymax)",
        "model_interval": "CQR: HGB quantile (alpha/2, 1-alpha/2) + conformal expansion",
        "cqr": {"alpha": alpha, "calib_frac": calib_frac},
        "hgb_params": {
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "l2_regularization": args.l2,
            "min_samples_leaf": args.min_samples_leaf
        }
    }

    # Save predictions
    out_csv = os.path.join(args.out_dir, "predictions_loso.csv")
    pd.DataFrame({
        "subject_id": subject_ids,
        "true": y,
        "pred": preds,
        "lo95": lo95,
        "hi95": hi95,
        "abs_error": abs_err,
        "covered": covered.astype(int)
    }).to_csv(out_csv, index=False)

    # Visualizations
    band = args.diag_band if args.diag_band is not None else args.thr
    scatter_path = os.path.join(args.out_dir, "scatter_true_vs_pred_with_ci.png")
    plot_scatter_with_interval_and_band(
        y, preds, lo95, hi95, scatter_path,
        title=f"LOSO CQR | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}  "
              f"P(|e|≤{args.thr})={pct_within:.3f}  Coverage95={coverage:.3f}",
        band=float(band)
    )

    bar_path = os.path.join(args.out_dir, "abs_error_bar.png")
    plot_abs_error_bar(subject_ids, abs_err, bar_path, thr=args.thr)

    cdf_path = os.path.join(args.out_dir, "error_cdf.png")
    plot_error_cdf(abs_err, cdf_path, thr=args.thr)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[DONE]")
    print(json.dumps(metrics, indent=2))
    print("Saved:", out_csv)
    print("Saved:", scatter_path)
    print("Saved:", bar_path)
    print("Saved:", cdf_path)


if __name__ == "__main__":
    main()