#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------
# label parsing
# -----------------------
def parse_lapse_value(v) -> float:
    """
    Robustly parse lapse_rate values possibly formatted as:
      - "1.52%"  -> 0.0152
      - "4.74"   -> 0.0474   (auto treat >1.0 as percent points)
      - 0.01515  -> 0.01515  (already ratio)
      - numeric  -> numeric (with auto conversion if >1.0)
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
    """x: (T,F), m:(T,) -> valid rows"""
    mv = m.astype(bool)
    if mv.sum() == 0:
        return None
    return x[mv]


def _q(v, q):
    return float(np.quantile(v, q)) if v.size else 0.0


def extract_subject_features(x_tf: np.ndarray, m_t: np.ndarray) -> np.ndarray:
    """
    Build strong tabular features for one subject from window-level features.
    x_tf: (T,F), m_t: (T,)
    Features per original feature dimension:
      mean, std, q10, q50, q90,
      mean_abs_diff, std_diff,
      half_mean_diff (first_half_mean - second_half_mean)
    Total dims = 8 * F
    """
    x_valid = _masked_valid(x_tf, m_t)
    T, F = x_tf.shape

    if x_valid is None or x_valid.shape[0] < 5:
        return np.zeros((8 * F,), dtype=np.float32)

    # basic stats per feature over time
    mu = x_valid.mean(axis=0)
    sd = x_valid.std(axis=0) + 1e-6
    q10 = np.quantile(x_valid, 0.10, axis=0)
    q50 = np.quantile(x_valid, 0.50, axis=0)
    q90 = np.quantile(x_valid, 0.90, axis=0)

    # temporal variability
    dx = np.diff(x_valid, axis=0)
    mean_abs_diff = np.mean(np.abs(dx), axis=0)
    std_diff = np.std(dx, axis=0) + 1e-6

    # first-half vs second-half
    # use original time axis split, but only valid points inside each half
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
    y01 = y / ymax
    y01 = np.clip(y01, eps, 1.0 - eps)
    return np.log(y01 / (1.0 - y01))


def z_to_y(z, ymax=0.05):
    y01 = 1.0 / (1.0 + np.exp(-z))
    return ymax * y01


# -----------------------
# plots
# -----------------------
def plot_scatter(y_true, y_pred, out_path, title):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad
    plt.plot([lo, hi], [lo, hi])
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
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", required=False, default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--target_col", default="lapse_rate_gt_500ms")
    ap.add_argument("--out_dir", default="./loso_lapse_hgb_opt")

    # bounds / transforms
    ap.add_argument("--ymax", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-4)

    # model params
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_iter", type=int, default=800)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--min_samples_leaf", type=int, default=3)

    # target requirement
    ap.add_argument("--thr", type=float, default=0.01, help="target abs error threshold")
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

    # Tabular features
    Xtab = np.stack([extract_subject_features(X[i], mask[i]) for i in range(N)], axis=0)

    # LOSO
    ymax = float(args.ymax)
    eps = float(args.eps)
    preds = np.zeros((N,), dtype=np.float32)

    for test_idx in range(N):
        train_idx = np.array([i for i in range(N) if i != test_idx], dtype=np.int64)

        # logit transform on y
        z_train = y_to_z(y[train_idx], ymax=ymax, eps=eps)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("hgb", HistGradientBoostingRegressor(
                loss="absolute_error",
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                max_iter=args.max_iter,
                l2_regularization=args.l2,
                min_samples_leaf=args.min_samples_leaf,
                random_state=7
            ))
        ])

        model.fit(Xtab[train_idx], z_train)
        z_pred = float(model.predict(Xtab[[test_idx]])[0])
        preds[test_idx] = float(z_to_y(z_pred, ymax=ymax))

    # Metrics
    abs_err = np.abs(preds - y)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds)) if N >= 3 else float("nan")
    pct_within = float(np.mean(abs_err <= args.thr))

    metrics = {
        "N": int(N),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        f"pct_within_{args.thr}": pct_within,
        "ymax": ymax,
        "model": "StandardScaler + HGBRegressor(loss=absolute_error) on logit(y/ymax)",
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
        "abs_error": abs_err
    }).to_csv(out_csv, index=False)

    # Visualizations
    scatter_path = os.path.join(args.out_dir, "scatter_true_vs_pred.png")
    plot_scatter(y, preds, scatter_path,
                 title=f"LOSO HGB+Logit | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}  P(|e|≤{args.thr})={pct_within:.3f}")

    bar_path = os.path.join(args.out_dir, "abs_error_bar.png")
    plot_abs_error_bar(subject_ids, abs_err, bar_path, thr=args.thr)

    cdf_path = os.path.join(args.out_dir, "error_cdf.png")
    plot_error_cdf(abs_err, cdf_path, thr=args.thr)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[DONE]")
    print(metrics)
    print("Saved:", out_csv)
    print("Saved:", scatter_path)
    print("Saved:", bar_path)
    print("Saved:", cdf_path)


if __name__ == "__main__":
    main()