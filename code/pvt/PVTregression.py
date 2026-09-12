#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_percent_cell(v):
    """
    Convert '1.52%' -> 1.52 (float), '0.43' -> 0.43, numeric -> float.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, np.integer, float, np.floating)):
        return float(v)
    s = str(v).strip()
    s = s.replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if m:
        return float(m.group(1))
    return np.nan


def masked_mean_std(X, mask):
    """
    X: (N,T,F), mask: (N,T)
    -> mean/std over valid timesteps -> (N,F), (N,F)
    """
    m = mask.astype(bool)[..., None]        # (N,T,1)
    cnt = m.sum(axis=1).clip(min=1)         # (N,1)
    mu = (X * m).sum(axis=1) / cnt          # (N,F)
    mu2 = ((X**2) * m).sum(axis=1) / cnt
    var = np.maximum(mu2 - mu**2, 1e-6)
    sd = np.sqrt(var)
    return mu.astype(np.float32), sd.astype(np.float32)


def plot_scatter(y_true, y_pred, out_path, title):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad
    plt.plot([lo, hi], [lo, hi])
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True lapse_rate_gt_500ms (%)")
    plt.ylabel("Pred (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_abs_error(subject_ids, y_true, y_pred, out_path):
    err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    order = np.argsort(err)[::-1]
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(err)), err[order])
    plt.xticks(np.arange(len(err)), np.array(subject_ids)[order], rotation=60, ha="right")
    plt.ylabel("|Pred-True| (%)")
    plt.title("Absolute Error per Subject (LOSO)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", required=False, default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--target_col", default="lapse_rate_gt_500ms")
    ap.add_argument("--out_dir", default="./loso_lapse_reg")
    ap.add_argument("--alpha", type=float, default=10.0, help="Ridge alpha")
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
            val = parse_percent_cell(label_map.loc[sid, args.target_col])
            if np.isnan(val):
                continue
            keep.append(i)
            y.append(val)
            kept_subjects.append(sid)

    keep = np.array(keep, dtype=np.int64)
    if len(keep) < 5:
        raise ValueError(f"Too few matched subjects: {len(keep)}. Check id alignment / NaNs.")

    X = X[keep]
    mask = mask[keep]
    subject_ids = np.array(kept_subjects, dtype=str)
    y = np.array(y, dtype=np.float32)

    # Tabular features: mean/std over time for each of 33 features -> 66 dims
    mu, sd = masked_mean_std(X, mask)            # (N,33)
    Xtab = np.concatenate([mu, sd], axis=1)      # (N,66)

    N = len(y)
    preds = np.zeros((N,), dtype=np.float32)

    # LOSO
    for test_idx in range(N):
        train_idx = np.array([i for i in range(N) if i != test_idx], dtype=np.int64)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha))
        ])

        model.fit(Xtab[train_idx], y[train_idx])
        preds[test_idx] = float(model.predict(Xtab[[test_idx]])[0])
        pred = np.clip(preds[test_idx], 0.0, 0.05)  
        preds[test_idx] = pred

    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds)) if N >= 3 else float("nan")

    # Save predictions
    out_csv = os.path.join(args.out_dir, "predictions_loso.csv")
    pd.DataFrame({
        "subject_id": subject_ids,
        "true": y,
        "pred": preds,
        "abs_error": np.abs(preds - y)
    }).to_csv(out_csv, index=False)

    # Plots
    scatter_path = os.path.join(args.out_dir, "scatter_true_vs_pred.png")
    plot_scatter(y, preds, scatter_path, title=f"LOSO Ridge | MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

    bar_path = os.path.join(args.out_dir, "abs_error_bar.png")
    plot_abs_error(subject_ids, y, preds, bar_path)

    metrics = {"N": int(N), "MAE": mae, "RMSE": rmse, "R2": r2, "model": "StandardScaler + Ridge", "alpha": args.alpha}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[DONE]")
    print(metrics)
    print("Saved:", out_csv)
    print("Saved:", scatter_path)
    print("Saved:", bar_path)


if __name__ == "__main__":
    main()