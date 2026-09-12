#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def plot_confusion_matrix(cm, labels, title, out_path, normalize=True):
    cm = np.array(cm, dtype=np.float32)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        cmn = cm / rs
        data = cmn
        vmax = 1.0
    else:
        data = cm
        vmax = None

    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(data, vmin=0.0, vmax=vmax)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.yticks(ticks, labels)

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_reg_scatter(y_true, y_pred, title, out_path):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 3 else np.nan

    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(y_true, y_pred)

    # y=x line
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad
    plt.plot([lo, hi], [lo, hi])

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)

    plt.text(
        0.03, 0.97,
        f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}",
        transform=plt.gca().transAxes,
        ha="left", va="top"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2) if np.isfinite(r2) else np.nan}


def plot_task_summary_bar(df, out_path, metric="acc", title="Task Summary"):
    plt.figure(figsize=(8.5, 4.2))
    x = np.arange(len(df))
    y = df[metric].values.astype(np.float32)

    plt.bar(x, y)
    plt.xticks(x, df["task"].values, rotation=20, ha="right")
    plt.ylim(0, 1.0 if metric in ["acc", "bal_acc", "macro_f1"] else max(1.0, float(np.nanmax(y) + 1e-6)))
    plt.ylabel(metric)
    plt.title(title)

    for i, v in enumerate(y):
        if np.isfinite(v):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def detect_mode(run_dir):
    """
    Detect what kind of outputs exist in run_dir.
    Returns:
      - 'multi_task_cls' if predictions_*.csv exists
      - 'mtl' if predictions_oof.csv exists
      - 'reg' if predictions.csv exists
      - None otherwise
    """
    if glob.glob(os.path.join(run_dir, "predictions_*.csv")):
        return "multi_task_cls"
    if os.path.exists(os.path.join(run_dir, "predictions_oof.csv")):
        return "mtl"
    if os.path.exists(os.path.join(run_dir, "predictions.csv")):
        return "reg"
    return None


def visualize_multi_task_cls(run_dir, out_dir, normalize_cm=True, make_grid=True):
    """
    For directories produced by 4-task classification script:
      - predictions_<task>.csv
      - report_<task>.json (optional)
    """
    pred_files = sorted(glob.glob(os.path.join(run_dir, "predictions_*.csv")))
    if not pred_files:
        print(f"[WARN] No predictions_*.csv found in {run_dir}")
        return

    rows = []
    cm_for_grid = []
    labels_for_grid = []
    titles_for_grid = []

    for pf in pred_files:
        task = os.path.basename(pf).replace("predictions_", "").replace(".csv", "")
        df = pd.read_csv(pf)

        # Prefer original labels if present
        if "true_label_original" in df.columns:
            # keep ordering stable (numeric if possible)
            uniq = pd.unique(df["true_label_original"])
            try:
                labels = [str(x) for x in sorted(uniq, key=lambda z: float(z))]
            except Exception:
                labels = [str(x) for x in sorted(uniq, key=lambda z: str(z))]
        else:
            # fallback to encoded labels
            n_classes = int(max(df["true_cls"].max(), df["pred_cls"].max()) + 1)
            labels = [str(i) for i in range(n_classes)]

        # map to integer indices for cm
        y_true = df["true_cls"].values.astype(int)
        y_pred = df["pred_cls"].values.astype(int)

        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro"))

        title = f"{task} | acc={acc:.3f} bal={bal:.3f} mf1={mf1:.3f}"
        out_path = os.path.join(out_dir, f"cm_{task}.png")
        plot_confusion_matrix(cm, labels[:n_classes], title, out_path, normalize=normalize_cm)

        rows.append({"task": task, "acc": acc, "bal_acc": bal, "macro_f1": mf1, "cm_png": os.path.basename(out_path)})

        cm_for_grid.append(cm)
        labels_for_grid.append(labels[:n_classes])
        titles_for_grid.append(task)

        print(f"[OK] Saved {out_path}")

    # summary bar
    res = pd.DataFrame(rows).sort_values("task")
    res.to_csv(os.path.join(out_dir, "task_summary.csv"), index=False)
    plot_task_summary_bar(res, os.path.join(out_dir, "summary_acc_bar.png"), metric="acc",
                          title="Classification Acc (OOF)")
    plot_task_summary_bar(res, os.path.join(out_dir, "summary_macroF1_bar.png"), metric="macro_f1",
                          title="Classification Macro-F1 (OOF)")
    print(f"[OK] Saved {os.path.join(out_dir, 'task_summary.csv')}")

    # optional 2x2 grid
    if make_grid and len(cm_for_grid) >= 2:
        n = len(cm_for_grid)
        cols = 2
        rows_n = int(np.ceil(n / cols))
        plt.figure(figsize=(10, 4 * rows_n))

        for i in range(n):
            cm = cm_for_grid[i].astype(np.float32)
            if normalize_cm:
                rs = cm.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                cmn = cm / rs
                data = cmn
                vmax = 1.0
            else:
                data = cm
                vmax = None

            ax = plt.subplot(rows_n, cols, i + 1)
            ax.imshow(data, vmin=0.0, vmax=vmax)
            ax.set_title(titles_for_grid[i] + (" (norm)" if normalize_cm else ""))

            labels = labels_for_grid[i]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")

            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    txt = f"{data[r, c]:.2f}" if normalize_cm else f"{int(cm[r, c])}"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=9)

        plt.tight_layout()
        grid_path = os.path.join(out_dir, "confusion_matrices_grid.png")
        plt.savefig(grid_path, dpi=220)
        plt.close()
        print(f"[OK] Saved {grid_path}")


def visualize_mtl(run_dir, out_dir, normalize_cm=True):
    """
    For directories from MTL script:
      - predictions_oof.csv (subject_id,true_reg,pred_reg,true_cls,pred_cls)
      - overall_metrics.json (optional)
    """
    pred_path = os.path.join(run_dir, "predictions_oof.csv")
    if not os.path.exists(pred_path):
        print(f"[WARN] Not found: {pred_path}")
        return

    df = pd.read_csv(pred_path)

    # Regression scatter
    if "true_reg" in df.columns and "pred_reg" in df.columns:
        reg_title = "Regression (OOF): True vs Pred"
        reg_png = os.path.join(out_dir, "reg_scatter.png")
        metrics = plot_reg_scatter(df["true_reg"], df["pred_reg"], reg_title, reg_png)
        print(f"[OK] Saved {reg_png} | {metrics}")

    # Classification CM
    if "true_cls" in df.columns and "pred_cls" in df.columns:
        y_true = df["true_cls"].values.astype(int)
        y_pred = df["pred_cls"].values.astype(int)
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
        labels = [str(i) for i in range(n_classes)]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro"))

        title = f"Classification (OOF) | acc={acc:.3f} bal={bal:.3f} mf1={mf1:.3f}"
        cm_png = os.path.join(out_dir, "cls_confusion_matrix.png")
        plot_confusion_matrix(cm, labels, title, cm_png, normalize=normalize_cm)
        print(f"[OK] Saved {cm_png}")


def visualize_reg(run_dir, out_dir):
    """
    For old regression directories:
      - predictions.csv with columns: subject_id,true,pred
    """
    pred_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        print(f"[WARN] Not found: {pred_path}")
        return
    df = pd.read_csv(pred_path)

    # allow different column names
    if "true" in df.columns and "pred" in df.columns:
        y_true, y_pred = df["true"], df["pred"]
    elif "true_reg" in df.columns and "pred_reg" in df.columns:
        y_true, y_pred = df["true_reg"], df["pred_reg"]
    else:
        raise ValueError(f"Cannot find (true,pred) columns in {pred_path}. columns={list(df.columns)}")

    reg_png = os.path.join(out_dir, "reg_scatter.png")
    metrics = plot_reg_scatter(y_true, y_pred, "Regression (OOF): True vs Pred", reg_png)
    print(f"[OK] Saved {reg_png} | {metrics}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="directory that contains predictions_*.csv or predictions_oof.csv or predictions.csv")
    ap.add_argument("--out_dir", default="", help="where to save figures (default: <run_dir>/viz)")
    ap.add_argument("--normalize_cm", action="store_true", help="row-normalize confusion matrices")
    ap.add_argument("--grid", action="store_true", help="for multi-task classification, also save a grid image")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir.strip() or os.path.join(run_dir, "viz")
    ensure_dir(out_dir)

    mode = detect_mode(run_dir)
    if mode is None:
        raise RuntimeError(
            f"Cannot detect outputs in {run_dir}. "
            f"Need one of: predictions_*.csv / predictions_oof.csv / predictions.csv"
        )

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] detected mode: {mode}")

    if mode == "multi_task_cls":
        visualize_multi_task_cls(run_dir, out_dir, normalize_cm=args.normalize_cm, make_grid=args.grid)
    elif mode == "mtl":
        visualize_mtl(run_dir, out_dir, normalize_cm=args.normalize_cm)
    elif mode == "reg":
        visualize_reg(run_dir, out_dir)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    print("[DONE]")


if __name__ == "__main__":
    main()
