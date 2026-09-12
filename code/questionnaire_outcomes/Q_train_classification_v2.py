#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# -----------------------
# Dataset
# -----------------------
class SubjectSeqClsDataset(Dataset):
    def __init__(self, X, mask, y_cls):
        self.X = torch.from_numpy(X).float()       # (N,T,F)
        self.m = torch.from_numpy(mask).float()    # (N,T)
        self.y = torch.from_numpy(y_cls).long()    # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.m[idx], self.y[idx]


# -----------------------
# Model
# -----------------------
class TemporalCNN_BiGRU_Classifier(nn.Module):
    def __init__(self, in_dim, n_classes, cnn_channels=128, gru_hidden=128, gru_layers=1, dropout=0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, n_classes)
        )

    def forward(self, x, mask):
        # x: (B,T,F)
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B,T,C)
        h, _ = self.gru(z)                               # (B,T,2H)

        # masked mean pooling
        m = mask.unsqueeze(-1)                           # (B,T,1)
        h = h * m
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom                    # (B,2H)

        return self.head(pooled)                         # (B,n_classes)


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_classes(y_raw: pd.Series):
    vals = y_raw.values
    uniq = pd.unique(vals)
    try:
        uniq_sorted = sorted(uniq, key=lambda x: float(x))
    except Exception:
        uniq_sorted = sorted(uniq, key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(uniq_sorted)}
    inv_mapping = {i: v for v, i in mapping.items()}
    y_enc = np.array([mapping[v] for v in vals], dtype=np.int64)
    return y_enc, mapping, inv_mapping


def compute_class_weights(y_cls: np.ndarray, n_classes: int):
    counts = np.bincount(y_cls, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)   # inverse freq
    return torch.tensor(w, dtype=torch.float32)


def fit_standardizer(X_tr: np.ndarray, mask_tr: np.ndarray):
    """
    X_tr: (Ntr,T,F), mask_tr: (Ntr,T)
    compute mean/std per feature using only mask==1 samples
    """
    # flatten valid timesteps
    m = mask_tr.astype(bool)
    # avoid loops: collect sums and sq sums per feature
    # valid_count per feature = sum(m) across all subjects/timesteps
    count = m.sum()
    if count == 0:
        # fallback
        mu = X_tr.mean(axis=(0, 1))
        sd = X_tr.std(axis=(0, 1)) + 1e-6
        return mu.astype(np.float32), sd.astype(np.float32)

    # mask broadcast: (N,T,1)
    mb = m[..., None]
    s1 = (X_tr * mb).sum(axis=(0, 1))
    s2 = ((X_tr ** 2) * mb).sum(axis=(0, 1))
    mu = s1 / float(count)
    var = s2 / float(count) - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-6))
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return ((X - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)


def train_fixed_epochs(model, dl_tr, device, epochs, lr, wd, class_weights=None, label_smoothing=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=float(label_smoothing)
    )
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in dl_tr:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            logits = model(xb, mb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, dl_te, device):
    model.eval()
    y_true, y_pred = [], []
    for xb, mb, yb in dl_te:
        xb, mb = xb.to(device), mb.to(device)
        logits = model(xb, mb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(yb.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_accuracy_scatter(results_df: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(results_df))
    y = results_df["acc"].values
    plt.scatter(x, y)
    plt.ylim(0, 1.0)
    plt.xticks(x, results_df["task"].values, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title(title)
    for i, v in enumerate(y):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrices_2x2(cm_dict, class_labels_dict, out_path, normalize=True, title="Confusion Matrices (OOF)"):
    tasks = list(cm_dict.keys())
    n = len(tasks)
    # If not exactly 4, still try to plot in a grid
    cols = 2
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(10, 4 * rows))

    for i, task in enumerate(tasks, start=1):
        cm = np.array(cm_dict[task], dtype=np.float32)
        if normalize:
            row_sum = cm.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            cmn = cm / row_sum
        else:
            cmn = cm

        ax = plt.subplot(rows, cols, i)
        im = ax.imshow(cmn, vmin=0.0, vmax=1.0 if normalize else None)

        labels = class_labels_dict[task]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)

        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_title(task + (" (norm)" if normalize else ""))

        # annotate
        for r in range(cmn.shape[0]):
            for c in range(cmn.shape[1]):
                txt = f"{cmn[r,c]:.2f}" if normalize else f"{int(cm[r,c])}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=9)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=False, default="./nn_dataset_subject_level.npz", help="nn_dataset_subject_level_5min.npz")
    ap.add_argument("--label_xlsx", required=False, default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx", help="sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--out_dir", default="./cv_4task_cls_out_5mins")

    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--tasks", default="", help="comma-separated task columns; if empty=all non-id columns")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--gru_hidden", type=int, default=128)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.25)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load features
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float32)               # (N,T,F)
    mask = data["mask"].astype(np.float32)         # (N,T)
    subject_ids = data["subject_ids"].astype(str)  # (N,)

    # Load labels
    df = pd.read_excel(args.label_xlsx)
    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in xlsx columns: {list(df.columns)}")
    df[args.id_col] = df[args.id_col].astype(str)
    label_map = df.set_index(args.id_col)

    if args.tasks.strip():
        task_cols = [c.strip() for c in args.tasks.split(",") if c.strip()]
    else:
        task_cols = [c for c in df.columns if c != args.id_col]

    for c in task_cols:
        if c not in df.columns:
            raise ValueError(f"task '{c}' not found in xlsx columns: {list(df.columns)}")

    # Align subjects
    keep_idx, keep_subjects = [], []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep_idx.append(i)
            keep_subjects.append(sid)
    if len(keep_idx) == 0:
        raise ValueError("No overlapping subjects between npz subject_ids and xlsx subject column.")

    keep_idx = np.array(keep_idx, dtype=np.int64)
    X = X[keep_idx]
    mask = mask[keep_idx]
    subject_ids = np.array(keep_subjects, dtype=str)
    N = len(subject_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    cm_dict = {}
    class_labels_dict = {}

    for task in task_cols:
        y_raw = label_map.loc[subject_ids, task].reset_index(drop=True)
        y_cls, mapping, inv_mapping = encode_classes(y_raw)
        n_classes = int(y_cls.max() + 1)

        counts = pd.Series(y_cls).value_counts().sort_index()
        min_count = int(counts.min())
        k_task = min(args.k, min_count)
        if k_task < 2:
            raise ValueError(f"Task '{task}' has a class with <2 samples; cannot run CV. dist={dict(counts)}")

        skf = StratifiedKFold(n_splits=k_task, shuffle=True, random_state=args.seed)

        print(f"\n=== Task: {task} | N={N} | classes={n_classes} | k={k_task} | dist={dict(counts)} ===")

        oof_pred = np.zeros((N,), dtype=np.int64)

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(N), y_cls), start=1):
            tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)

            # fold-wise standardization (NO leakage)
            mu, sd = fit_standardizer(X[tr_idx], mask[tr_idx])
            X_tr = apply_standardizer(X[tr_idx], mu, sd)
            X_te = apply_standardizer(X[te_idx], mu, sd)

            ds_tr = SubjectSeqClsDataset(X_tr, mask[tr_idx], y_cls[tr_idx])
            ds_te = SubjectSeqClsDataset(X_te, mask[te_idx], y_cls[te_idx])

            bs = min(args.batch_size, len(ds_tr))
            dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
            dl_te = DataLoader(ds_te, batch_size=1, shuffle=False)

            class_w = compute_class_weights(y_cls[tr_idx], n_classes) if args.use_class_weights else None

            model = TemporalCNN_BiGRU_Classifier(
                in_dim=X.shape[-1],
                n_classes=n_classes,
                cnn_channels=args.cnn_channels,
                gru_hidden=args.gru_hidden,
                gru_layers=args.gru_layers,
                dropout=args.dropout
            ).to(device)

            model = train_fixed_epochs(
                model, dl_tr, device,
                epochs=args.epochs, lr=args.lr, wd=args.wd,
                class_weights=class_w,
                label_smoothing=args.label_smoothing
            )

            y_true, y_pred = predict(model, dl_te, device)
            oof_pred[te_idx] = y_pred

            acc = float(accuracy_score(y_true, y_pred))
            bal = float(balanced_accuracy_score(y_true, y_pred))
            mf1 = float(f1_score(y_true, y_pred, average="macro"))
            print(f"[Fold {fold}] acc={acc:.3f} bal_acc={bal:.3f} macro_f1={mf1:.3f} n_test={len(te_idx)}")

        # OOF metrics
        overall_acc = float(accuracy_score(y_cls, oof_pred))
        overall_bal = float(balanced_accuracy_score(y_cls, oof_pred))
        overall_mf1 = float(f1_score(y_cls, oof_pred, average="macro"))
        overall_cm = confusion_matrix(y_cls, oof_pred, labels=list(range(n_classes)))

        # class label names for plotting (original values)
        labels_for_plot = [str(inv_mapping[i]) for i in range(n_classes)]
        cm_dict[task] = overall_cm
        class_labels_dict[task] = labels_for_plot

        # save per-task predictions
        pred_csv = os.path.join(args.out_dir, f"predictions_{task.replace(' ', '_')}.csv")
        pd.DataFrame({
            "subject": subject_ids,
            "true_cls": y_cls,
            "pred_cls": oof_pred,
            "true_label_original": [labels_for_plot[int(v)] for v in y_cls],
            "pred_label_original": [labels_for_plot[int(v)] for v in oof_pred],
        }).to_csv(pred_csv, index=False)

        # save per-task report
        rep = {
            "task": task,
            "n_classes": n_classes,
            "dist": {str(k): int(v) for k, v in counts.to_dict().items()},
            "overall": {
                "acc": overall_acc,
                "bal_acc": overall_bal,
                "macro_f1": overall_mf1,
                "confusion_matrix": overall_cm.tolist()
            }
        }
        with open(os.path.join(args.out_dir, f"report_{task.replace(' ', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)

        all_results.append({
            "task": task,
            "n_classes": n_classes,
            "acc": overall_acc,
            "bal_acc": overall_bal,
            "macro_f1": overall_mf1,
            "k_used": k_task
        })

        print(f"[Task Overall] {task} acc={overall_acc:.3f} bal_acc={overall_bal:.3f} macro_f1={overall_mf1:.3f}")
        print(f"  saved: {pred_csv}")

    # summary
    results_df = pd.DataFrame(all_results).sort_values("task")
    results_csv = os.path.join(args.out_dir, "results.csv")
    results_df.to_csv(results_csv, index=False)

    # accuracy scatter
    fig_acc = os.path.join(args.out_dir, "accuracy_scatter.png")
    plot_accuracy_scatter(results_df, fig_acc, title="4 Questionnaire Classification Accuracy (OOF)")

    # confusion matrices 2x2 (normalized)
    fig_cm = os.path.join(args.out_dir, "confusion_matrices_4tasks.png")
    plot_confusion_matrices_2x2(
        cm_dict, class_labels_dict,
        fig_cm,
        normalize=True,
        title="Confusion Matrices (OOF, row-normalized)"
    )

    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {results_csv}")
    print(f"Saved: {fig_acc}")
    print(f"Saved: {fig_cm}")


if __name__ == "__main__":
    main()
