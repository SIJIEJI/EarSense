#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Confusion matrix plot
# -------------------------
def plot_confusion(cm, labels, title, out_path, normalize=True):
    cm = np.array(cm, dtype=np.float32)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        cm = cm / rs

    plt.figure(figsize=(6.8, 5.8))
    plt.imshow(cm, vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_loso_acc(subject_ids, correct, out_path, title="LOSO per-subject accuracy (0/1)"):
    correct = np.asarray(correct).astype(float)
    x = np.arange(len(subject_ids))
    plt.figure(figsize=(11, 3.8))
    plt.bar(x, correct)
    plt.xticks(x, subject_ids, rotation=60, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Correct (0/1)")
    plt.title(title)
    mean_acc = float(correct.mean()) if len(correct) else 0.0
    plt.axhline(mean_acc)
    plt.text(0.01, 0.95, f"Mean acc = {mean_acc:.3f}", transform=plt.gca().transAxes, ha="left", va="top")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


# -------------------------
# CORAL predict
# -------------------------
@torch.no_grad()
def coral_predict_class(logits_ord):
    # logits_ord: (B, K-1)
    p = torch.sigmoid(logits_ord)
    passed = (p > 0.5).sum(dim=1)
    return passed.long()


# -------------------------
# Model (same as training)
# -------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, c, k=5, dropout=0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad)
        self.bn2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return F.gelu(h + x)


class DeepOrdinalNet(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes=5,
        cnn_channels=384,
        n_res_blocks=6,
        gru_hidden=384,
        gru_layers=2,
        dropout=0.0,
        use_reg_head=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_reg_head = use_reg_head

        self.stem1 = nn.Sequential(
            nn.Conv1d(in_dim, cnn_channels, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        self.res = nn.Sequential(*[
            ResidualConvBlock(cnn_channels, k=5, dropout=dropout)
            for _ in range(n_res_blocks)
        ])

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.attn = nn.Linear(gru_hidden * 2, 1)

        self.ord_head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, num_classes - 1)
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1)
        )

    def _downsample_mask(self, mask, stride):
        m = mask.unsqueeze(1)  # (B,1,T)
        m = F.max_pool1d(m, kernel_size=stride, stride=stride, ceil_mode=True)
        return m.squeeze(1)

    def forward(self, x, mask):
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)

        x = self.stem1(x)
        mask = self._downsample_mask(mask, stride=2)

        x = self.stem2(x)
        mask = self._downsample_mask(mask, stride=2)

        x = self.res(x)        # (B,C,T')
        x = x.transpose(1, 2)  # (B,T',C)

        h, _ = self.gru(x)     # (B,T',2H)

        scores = self.attn(h).squeeze(-1)  # (B,T')
        scores = scores.masked_fill(mask <= 0.0, -1e9)
        w = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)        # (B,2H)

        logits_ord = self.ord_head(pooled) # (B,K-1)
        pred_reg = self.reg_head(pooled).squeeze(-1) if self.use_reg_head else None
        return logits_ord, pred_reg


# -------------------------
# Mapping helpers
# -------------------------
def _norm_key(v):
    """Make excel raw label robustly match ckpt mapping keys (strings)."""
    if v is None:
        return None
    # numpy scalar
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if abs(fv - round(fv)) < 1e-6:
            return str(int(round(fv)))
        return str(fv)
    # python number
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, (float,)):
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return str(v)
    return str(v)


def encode_with_ckpt_mapping(y_raw, raw_to_idx):
    y_idx = []
    for v in y_raw:
        k = _norm_key(v)
        if k in raw_to_idx:
            y_idx.append(int(raw_to_idx[k]))
        else:
            # fallback: try raw string
            ks = str(v)
            if ks in raw_to_idx:
                y_idx.append(int(raw_to_idx[ks]))
            else:
                raise KeyError(f"Label value '{v}' not found in ckpt mapping keys: {list(raw_to_idx.keys())[:10]} ...")
    return np.array(y_idx, dtype=np.int64)


# -------------------------
# Checkpoint load
# -------------------------
def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    mean = np.array(ckpt.get("mean"), dtype=np.float32) if "mean" in ckpt else None
    std = np.array(ckpt.get("std"), dtype=np.float32) if "std" in ckpt else None
    num_classes = int(ckpt.get("num_classes"))

    # model args used during training
    tr_args = ckpt.get("args", {})
    model_kwargs = dict(
        num_classes=num_classes,
        cnn_channels=int(tr_args.get("cnn_channels", 384)),
        n_res_blocks=int(tr_args.get("n_res_blocks", 6)),
        gru_hidden=int(tr_args.get("gru_hidden", 384)),
        gru_layers=int(tr_args.get("gru_layers", 2)),
        dropout=float(tr_args.get("dropout", 0.0)),
        use_reg_head=bool(tr_args.get("use_reg_head", False))
    )

    raw_to_idx = ckpt.get("class_mapping_raw_to_idx", None)
    idx_to_raw = ckpt.get("class_mapping_idx_to_raw", None)

    return ckpt["model_state"], mean, std, num_classes, model_kwargs, raw_to_idx, idx_to_raw


def normalize_x(x, mean, std):
    if mean is None or std is None:
        return x.astype(np.float32)
    return ((x - mean[None, :]) / (std[None, :] + 1e-6)).astype(np.float32)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--task_col", default="PSQI_class")

    # One checkpoint for all subjects (NOT strict LOOSO if trained on all)
    ap.add_argument("--ckpt", default="./PSQI_overfit_showcase/checkpoint_overfit_train.pt")

    # Optional: strict LOOSO mode: one ckpt per subject (trained with that subject held out)
    # Example: ./checkpoints/ckpt_loso_{subject}.pt
    ap.add_argument("--ckpt_pattern", default="", help="optional pattern with {subject} placeholder for strict LOOSO")

    ap.add_argument("--out_dir", default="./PSQI_loso_eval")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--normalize_cm", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)           # (N,T,F)
    mask = d["mask"].astype(np.float32)     # (N,T)
    subject_ids = d["subject_ids"].astype(str)

    # Load labels
    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    if args.task_col not in df.columns:
        raise ValueError(f"{args.task_col} not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[args.task_col])
    label_map = df.set_index(args.id_col)

    keep = []
    y_raw = []
    kept_subjects = []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_raw.append(label_map.loc[sid, args.task_col])
            kept_subjects.append(sid)

    keep = np.array(keep, dtype=np.int64)
    if len(keep) == 0:
        raise ValueError("No overlapping subjects between npz subject_ids and xlsx subject column.")

    X = X[keep]
    mask = mask[keep]
    subject_ids = np.array(kept_subjects, dtype=str)
    y_raw = np.array(y_raw)

    N, T, Fdim = X.shape
    print("N subjects for eval:", N)

    # Cache checkpoints if pattern mode loads many
    ckpt_cache = {}

    def get_ckpt_path_for_subject(sid):
        if args.ckpt_pattern.strip():
            return args.ckpt_pattern.format(subject=sid)
        return args.ckpt

    preds = np.zeros((N,), dtype=np.int64)
    trues = None
    correct = np.zeros((N,), dtype=np.int64)

    # For labels -> idx mapping, we rely on checkpoint mapping.
    # If ckpt_pattern is used, each ckpt may have same mapping; we assume consistent.
    # We'll load mapping from first ckpt.
    first_ckpt_path = get_ckpt_path_for_subject(subject_ids[0])
    if not os.path.exists(first_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {first_ckpt_path}")

    model_state, mean, std, num_classes, model_kwargs, raw_to_idx, idx_to_raw = load_checkpoint(first_ckpt_path, device)

    if raw_to_idx is None:
        raise ValueError("Checkpoint does not contain class_mapping_raw_to_idx. Please save mapping in ckpt.")

    trues = encode_with_ckpt_mapping(y_raw, raw_to_idx)
    labels_show = [str(idx_to_raw[str(i)]) if (idx_to_raw and str(i) in idx_to_raw) else str(i) for i in range(num_classes)]

    # Warn if not strict LOOSO
    if not args.ckpt_pattern.strip():
        print("\n[WARN] You are using ONE checkpoint for all subjects.")
        print("       If that checkpoint was trained on all subjects, this is NOT strict LOOSO.")
        print("       For strict LOOSO, provide --ckpt_pattern with per-subject checkpoints.\n")

    for i, sid in enumerate(subject_ids):
        ckpt_path = get_ckpt_path_for_subject(sid)
        if ckpt_path not in ckpt_cache:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found for subject '{sid}': {ckpt_path}")
            ckpt_cache[ckpt_path] = load_checkpoint(ckpt_path, device)

        model_state, mean, std, num_classes_i, model_kwargs_i, raw_to_idx_i, idx_to_raw_i = ckpt_cache[ckpt_path]
        if num_classes_i != num_classes:
            raise ValueError(f"num_classes mismatch across checkpoints: {num_classes_i} vs {num_classes}")

        # Build model
        model = DeepOrdinalNet(in_dim=Fdim, **model_kwargs_i).to(device)
        model.load_state_dict(model_state, strict=True)
        model.eval()

        # Prepare one sample
        x = normalize_x(X[i], mean, std)  # (T,F)
        m = mask[i]                       # (T,)
        xb = torch.from_numpy(x[None]).float().to(device)   # (1,T,F)
        mb = torch.from_numpy(m[None]).float().to(device)   # (1,T)

        with torch.no_grad():
            logits_ord, _ = model(xb, mb)
            pred = coral_predict_class(logits_ord).cpu().numpy()[0]

        preds[i] = int(pred)
        correct[i] = int(preds[i] == trues[i])

        print(f"[{i+1:02d}/{N}] subject={sid}  true={trues[i]}  pred={preds[i]}  correct={correct[i]}")

    # Overall metrics
    acc = float(accuracy_score(trues, preds))
    bal = float(balanced_accuracy_score(trues, preds))
    mf1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))

    # Save CSV
    out_csv = os.path.join(args.out_dir, "predictions_loso.csv")
    rows = []
    for i in range(N):
        rows.append({
            "subject_id": subject_ids[i],
            "true_idx": int(trues[i]),
            "pred_idx": int(preds[i]),
            "true_raw": labels_show[int(trues[i])] if int(trues[i]) < len(labels_show) else str(trues[i]),
            "pred_raw": labels_show[int(preds[i])] if int(preds[i]) < len(labels_show) else str(preds[i]),
            "correct": int(correct[i])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plots
    cm_path = os.path.join(args.out_dir, "confusion_matrix_loso.png")
    # show confusion matrix with percentages, but also show raw counts for reference (normalize=False)
    plot_confusion(
        cm, labels_show,
        f"LOSO eval | acc={acc:.3f} bal={bal:.3f} macroF1={mf1:.3f}",
        cm_path,
        normalize=args.normalize_cm
    )

    bar_path = os.path.join(args.out_dir, "loso_subject_acc.png")
    plot_loso_acc(subject_ids, correct, bar_path, title=f"LOSO per-subject correctness (mean acc={acc:.3f})")

    # Metrics JSON
    metrics = {
        "N": int(N),
        "acc": acc,
        "bal_acc": bal,
        "macro_f1": mf1,
        "num_classes": int(num_classes),
        "labels_show": labels_show,
        "note": "If using a single ckpt trained on all subjects, this is NOT strict LOOSO."
                if not args.ckpt_pattern.strip() else "Strict LOOSO (per-subject checkpoint)."
    }
    with open(os.path.join(args.out_dir, "metrics_loso.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n[OVERALL]")
    print(metrics)
    print("Saved:", out_csv)
    print("Saved:", cm_path)
    print("Saved:", bar_path)


if __name__ == "__main__":
    main()